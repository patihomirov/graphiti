"""Queue service for managing episode processing."""

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from config.schema import ResilienceConfig
from services.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    QueueCapacityExceeded,
    is_transient_error,
)
from services.episode_spool import EpisodeSpool
from services.queue_journal import JournalRetryer, QueueJournal

logger = logging.getLogger(__name__)


class ServerStoppingError(Exception):
    """Raised when an episode is submitted while the server is stopping.

    The episode is NOT queued and NOT lost: the caller receives a
    ``graphiti_backpressure:`` response and is expected to retry after the
    server has restarted (pending episodes are spilled to the disk spool).
    """


class QueueService:
    """Service for managing sequential episode processing queues by group_id.

    The write path is protected by a circuit breaker and bounded by a max queue
    depth. When the breaker is open or the depth is exhausted, new episodes are
    rejected with ``CircuitOpenError`` / ``QueueCapacityExceeded`` (fail fast)
    instead of silently piling into an in-memory queue that would be lost on
    restart. Episodes that fail during processing are persisted to a disk spool
    so they can be replayed by a background retryer.
    """

    def __init__(self, resilience: ResilienceConfig | None = None):
        """Initialize the queue service.

        Args:
            resilience: Optional resilience config. Defaults to a config with
                all stock defaults so callers that omit the argument keep working.
        """
        self.resilience: ResilienceConfig = resilience or ResilienceConfig()
        self._breaker: CircuitBreaker | None = None
        self._spool: EpisodeSpool | None = None

        # Durable SQLite journal (source of truth for the write path when
        # resilience.journal_enabled is true). Constructed eagerly: a failure to
        # open it is fatal (fail-closed, D5) rather than a silent fallback to
        # the lossy in-memory queue.
        self._journal: QueueJournal | None = None
        if self.resilience.journal_enabled:
            journal_path = self.resilience.journal_path or '~/.graphiti/journal.db'
            self._journal = QueueJournal(
                journal_path,
                lease_seconds=self.resilience.journal_lease_seconds,
            )
        # Alarm that wakes group workers when journal rows become due for retry.
        self._journal_retryer: JournalRetryer | None = None
        # Sync view of journal-backed per-group unfinished counts (legacy API).
        self._journal_queue_sizes: dict[str, int] = {}
        # Rebuilds graphiti.add_episode kwargs from a persisted journal plan.
        self._episode_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        # Claim owner tag for journal rows (f'p<pid>'), a phase-2 lease hook.
        self._worker_id = f'p{os.getpid()}'
        # How long a journal worker sleeps between no-work wake checks.
        self._journal_idle_timeout = 30.0

        # Global journal worker pool (phase 2). ``journal_workers`` tasks serve
        # ALL groups: a worker picks a group with due rows, claims exactly one
        # row under that group's processing lock, then repeats. The per-group
        # lock keeps within-group processing strictly FIFO and never
        # parallelizes serial zones.
        self._pool_tasks: list[asyncio.Task] = []
        self._pool_started = False
        self._group_locks: dict[str, asyncio.Lock] = {}
        # Pool wake primitive: a sticky event + generation counter (a bare
        # event.clear() lost wakeups under N workers).
        self._wake_event = asyncio.Event()
        self._wake_generation = 0
        self._poll_interval = 0.05
        # Global cap on concurrent add_episode calls. initialize() may replace it
        # with the server-wide semaphore so the pool and the direct path share
        # one LLM concurrency limit.
        self._llm_semaphore = asyncio.Semaphore(int(self.resilience.semaphore_limit))
        # Periodic lease steward (phase 2): requeues rows whose owner vanished,
        # while never stealing this process's own in-flight rows.
        self._steward_task: asyncio.Task | None = None

        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Worker task handles per group_id, used to cancel in-flight episodes
        # during a graceful drain.
        self._worker_tasks: dict[str, asyncio.Task] = {}
        # In-flight episode count per group. A counter (not a bool) so
        # ``_any_busy`` sees every concurrent in-flight episode, even if a
        # group were ever processed by more than one worker.
        self._busy_counts: dict[str, int] = {}
        # Flipped by the graceful shutdown coordinator on SIGTERM/SIGINT.
        self._stopping: bool = False
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None

        # Total number of episodes currently queued or being processed across all
        # group_ids. Guarded by _depth_lock so concurrent submissions do not blow
        # past max_queue_depth.
        self._queue_depth: int = 0
        self._depth_lock = asyncio.Lock()

        if self.resilience.enabled:
            self._breaker = CircuitBreaker(
                failure_threshold=self.resilience.failure_threshold,
                open_timeout_seconds=self.resilience.open_timeout_seconds,
                probe_timeout_seconds=self.resilience.probe_timeout_seconds,
            )
            if self.resilience.spool_enabled:
                self._spool = EpisodeSpool(
                    self.resilience.spool_dir,
                    backoff_base_seconds=self.resilience.spool_backoff_base_seconds,
                )

    async def add_episode_task(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
        plan: dict[str, Any] | None = None,
    ) -> int:
        """Add an episode processing task to the queue.

        Applies fail-fast backpressure *before* queueing: if the circuit breaker
        is open or the total queue depth has reached ``max_queue_depth``, the
        episode is rejected (``CircuitOpenError`` / ``QueueCapacityExceeded``)
        rather than being accepted into a queue that could be lost on restart.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode
            plan: Optional serializable episode plan used to spool on failure

        Returns:
            The position in the queue

        Raises:
            ServerStoppingError: If the server is shutting down (fail fast).
            CircuitOpenError: If the circuit breaker is open (fail fast).
            QueueCapacityExceeded: If the queue depth limit is reached (fail fast).
        """
        if self._stopping:
            raise ServerStoppingError(
                'Server is stopping (graceful drain in progress). '
                'Episode NOT queued; retry after the server restarts.'
            )

        # Journal-backed path: the plan is persisted atomically to SQLite (the
        # source of truth) BEFORE any in-memory state changes, so it survives a
        # hard kill. Backpressure is computed from journal rows so it stays
        # correct after a restart.
        if self._journal is not None and plan is not None:
            return await self._add_episode_task_journal(group_id, plan)

        # Legacy path (journal disabled, or a bare process_func with plan=None):
        # the in-memory queue remains the notification mechanism.
        return await self._add_episode_task_legacy(group_id, process_func, plan)

    async def _add_episode_task_journal(self, group_id: str, plan: dict[str, Any]) -> int:
        """Journal-backed add_episode_task: persist the plan, then wake a worker.

        Fail-fast backpressure (stopping, breaker, capacity) is applied *before*
        the atomic INSERT. The in-memory queue is NOT used for real episodes;
        workers claim rows directly from the journal table (claim semantics).
        """
        async with self._depth_lock:
            if self._breaker is not None:
                allowed = await self._breaker.allow_request()
                if not allowed:
                    snap = await self._breaker.get_snapshot()
                    raise CircuitOpenError(
                        f'Circuit is open (state={snap["state"]}). '
                        f'Rejecting episode; retry in ~{snap["retry_after_seconds"]}s.'
                    )
            unfinished = await self._journal.count_unfinished()
            if unfinished >= self.resilience.max_queue_depth:
                raise QueueCapacityExceeded(
                    f'Queue depth {unfinished} >= max {self.resilience.max_queue_depth}. '
                    'Rejecting episode; retry once the queue drains.'
                )
            _, inserted = await self._journal.enqueue(plan)

        if not inserted:
            logger.info(
                'Duplicate episode %s (name=%s) skipped by journal dedup',
                plan.get('uuid'),
                plan.get('name'),
            )

        await self._wake_group(group_id)
        size = await self._journal.count_unfinished_group(group_id)
        self._journal_queue_sizes[group_id] = size
        return size

    async def _add_episode_task_legacy(
        self,
        group_id: str,
        process_func: Callable[[], Awaitable[None]],
        plan: dict[str, Any] | None = None,
    ) -> int:
        """Legacy in-memory add_episode_task (journal disabled / plan=None)."""
        if self.resilience.enabled:
            async with self._depth_lock:
                if self._breaker is not None:
                    allowed = await self._breaker.allow_request()
                    if not allowed:
                        snap = await self._breaker.get_snapshot()
                        raise CircuitOpenError(
                            f'Circuit is open (state={snap["state"]}). '
                            f'Rejecting episode; retry in ~{snap["retry_after_seconds"]}s.'
                        )
                if self._queue_depth >= self.resilience.max_queue_depth:
                    raise QueueCapacityExceeded(
                        f'Queue depth {self._queue_depth} >= max {self.resilience.max_queue_depth}. '
                        'Rejecting episode; retry once the queue drains.'
                    )

                # Initialize queue for this group_id if it doesn't exist
                if group_id not in self._episode_queues:
                    self._episode_queues[group_id] = asyncio.Queue()

                # Queue holds (process_func, plan) so the worker can spool on failure.
                await self._episode_queues[group_id].put((process_func, plan))
                self._queue_depth += 1

                # Start a worker for this queue if one isn't already running.
                # Claim the worker slot *before* scheduling so concurrent submits
                # cannot spawn duplicate workers for the same group_id.
                if not self._queue_workers.get(group_id, False):
                    self._queue_workers[group_id] = True
                    task = asyncio.create_task(self._process_episode_queue(group_id))
                    self._worker_tasks[group_id] = task

                return self._episode_queues[group_id].qsize()

        # Resilience disabled: preserve the original, unbounded behaviour.
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        await self._episode_queues[group_id].put((process_func, plan))

        # Claim the worker slot before scheduling (same reasoning as above).
        if not self._queue_workers.get(group_id, False):
            self._queue_workers[group_id] = True
            task = asyncio.create_task(self._process_episode_queue(group_id))
            self._worker_tasks[group_id] = task

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a group sequentially (legacy in-memory mode).

        Journal-backed mode is served by the global worker pool
        (``_journal_pool_worker``); this loop only ever drains the legacy
        in-memory queues.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                await self._process_legacy_once(group_id)
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    async def _process_legacy_once(self, group_id: str) -> bool:
        """Handle a single episode from the in-memory queue (legacy mode)."""
        # Get the next episode processing tuple from the queue (waits if empty).
        process_func, plan = await self._episode_queues[group_id].get()

        try:
            # Mark the worker busy so the graceful drain knows this
            # episode is in flight (not merely pending in the queue).
            self._busy_counts[group_id] = self._busy_counts.get(group_id, 0) + 1
            # Process the episode
            await process_func()
        except asyncio.CancelledError:
            # The graceful drain cancels workers whose in-flight episode
            # did not finish within the grace window. Spool it so the
            # retryer replays it after restart instead of losing it.
            if self._stopping and self._spool is not None and plan is not None:
                try:
                    self._spool.save(plan, reason='cancelled during graceful drain')
                    logger.warning(
                        'Graceful drain: spooled in-flight episode %s (name=%s) of group %s',
                        plan.get('uuid'),
                        plan.get('name'),
                        group_id,
                    )
                except Exception as se:
                    logger.error(f'Failed to spool cancelled episode {plan.get("uuid")}: {se}')
            raise
        except Exception as e:
            await self._handle_processing_failure(group_id, plan, e)
        else:
            if self._breaker is not None:
                await self._breaker.record_success()
        finally:
            count = self._busy_counts.get(group_id, 0) - 1
            if count > 0:
                self._busy_counts[group_id] = count
            else:
                self._busy_counts.pop(group_id, None)
            # Decrement the shared depth invariant and mark done regardless
            # of success/failure.
            self._queue_depth = max(0, self._queue_depth - 1)
            self._episode_queues[group_id].task_done()
        return True

    async def _process_journal_once(self, group_id: str) -> bool:
        """Claim and process a single journal row for a group. Returns False if none."""
        row = await self._journal.claim_next(group_id, self._worker_id)
        if row is None:
            return False

        self._busy_counts[group_id] = self._busy_counts.get(group_id, 0) + 1
        # Keep the lease fresh for long in-flight LLM calls so the steward never
        # steals our own row; cancelled in ``finally`` once the row is done.
        heartbeat = asyncio.create_task(self._journal_heartbeat_loop(row['id']))
        try:
            plan = json.loads(row['plan_json'])
            if self._episode_builder is None:  # guarded by caller, defensive
                raise RuntimeError('No episode_builder configured for journal processing')
            kwargs = self._episode_builder(plan)
            # The pool bounds concurrency per group, but the LLM + graph write is
            # the scarce resource shared with the direct path: hold the global
            # semaphore around the actual add_episode call.
            async with self._llm_semaphore:
                await self._graphiti_client.add_episode(**kwargs)
        except asyncio.CancelledError:
            # The row stays 'processing'; requeue_stale() on the next boot
            # (or a stale-lease cleanup) heals it after the hard kill.
            raise
        except Exception as e:
            await self._handle_journal_failure(group_id, row, e)
        else:
            await self._journal.mark_done(row['id'])
            if self._breaker is not None:
                await self._breaker.record_success()
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            count = self._busy_counts.get(group_id, 0) - 1
            if count > 0:
                self._busy_counts[group_id] = count
            else:
                self._busy_counts.pop(group_id, None)
            size = await self._journal.count_unfinished_group(group_id)
            self._journal_queue_sizes[group_id] = size
        return True

    async def _journal_pool_worker(self) -> None:
        """Serve journal groups from the global worker pool.

        A worker picks a group with due rows whose processing lock is free,
        claims and processes exactly one row under that lock, then repeats. The
        per-group lock guarantees at most one in-flight episode per group, so
        within-group order stays strictly FIFO (claim by id) and serial zones
        (auto-previous context / saga chains / update_communities) are never
        parallelized - the pool size only adds cross-group concurrency.
        """
        try:
            while True:
                if self._stopping:
                    return
                if self._episode_builder is None:
                    await self._journal_wait_global()
                    continue
                group_id = await self._pick_group()
                if group_id is None:
                    await self._journal_wait_global()
                    continue
                lock = self._group_locks.setdefault(group_id, asyncio.Lock())
                if not await lock.acquire():
                    # Another worker holds this group's lock; re-scan the pool.
                    await self._journal_wait_global()
                    continue
                try:
                    await self._process_journal_once(group_id)
                finally:
                    lock.release()
        except asyncio.CancelledError:
            logger.info('Journal pool worker cancelled')
            raise
        except Exception as e:
            logger.error('Unexpected error in journal pool worker: %s', e)

    async def _pick_group(self) -> str | None:
        """Return the first group with due rows whose lock is currently free."""
        digest = await self._journal.due_digest()
        for group_id in digest:
            lock = self._group_locks.get(group_id)
            if lock is None or not lock.locked():
                return group_id
        return None

    async def _has_due_work(self) -> bool:
        """Whether any group currently has due rows this worker could claim.

        Considers *pending-and-due* rows only, so a worker whose group is
        mid-flight (row ``processing``) idles instead of hot-polling.
        """
        return await self._pick_group() is not None

    async def _journal_wait_global(self) -> None:
        """Idle a pool worker until the pool is woken or the idle timeout hits.

        The wake event is cleared only after verifying there is no claimable
        work, with the queue re-checked right after the clear and again after
        any wake, so a wake that lands mid-wait is never lost (a generation
        counter pins it). This replaces the old per-group ``_journal_wait``,
        whose ``event.clear()`` race was unsafe with N workers.
        """
        while True:
            if self._stopping:
                return
            if await self._has_due_work():
                return
            event = self._wake_event
            gen = self._wake_generation
            event.clear()
            if await self._has_due_work():
                return
            try:
                await asyncio.wait_for(event.wait(), timeout=self._journal_idle_timeout)
            except asyncio.TimeoutError:
                return
            if gen != self._wake_generation:
                return
            if await self._has_due_work():
                return
            # The event was set by a wake already consumed by another worker and
            # the generation has not advanced: nothing new is due. Step back
            # briefly rather than spinning, then re-check from the top.
            await asyncio.sleep(self._poll_interval)

    async def _wake_group(self, group_id: str) -> None:
        """Notify the journal worker pool that work may be available."""
        self._notify_pool()
        self._ensure_pool()

    async def _journal_heartbeat_loop(self, row_id: int) -> None:
        """Renew a claimed row's lease while its add_episode is in flight.

        Runs as a side task per in-flight row and is cancelled once the row is
        finished; the lease outlives a long LLM call, so the steward cannot
        reclaim it and no two workers process the same episode.
        """
        interval = max(0.1, self._journal.lease_seconds / 2.0)
        try:
            while True:
                await asyncio.sleep(interval)
                alive = await self._journal.heartbeat(row_id, self._worker_id)
                if not alive:
                    logger.warning(
                        'Lease lost on journal row %s (worker %s); stopping heartbeat',
                        row_id,
                        self._worker_id,
                    )
                    return
        except asyncio.CancelledError:
            return

    async def _journal_steward(self) -> None:
        """Periodically requeue processing rows whose owner vanished.

        Uses ``requeue_stale_except`` so rows owned by this process are never
        stolen - long in-flight LLM calls are not double-processed; a vanished
        owner's rows (lease expired beyond ``journal_grace_seconds``) are moved
        back to pending and the pool is woken to retry them.
        """
        interval = max(1.0, float(self.resilience.journal_steward_interval_seconds))
        while True:
            if self._stopping:
                return
            await asyncio.sleep(interval)
            try:
                stolen = await self._journal.requeue_stale_except(
                    self._worker_id, self.resilience.journal_grace_seconds
                )
                if stolen:
                    logger.warning(
                        'Lease steward requeued %d row(s) whose owner vanished',
                        stolen,
                    )
                    self._notify_pool()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error('Lease steward error: %s', e)

    def _notify_pool(self) -> None:
        """Wake the journal pool: bump the generation and set the sticky event."""
        self._wake_generation += 1
        self._wake_event.set()

    def _ensure_pool(self) -> None:
        """Start the global journal worker pool (idempotent)."""
        if self._pool_started or self._journal is None:
            return
        self._pool_started = True
        n = max(1, int(self.resilience.journal_workers))
        for i in range(1, n + 1):
            task = asyncio.create_task(
                self._journal_pool_worker(),
                name=f'journal-pool-worker-{i}',
            )
            self._pool_tasks.append(task)
        logger.info('Started %d journal pool worker(s)', n)

    async def _handle_journal_failure(
        self, group_id: str, row: dict[str, Any], exc: BaseException
    ) -> None:
        """React to a failed journal-processed episode: trip the breaker, back off.

        Transient failures trip the breaker; the row is kept ``pending`` with an
        exponential ``next_retry_at`` (the JournalRetryer alarm wakes the worker
        when it is due). Exhausted rows are moved to ``failed`` like the spool.
        """
        transient = is_transient_error(exc) if self._breaker is not None else False
        if transient and self._breaker is not None:
            await self._breaker.record_failure(exc)

        row_id = int(row['id'])
        attempt = int(row.get('attempt', 0)) + 1
        if attempt >= self.resilience.max_spool_attempts:
            await self._journal.mark_failed(row_id, error=str(exc))
            logger.error(
                'Episode %s (name=%s) for group %s gave up after %d attempts: %s',
                row.get('uuid'),
                row.get('name'),
                group_id,
                attempt,
                exc,
            )
            return

        backoff = self.resilience.spool_backoff_base_seconds * (2 ** (attempt - 1))
        next_retry_at = (
            datetime.now(timezone.utc) + timedelta(seconds=backoff)
        ).isoformat()
        await self._journal.bump_retry(
            row_id, attempt=attempt, error=str(exc), next_retry_at=next_retry_at
        )
        logger.warning(
            'Retry attempt %d/%d (backoff %.1fs) for episode %s (name=%s) group %s: %s',
            attempt,
            self.resilience.max_spool_attempts,
            backoff,
            row.get('uuid'),
            row.get('name'),
            group_id,
            exc,
        )

    async def _handle_processing_failure(
        self, group_id: str, plan: dict[str, Any] | None, exc: BaseException
    ) -> None:
        """React to a failed episode: trip the breaker and spool so nothing is lost."""
        transient = is_transient_error(exc) if self._breaker is not None else False

        if transient and self._breaker is not None:
            await self._breaker.record_failure(exc)
            logger.error(
                f'Transient failure processing episode {plan.get("uuid") if plan else None} '
                f'(name={plan.get("name") if plan else None}) for group {group_id}: {exc}'
            )
        else:
            logger.error(
                f'Permanent failure processing episode {plan.get("uuid") if plan else None} '
                f'(name={plan.get("name") if plan else None}) for group {group_id}: {exc}'
            )

        if self._spool is not None and plan is not None:
            try:
                self._spool.save(plan, reason=str(exc))
            except Exception as se:
                logger.error(f'Failed to spool episode {plan.get("uuid")}: {se}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id.

        Journal-backed: returns the last-known unfinished row count for the
        group (authoritative async counts live in ``get_resilience_snapshot``
        and the journal counters). Legacy: the in-memory queue size.
        """
        if self._journal is not None:
            return self._journal_queue_sizes.get(group_id, 0)
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    @property
    def stopping(self) -> bool:
        """True once a graceful shutdown drain has been initiated."""
        return self._stopping

    def begin_stopping(self) -> None:
        """Flip the stopping flag; every further add_memory is rejected."""
        if not self._stopping:
            self._stopping = True
            logger.warning('Queue service entering stopping mode: new episodes will be rejected')

    def _any_busy(self) -> bool:
        """Whether any worker currently has an episode in flight."""
        return any(c > 0 for c in self._busy_counts.values())

    async def drain(self, wait_current_seconds: float = 7.0) -> dict[str, Any]:
        """Spill every pending episode to the disk spool and drain in-flight work.

        Called from the graceful shutdown coordinator on SIGTERM/SIGINT (and
        once more, idempotently, on server exit). Steps:

        1. Reject new submissions (stopping flag).
        2. Move every queued-but-unprocessed episode to the disk spool.
        3. Give the in-flight episode a grace window to finish; on timeout
           cancel its worker, which spools the cancelled episode.
        """
        self.begin_stopping()
        start = asyncio.get_running_loop().time()

        if self._journal is not None:
            # Journal-backed: pending rows are already durable in SQLite, so
            # nothing is spilled. Give the in-flight episode a grace window to
            # finish, then cancel overdue workers. A cancelled 'processing' row
            # stays in the journal and is requeued by requeue_stale() on the
            # next boot.
            deadline = start + wait_current_seconds
            while self._any_busy() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)
            waited_seconds = asyncio.get_running_loop().time() - start

            cancelled_workers: list[asyncio.Task] = []
            if self._any_busy():
                cancelled_workers = [t for t in self._pool_tasks if not t.done()]
                for task in cancelled_workers:
                    task.cancel()
                if cancelled_workers:
                    await asyncio.gather(*cancelled_workers, return_exceptions=True)

            logger.warning(
                'Graceful drain (journal): waited current %.1fs, cancelled %d overdue '
                'worker(s); pending rows stay durable in the journal',
                waited_seconds,
                len(cancelled_workers),
            )
            return {
                'spooled_pending': 0,
                'waited_seconds': round(waited_seconds, 2),
                'cancelled_workers': len(cancelled_workers),
            }

        spooled_pending = 0
        for group_id in list(self._episode_queues):
            queue = self._episode_queues[group_id]
            while True:
                try:
                    _, plan = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._queue_depth = max(0, self._queue_depth - 1)
                queue.task_done()
                if plan is None:
                    logger.error(
                        'Graceful drain: pending episode in group %s has no spoolable plan; it is lost', group_id
                    )
                    continue
                if self._spool is None:
                    logger.error(
                        'Graceful drain: spool disabled, pending episode %s (name=%s) cannot be persisted',
                        plan.get('uuid'),
                        plan.get('name'),
                    )
                    continue
                try:
                    self._spool.save(plan, reason='graceful drain on shutdown')
                    spooled_pending += 1
                except Exception as e:
                    logger.error('Graceful drain: failed to spool episode %s: %s', plan.get('uuid'), e)

        deadline = start + wait_current_seconds
        while self._any_busy() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
        waited_seconds = asyncio.get_running_loop().time() - start

        cancelled_workers: list[asyncio.Task] = []
        if self._any_busy():
            cancelled_workers = [
                task
                for group_id, task in self._worker_tasks.items()
                if self._busy_counts.get(group_id, 0) > 0 and not task.done()
            ]
            for task in cancelled_workers:
                task.cancel()
            if cancelled_workers:
                await asyncio.gather(*cancelled_workers, return_exceptions=True)

        logger.warning(
            'Graceful drain: spooled %d pending, waited current %.1fs, cancelled %d overdue worker(s)',
            spooled_pending,
            waited_seconds,
            len(cancelled_workers),
        )
        return {
            'spooled_pending': spooled_pending,
            'waited_seconds': round(waited_seconds, 2),
            'cancelled_workers': len(cancelled_workers),
        }

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id.

        Journal-backed mode runs the global pool, which serves every group, so
        this reports whether the pool is up; legacy mode reports the per-group
        worker flag.
        """
        if self._journal is not None:
            return self._pool_started
        return self._queue_workers.get(group_id, False)

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """The circuit breaker guarding the write path, if resilience is enabled."""
        return self._breaker

    @property
    def spool(self) -> EpisodeSpool | None:
        """The episode spool persisting failed writes, if spooling is enabled."""
        return self._spool

    @property
    def journal(self) -> QueueJournal | None:
        """The durable SQLite journal backing the write path, if enabled."""
        return self._journal

    async def initialize(
        self,
        graphiti_client: Any,
        episode_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
            episode_builder: Rebuilds graphiti.add_episode kwargs from a persisted
                journal plan. Required for journal-backed processing; set only when
                resilience is enabled.
            llm_semaphore: Optional shared cap on concurrent add_episode calls. When
                omitted, a private semaphore sized from ``resilience.semaphore_limit``
                is used; pass the server-wide semaphore so the journal pool and the
                direct path share one LLM concurrency limit.
        """
        self._graphiti_client = graphiti_client
        self._episode_builder = episode_builder
        if llm_semaphore is not None:
            self._llm_semaphore = llm_semaphore

        if self._journal is not None:
            if self._episode_builder is None:
                logger.warning(
                    'Queue journal enabled but no episode_builder provided; '
                    'journal rows cannot be processed until one is set'
                )
            # Heal hard-killed in-flight rows (kill -9): processing -> pending.
            requeued = await self._journal.requeue_stale()
            if requeued:
                logger.warning(
                    'Requeued %d stale processing row(s) from the journal (killed in flight)',
                    requeued,
                )
            # Alarm: wakes workers when backed-off rows become due. It never
            # ingests and never probes the breaker (cf. 680a069 fix).
            retryer = JournalRetryer(
                self._journal,
                self._wake_group,
                interval_seconds=self.resilience.retryer_interval_seconds,
            )
            retryer.start()
            self._journal_retryer = retryer
            # Start the global worker pool and pick up rows persisted by a
            # previous process (or before workers existed) without waiting for
            # the first new add_memory.
            self._ensure_pool()
            for group_id in await self._journal.distinct_groups_pending():
                await self._wake_group(group_id)
            # Lease steward: periodically reclaim rows whose owner vanished.
            self._steward_task = asyncio.create_task(self._journal_steward())

        logger.info('Queue service initialized with graphiti client')

    async def get_resilience_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the resilience state.

        Journal-backed: queue_depth and pending_episodes are read from the
        journal table (correct after a restart), and a ``journal`` block is
        included for /health.
        """
        if self._breaker is not None:
            snapshot = await self._breaker.get_snapshot()
        else:
            snapshot = {
                'state': 'disabled',
                'failure_count': 0,
                'retry_after_seconds': 0.0,
                'last_failure_ts': None,
            }
        snapshot['max_queue_depth'] = self.resilience.max_queue_depth
        snapshot['spool_enabled'] = self._spool is not None

        if self._journal is not None:
            stats = await self._journal.stats()
            metrics = await self._journal.processing_stats(window_seconds=3600.0)
            snapshot['queue_depth'] = stats['unfinished']
            snapshot['pending_episodes'] = stats['pending']
            snapshot['journal'] = {
                'enabled': True,
                'path': str(self._journal.db_path),
                **stats,
                'processed_1h': metrics['processed'],
                'avg_processing_seconds': metrics['avg_processing_seconds'],
            }
        else:
            snapshot['queue_depth'] = self._queue_depth
            snapshot['pending_episodes'] = self._spool.count_pending() if self._spool is not None else 0
            snapshot['journal'] = {'enabled': False}
        snapshot['stopping'] = self._stopping
        return snapshot

    async def close(self) -> None:
        """Stop the journal alarm and pool, then close the journal connection."""
        if self._journal_retryer is not None:
            try:
                await self._journal_retryer.stop()
            except Exception as e:
                logger.error('Failed to stop journal retryer during shutdown: %s', e)
            self._journal_retryer = None
        # Stop the lease steward before canceling the pool that might feed it.
        if self._steward_task is not None:
            self._steward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._steward_task
            self._steward_task = None
        # Cancel pool workers before closing the journal underneath them.
        if self._pool_tasks:
            for task in self._pool_tasks:
                task.cancel()
            await asyncio.gather(*self._pool_tasks, return_exceptions=True)
            self._pool_tasks = []
            self._pool_started = False
        if self._journal is not None:
            try:
                await self._journal.close()
            except Exception as e:
                logger.error('Failed to close journal at %s: %s', self._journal.db_path, e)

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

        Returns:
            The position in the queue

        Raises:
            CircuitOpenError: If the circuit breaker is open (fail fast).
            QueueCapacityExceeded: If the queue depth limit is reached (fail fast).
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        # Build a flat, JSON-serializable "plan" used for disk spooling on failure.
        # Complex structures (entity_types/edge_types/edge_type_map and the
        # EpisodeType enum) are rebuilt from the live service at retry time.
        plan: dict[str, Any] = {
            'group_id': group_id,
            'name': name,
            'episode_body': content,
            'source_description': source_description,
            'source': getattr(episode_type, 'name', episode_type),
            'uuid': uuid,
            'reference_time': (
                (reference_time or datetime.now(timezone.utc)).isoformat()
            ),
            'excluded_entity_types': excluded_entity_types,
            'previous_episode_uuids': previous_episode_uuids,
            'custom_extraction_instructions': custom_extraction_instructions,
            'update_communities': update_communities,
            'saga': saga,
            'saga_previous_episode_uuid': saga_previous_episode_uuid,
        }

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} (name={name}) for group {group_id}')

                # Process the episode using the graphiti client
                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} (name={name}) for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} (name={name}) for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing.
        return await self.add_episode_task(group_id, process_episode, plan=plan)
