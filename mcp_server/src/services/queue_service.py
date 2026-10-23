"""Queue service for managing episode processing."""

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

from config.schema import ResilienceConfig
from services.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    QueueCapacityExceeded,
    _openai_api_status_code,
    _retry_after_from_exc,
    is_openai_timeout_error,
    is_permanent_error,
    is_transient_error,
)
from services.episode_spool import EpisodeSpool
from services.model_reputation import ModelReputationTracker
from services.queue_journal import JournalRetryer, QueueJournal
from services.rate_limiter import ChannelRateLimiter
from services.storm_throttle import StormThrottle

logger = logging.getLogger(__name__)

# Feature B: cooldown (seconds, monotonic) during which the zai overflow channel
# is treated as broken after ``resilience.zai_loss_strikes`` consecutive proxy
# losses (502 / 5xx / connection / auth). After the cooldown a fresh attempt is
# allowed and one success heals the channel.
ZAI_BROKEN_COOLDOWN_SECONDS = 300.0


def _error_signature(error: str) -> str:
    """Normalized poison-detection signature: first 120 chars, whitespace-flat.

    Whitespace is collapsed so ``"Request timed out."`` and
    ``"Request  timed out."`` share a signature; case is preserved (provider
    error bodies are case-stable).
    """
    collapsed = ' '.join(str(error).split())
    return collapsed[:120]


def _parse_iso_or_none(value: str) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp leniently; None when unparseable."""
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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

    def __init__(
        self,
        resilience: ResilienceConfig | None = None,
        journal_max_pending: int | None = None,
        storm_time_fn=None,
        zai_time_fn=None,
    ):
        """Initialize the queue service.

        Args:
            resilience: Optional resilience config. Defaults to a config with
                all stock defaults so callers that omit the argument keep working.
            journal_max_pending: Optional soft ceiling for durable-journal intake
                (pending+processing). Defaults to ``resilience.journal_max_pending``
                (500) so the journal path is decoupled from the legacy in-memory
                ``max_queue_depth`` (20).
            storm_time_fn: Optional clock for the per-model StormThrottle
                (defaults to time.monotonic). Test hook only.
            zai_time_fn: Optional monotonic clock for the zai overflow channel
                (defaults to time.monotonic). Test hook only.
        """
        self.resilience: ResilienceConfig = resilience or ResilienceConfig()
        self._journal_max_pending: int = (
            journal_max_pending
            if journal_max_pending is not None
            else self.resilience.journal_max_pending
        )
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
        # RCA C10: with a global pool of N workers a single per-process worker_id
        # would make every pool worker indistinguishable (a risk for the claim /
        # lease guard thinking two workers are one). Each pool worker receives its
        # own id ``f'{self._worker_id}-{i}'``; ``_worker_scope`` is the SET of ids
        # owned by this process, used by the lease steward so it never steals ANY
        # of our own in-flight rows.
        self._worker_base = self._worker_id
        self._worker_scope: set[str] = {self._worker_id}
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
        # Phase 5 (parallel-follower) per-group count of in-flight NON-SERIAL
        # rows claimed by follower workers (without the group lock). Bounds how
        # many parallel-safe rows one group may drain at once; the leader's
        # serial row is tracked separately in ``_busy_counts``. Reset on each
        # claim/finish, like ``_busy_counts``.
        self._active_followers: dict[str, int] = {}
        # Per-group cap on concurrent follower (non-serial) rows. 0 in config =
        # auto-derive as max(1, semaphore_limit // 2) so the followers never
        # dwarf the shared LLM semaphore that bounds total in-flight.
        self._follower_limit: int = max(
            1,
            int(
                self.resilience.journal_follower_limit
                or max(1, int(self.resilience.semaphore_limit) // 2)
            ),
        )
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
        # Absolute budget for one add_episode call inside the journal worker
        # (per model attempt). A hung LLM call past this is aborted so a wedged
        # serial group is released to retry instead of stranding for hours.
        self._journal_processing_timeout = float(
            self.resilience.journal_processing_timeout_seconds
        )
        # RCA C1/C2/C3 retry-semantics knobs, read from the resilience config:
        # permanent fast-fail threshold, poison signature-repeat threshold and
        # the per-episode wall-clock deadline (0 = off).
        self._permanent_fail_fast_attempts = max(
            1, int(getattr(self.resilience, 'permanent_fail_fast_attempts', 2) or 2)
        )
        self._poison_repeat_threshold = max(
            2, int(getattr(self.resilience, 'journal_poison_repeat_threshold', 3) or 3)
        )
        self._episode_wall_deadline_seconds = float(
            getattr(self.resilience, 'episode_wall_deadline_seconds', 21600.0) or 0.0
        )

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
        # Optional second Graphiti instance for the zai overflow channel (Feature B).
        self._graphiti_client_zai: Any = None

        # Total number of episodes currently queued or being processed across all
        # group_ids. Guarded by _depth_lock so concurrent submissions do not blow
        # past max_queue_depth.
        self._queue_depth: int = 0
        self._depth_lock = asyncio.Lock()

        # Phase 4: in-memory per-model reputation driving failover candidate
        # ordering (healthy channels first) and the /health per_model block.
        # Always present (cheap, defaults are behaviour-preserving); it only
        # starts to matter when model_fallbacks is non-empty.
        self._reputation = ModelReputationTracker(
            window_seconds=self.resilience.model_reputation_window_seconds,
            per_min_threshold=self.resilience.model_reputation_429_per_min_threshold,
            cooldown_seconds=self.resilience.model_reputation_429_cooldown_seconds,
        )

        # Feature A: per-model adaptive storm throttle. Detects a 429 storm per
        # model (window/threshold hysteresis, OR the reputation per-minute rate)
        # and slows ONLY the storming model: min inter-request interval, a
        # concurrency ceiling and hard Retry-After pauses. Healthy models keep
        # their normal tempo (per-model isolation).
        self._storm = StormThrottle(
            window_seconds=self.resilience.storm_detection_429_window_seconds,
            threshold=self.resilience.storm_detection_429_threshold,
            respect_retry_after=self.resilience.respect_retry_after,
            time_fn=storm_time_fn,
        )
        # Feature B: zai overflow channel state. The channel is a second Graphiti
        # client (``_graphiti_client_zai``, set by initialize) plus a small
        # strike/cool-down tracker so a broken zai proxy auto-rolls back to the
        # ozon channel without blocking episodes.
        self._zai_time_fn = zai_time_fn or time.monotonic
        self._zai_consecutive_losses = 0
        self._zai_broken_until = 0.0
        self._zai_last_error: str | None = None
        # Last value sampled by the overflow gate (unfinished pending+processing)
        # the last time failover candidates were built; surfaced in /health so a
        # threshold mismatch is visible without digging through logs.
        self._zai_backlog_unfinished = 0
        # Pacing state (in-flight call count / last-call timestamp per model)
        # for storming models. Guarded by a short-held threading lock (the
        # critical sections contain no awaits).
        self._storm_lock = threading.Lock()
        self._storm_inflight: dict[str, int] = {}
        self._storm_last_call: dict[str, float] = {}

        # Zai-first balance (2026-10-22): a small in-memory ring of recent
        # timeout / stream-timeout / connection-error timestamps per channel.
        # The ModelReputationTracker only scores 429 storms, so a formally
        # "healthy" but actually timeouting active ozon line keeps winning the
        # candidate order. This counter feeds ``_active_channel_degraded``: when
        # the active channel shows ``zai_first_timeout_threshold``+ transport
        # timeouts inside ``zai_first_timeout_window_seconds``, the zai overflow
        # channel is promoted to first candidate (drain through zai). Ring is
        # bounded; state is lost on restart (fresh process = legacy order).
        self._timeout_lock = threading.Lock()
        self._channel_timeouts: dict[str, deque[float]] = {}
        self._channel_timeout_budget = 200
        # Monotonic clock for the timeout ring (test hook; defaults to the
        # real clock, mirrors zai_time_fn / storm_time_fn).
        self._timeout_time_fn = time.monotonic

        # Always-active per-channel rate limiter (2026-10-21): a strict
        # sliding-window calls-per-minute ceiling on every (provider, model)
        # channel. Unlike StormThrottle (reactive - it only slows a model AFTER
        # 429s start) this is the honest, always-on pacer that keeps the
        # request stream uniform so the gateway never sees packets. A value of
        # 0 from config keeps the legacy behaviour (OFF).
        self._rate_limiter = ChannelRateLimiter(
            max_calls_per_minute=int(
                getattr(self.resilience, 'channel_max_calls_per_minute', 0) or 0
            ),
        )
        # Embedder channel limiter (2026-10-22): embeddings are the single largest
        # request source at the gateway (~61% of calls in the RCA window, from the
        # per-edge embedding fan-out) and share the gateway bucket with chat, so an
        # embedding storm throttles chat too. A SEPARATE sliding-window ceiling for
        # the embedder channel (base_url|embedding_model) smooths embedding peaks
        # (measured 187/min) to <= the configured ceiling without gating the healthy
        # chat path. 0 from config keeps it OFF. Separate instance on purpose: the
        # chat limiter and the embedder limiter must not steal slots from each other.
        self._embed_rate_limiter = ChannelRateLimiter(
            max_calls_per_minute=int(
                getattr(self.resilience, 'channel_max_embeddings_per_minute', 0) or 0
            ),
        )
        # When the limiter estimates a wait longer than this, the failover loop
        # skips to the next candidate instead of queueing behind a saturated
        # channel (see ``_channel_wait_or_skip``).
        self._channel_wait_abort_seconds = float(
            getattr(self.resilience, 'channel_wait_abort_seconds', 60.0) or 60.0
        )

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
    ) -> int | tuple[int, int]:
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
            The position in the queue (legacy path), or the journal
            ``(row_id, unfinished_group_size)`` pair on the journal-backed path.

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

    async def _add_episode_task_journal(self, group_id: str, plan: dict[str, Any]) -> tuple[int, int]:
        """Journal-backed add_episode_task: persist the plan, then wake a worker.

        Fail-fast backpressure (stopping, breaker, capacity) is applied *before*
        the atomic INSERT. The in-memory queue is NOT used for real episodes;
        workers claim rows directly from the journal table (claim semantics).

        Returns the ``(row_id, unfinished_group_size)`` pair: the durable
        journal row id (used to surface ``journal_id`` back to ``add_memory``)
        and the group's unfinished size (metrics, same as the legacy path).
        """
        async with self._depth_lock:
            breakthrough = self.resilience.enqueue_breakthrough_max_pending
            if self._breaker is not None:
                allowed = await self._breaker.allow_request()
                # Phase 4 bounded breakthrough: while the breaker is OPEN,
                # allow enqueues up to journal_max_pending + breakthrough so
                # canaries / critical episodes still reach the durable journal
                # (and drain on a live fallback via per-model routing) instead
                # of being rejected with graphiti_backpressure:. The capacity
                # check below bounds the accumulation, so the gateway is never
                # hammered.
                if not allowed and (breakthrough <= 0 or self._breaker.state != 'open'):
                    snap = await self._breaker.get_snapshot()
                    raise CircuitOpenError(
                        f'Circuit is open (state={snap["state"]}). '
                        f'Rejecting episode; retry in ~{snap["retry_after_seconds"]}s.'
                    )
            unfinished = await self._journal.count_unfinished()
            # Journal rows are durable on disk (survive restart), so the intake
            # ceiling is the wider ``journal_max_pending`` (default 500), NOT the
            # RAM-protecting ``max_queue_depth`` (20) that still bounds the legacy
            # in-memory path.
            capacity = self._journal_max_pending
            if self._breaker is not None and self._breaker.state == 'open':
                capacity += breakthrough
            if unfinished >= capacity:
                raise QueueCapacityExceeded(
                    f'Queue depth {unfinished} >= max {capacity}. '
                    'Rejecting episode; retry once the queue drains.'
                )
            row_id, inserted = await self._journal.enqueue(plan)

        if not inserted:
            logger.info(
                'Duplicate episode %s (name=%s) skipped by journal dedup',
                plan.get('uuid'),
                plan.get('name'),
            )

        await self._wake_group(group_id)
        size = await self._journal.count_unfinished_group(group_id)
        self._journal_queue_sizes[group_id] = size
        return int(row_id), size

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

    async def _process_journal_once(
        self,
        group_id: str,
        leader: bool = True,
        worker_id: str | None = None,
    ) -> bool | str:
        """Claim and process a single journal row for a group.

        Returns:
            ``False`` when no row was claimable; ``True`` when a row was actually
            processed (LLM bound or a real outcome); ``'deadline'`` when a stale
            row was surrendered at claim time (cheap bookkeeping only - the pool
            must not apply the inter-episode pause for it).

        ``leader`` selects the claim flavour: the group leader claims serial-safe
        rows (``leader=True``, strict FIFO, held under the group lock), while a
        follower (``leader=False``) may only claim ``requires_serial=0`` rows and
        runs them in parallel with the leader and the other followers (no group
        lock). Follower in-flight rows are mirrored in ``_active_followers`` so
        ``_pick_group`` can respect the per-group parallel cap.

        ``worker_id`` is the calling pool worker's unique id (RCA C10); defaults
        to the process-level ``self._worker_id`` for legacy callers.
        """
        worker_id = worker_id or self._worker_id
        row = await self._journal.claim_next(group_id, worker_id, leader=leader)
        if row is None:
            return False

        # RCA C3 wall-clock deadline: an episode that has already started
        # processing (attempt >= 1) and exceeded its deadline is surrendered to
        # failed right at claim time, so a row that has been oscillating for
        # hours cannot even be re-processed. Never-attempted backlog rows are
        # exempt (they are not poison, just waiting in FIFO).
        deadline_error = self._deadline_error(row)
        if self._deadline_exceeded(row):
            await self._journal.mark_failed(
                row['id'], deadline_error, worker_id=worker_id
            )
            logger.warning(
                'Episode %s (name=%s) group %s: refused to claim - %s',
                row.get('uuid'),
                row.get('name'),
                group_id,
                deadline_error,
            )
            # Sentinel: caller (pool worker) must NOT apply the inter-episode
            # pause for a deadline reject - the reject is cheap bookkeeping, not
            # an LLM bound, and pausing 24s per stale row stalls the whole queue
            # while hundreds of stale rows are surrendered (RCA-C3 regression).
            return 'deadline'

        self._busy_counts[group_id] = self._busy_counts.get(group_id, 0) + 1
        if not leader:
            self._active_followers[group_id] = self._active_followers.get(group_id, 0) + 1
        # Keep the lease fresh for long in-flight LLM calls so the steward never
        # steals our own row; cancelled in ``finally`` once the row is done.
        heartbeat = asyncio.create_task(
            self._journal_heartbeat_loop(row['id'], worker_id)
        )
        # Heal our OWN row when its lease goes stale (heartbeat died / add_episode
        # hung past reason); cancelled with the heartbeat once the row is done.
        watchdog = asyncio.create_task(
            self._journal_watchdog_loop(
                row['id'],
                worker_id,
                created_at=row.get('created_at'),
                attempt=(row.get('attempt') or 0),
                started_at=time.monotonic(),
            )
        )
        try:
            plan = json.loads(row['plan_json'])
            if self._episode_builder is None:  # guarded by caller, defensive
                raise RuntimeError('No episode_builder configured for journal processing')
            kwargs = self._episode_builder(plan)
            # Multi-model failover (phase 3, variant 'b'): the whole episode is
            # processed by a single chosen model; on a transient failure (429 /
            # empty response / transport / malformed JSON) the worker retries the
            # same row inline, inside this claim, with the next not-yet-attempted
            # fallback model. The shared LLM semaphore is held around each
            # add_episode call.
            await self._process_with_failover(row, kwargs)
        except asyncio.CancelledError:
            # The row stays 'processing'; requeue_stale() on the next boot
            # (or a stale-lease cleanup) heals it after the hard kill.
            raise
        except Exception as e:
            await self._handle_journal_failure(group_id, row, e, worker_id=worker_id)
        else:
            await self._journal.mark_done(row['id'], worker_id)
            if self._breaker is not None:
                await self._breaker.record_success()
        finally:
            heartbeat.cancel()
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(heartbeat, watchdog, return_exceptions=True)
            count = self._busy_counts.get(group_id, 0) - 1
            if count > 0:
                self._busy_counts[group_id] = count
            else:
                self._busy_counts.pop(group_id, None)
            if not leader:
                followers = self._active_followers.get(group_id, 0) - 1
                if followers > 0:
                    self._active_followers[group_id] = followers
                else:
                    self._active_followers.pop(group_id, None)
            size = await self._journal.count_unfinished_group(group_id)
            self._journal_queue_sizes[group_id] = size
        return True

    # --- RCA C3 helper: per-episode wall-clock deadline ----------------------

    def _deadline_exceeded(self, row: dict[str, Any]) -> bool:
        """Whether a claimed row has blown its wall-clock deadline (RCA C3).

        Only rows that have ALREADY started processing (``attempt >= 1``) are
        subject: a never-attempted row sitting in the FIFO backlog is legitimate
        debt, not a stuck/poison episode, and must not be killed on arrival.
        """
        if self._episode_wall_deadline_seconds <= 0:
            return False
        if (row.get('attempt') or 0) < 1:
            return False
        created_at = row.get('created_at')
        if not created_at:
            return False
        created = _parse_iso_or_none(created_at)
        if created is None:
            return False
        age = datetime.now(timezone.utc) - created
        return age.total_seconds() > self._episode_wall_deadline_seconds

    @staticmethod
    def _deadline_error(row: dict[str, Any]) -> str:
        """Human-readable failure reason for a deadline-surrender (RCA C3)."""
        return (
            f'deadline exceeded {row.get("attempt") or 0} attempts, '
            f'created={row.get("created_at")}, wall-clock budget exhausted'
        )

    # --- RCA C2 poison-detection helpers -------------------------------------

    def _poison_state(self, row: dict[str, Any], error: str) -> tuple[str, int]:
        """Return ``(signature, repeat_count)`` for the poison bookkeeping.

        A signature is the normalized (whitespace-collapsed) first 120 chars of
        the error text. When it matches the row's last signature the counter
        advances, otherwise it resets to 1 (the signature changed).
        """
        signature = _error_signature(error)
        if row.get('repeat_signature') == signature:
            repeat_count = int(row.get('repeat_count') or 0) + 1
        else:
            repeat_count = 1
        return signature, repeat_count

    async def _process_with_failover(
        self, row: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        """Run one claimed journal row, failing over across models on transient errors.

        Implements the worker-level (variant 'b') multi-model design: the ENTIRE
        episode is processed by a single chosen model so graph extraction stays
        consistent (never a mix of models inside one episode). When the currently
        selected model fails transiently (RateLimitError / EmptyResponseError /
        transport error), the worker retries the SAME row inline — within the
        same claim, without releasing it and without any circuit-breaker
        interaction — using the next model from ``resilience.model_fallbacks``
        that this row has not tried yet (``attempted_models`` journal column).

        The breaker only ever sees the final outcome of the whole chain
        (record_success / record_failure in the caller), so a failover hop never
        burns a half-open probe and never trips the breaker on a mid-chain
        failure (cf. the 680a069 fix). Inline fallback attempts do NOT increment
        ``attempt``: one logical attempt = one run of the episode, so a 429 storm
        cannot exhaust ``max_spool_attempts``. When every candidate is exhausted
        the last exception is re-raised so the standard path (bump_retry +
        exponential backoff) takes over.
        """
        row_id = int(row['id'])
        llm_client = getattr(self._graphiti_client, 'llm_client', None)
        setter = self._model_override_setter(llm_client)
        attempted = self._attempted_from_row(row)
        # Backlog gate (Feature B overflow): sample the journal once per episode
        # and let the gate decide whether the zai channel may even be a
        # candidate. Sampling here (once per row) keeps the cost to a single
        # COUNT per episode while still reacting to backlog growth/drain.
        zai_allowed = await self._zai_overflow_allowed()
        candidates = self._failover_candidates(
            attempted, llm_client, self._reputation, zai_allowed=zai_allowed
        )

        last_error: BaseException | None = None
        for candidate in candidates:
            # Feature B: a candidate is either the active ozon channel (plain model
            # id, routed through self._graphiti_client with a model override) or the
            # zai overflow channel (provider-qualified id, routed through the
            # dedicated second Graphiti instance). One episode always runs on ONE
            # channel end-to-end so graph extraction stays consistent across models.
            zai_channel = self._is_zai_channel(candidate)
            if zai_channel and self._graphiti_client_zai is None:
                # Defensive: the candidate should only be offered when the zai
                # client exists; if not, skip to the next channel.
                logger.warning(
                    'Episode %s (name=%s): zai channel %r chosen but no zai client; '
                    'skipping to next channel',
                    row.get('uuid'),
                    row.get('name'),
                    candidate,
                )
                last_error = last_error or RuntimeError('zai channel without zai client')
                continue
            client = self._graphiti_client_zai if zai_channel else self._graphiti_client
            # The ozon override is pinned only for ozon candidates; the zai client
            # lives on its own model and override (None resets the ozon one).
            if setter is not None:
                setter(None if zai_channel else candidate)
            storm_call = False
            try:
                if self._storm.in_storm(candidate):
                    # Feature A pacing applies ONLY to the storming model;
                    # other candidates run at their normal tempo.
                    # (1) Hard Retry-After: do not contact the model right now —
                    #     skip to the next channel / fall through to L4 backoff.
                    throttle = self._storm.is_throttled(candidate)
                    if throttle is not None:
                        logger.warning(
                            'Episode %s (name=%s): model %r in storm, throttled '
                            'by Retry-After (%.1fs); skipping to next channel',
                            row.get('uuid'),
                            row.get('name'),
                            candidate,
                            throttle,
                        )
                        last_error = last_error or RateLimitError(
                            f'model {candidate} throttled by Retry-After during storm'
                        )
                        continue
                    # (2) Min inter-request delay on the storming model.
                    await self._storm_pace(candidate)
                    # (3) Concurrency ceiling: a storming model serves at most
                    #     ``storm_concurrency_ceiling`` calls at once. At the
                    #     ceiling the call is not started here — the fallback /
                    #     L4 path picks it up so the model is never overloaded.
                    if not await self._storm_try_acquire(candidate):
                        logger.warning(
                            'Episode %s (name=%s): model %r in storm at '
                            'concurrency ceiling (%d); skipping to next channel',
                            row.get('uuid'),
                            row.get('name'),
                            candidate,
                            self.resilience.storm_concurrency_ceiling,
                        )
                        last_error = last_error or RateLimitError(
                            f'model {candidate} at storming concurrency ceiling'
                        )
                        continue
                    storm_call = True
                # Always-active per-channel pacing gate (2026-10-21): BEFORE an
                # episode starts on a channel, ask the rate limiter whether the
                # (provider, model) channel has a slot in the current window. A
                # short/zero wait means the actual per-call smoothing will be
                # done by the client inside add_episode; an estimated wait that
                # exceeds the abort window means the channel is saturated and we
                # move to the next failover candidate instead of queueing behind
                # it. Honest behaviour: a 429 afterwards never loosens the
                # limiter - the stream stays at the configured ceiling.
                if not await self._channel_wait_or_skip(candidate):
                    logger.warning(
                        'Episode %s (name=%s): channel %r saturated (>%.0fs '
                        'estimated slot wait); skipping to next candidate',
                        row.get('uuid'),
                        row.get('name'),
                        candidate,
                        self._channel_wait_abort_seconds,
                    )
                    last_error = last_error or RateLimitError(
                        f'channel {candidate} saturated: slot wait exceeds abort window'
                    )
                    continue
                # LLM + graph write is the scarce resource shared with the direct
                # path: hold the global semaphore around the actual call. A hung
                # call past ``_journal_processing_timeout`` is aborted so a wedged
                # LLM fetch cannot strand this serial group; the row goes to
                # retry (backoff) via the standard failure path below.
                # The semaphore ACQUIRE itself is also time-bounded: a wedged
                # holder can never leave the whole failover chain waiting forever
                # (observed: a worker stuck on an acquire for 30+ min while the
                # lease heartbeat kept running - the regression this guard fixes).
                await asyncio.wait_for(
                    self._llm_semaphore.acquire(),
                    timeout=self._journal_processing_timeout,
                )
                try:
                    logger.info(
                        'jp-progress: episode %s (name=%s) candidate %r: entering LLM call',
                        row.get('uuid'),
                        row.get('name'),
                        candidate,
                    )
                    await asyncio.wait_for(
                        client.add_episode(**kwargs),
                        timeout=self._journal_processing_timeout,
                    )
                finally:
                    self._llm_semaphore.release()
                self._record_reputation(candidate, None)  # success
                self._zai_channel_success(candidate)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                # Failover only applies to transient provider trouble AND only
                # when model switching is actually available (a real llm_client
                # exposing set_model_override, an known model id). Otherwise the
                # standard failure path handles the episode unchanged.
                if not self._failover_eligible(setter, candidate, e):
                    raise
                self._record_reputation(candidate, e)
                # Feature B auto-rollback: consecutive zai proxy losses mark the
                # channel broken on a cooldown (ozon keeps working regardless).
                self._zai_channel_failure(candidate, e)
                attempted.append(candidate)
                await self._journal.set_attempted_models(row_id, attempted)
                remaining = len(candidates) - len(attempted)
                logger.warning(
                    'Episode %s (name=%s) failed on model %r (%s); %d fallback(s) left',
                    row.get('uuid'),
                    row.get('name'),
                    candidate,
                    e,
                    remaining,
                )
            finally:
                if storm_call:
                    self._storm_release(candidate)
                if setter is not None:
                    setter(None)

        if last_error is not None:
            # Carry the last-attempted model on the exception so the caller's
            # failure handler can apply storm-aware backoff (bump_retry resets
            # the row's attempted_models column, so the row dict no longer
            # carries it by the time we get here).
            failed_model = attempted[-1] if attempted else None
            if failed_model is not None:
                with contextlib.suppress(Exception):
                    last_error._storm_model = failed_model
            raise last_error
        raise RuntimeError('failover produced no outcome')  # pragma: no cover

    @staticmethod
    def _attempted_from_row(row: dict[str, Any]) -> list[str]:
        """Parse the ``attempted_models`` JSON column of a claimed row (anti-loop)."""
        raw = row.get('attempted_models') or '[]'
        try:
            parsed = json.loads(raw)
            return [str(m) for m in parsed] if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []

    def _failover_candidates(
        self,
        attempted: list[str],
        llm_client: Any,
        reputation: ModelReputationTracker | None = None,
        zai_allowed: bool | None = None,
    ) -> list[str | None]:
        """Ordered list of channels (provider-qualified model ids) to try for a row.

        Channel order: the active ozon model first (plain id), then the zai
        overflow channel (``zai/...`` - a different provider quota) when Feature B
        is enabled, the channel is not in a broken cool-down AND the overflow
        backlog gate allows it (``zai_allowed is not False``), then
        ``resilience.model_fallbacks`` (ozon ids) in configured order. Models
        already in ``attempted`` are skipped (never re-tried within the current
        attempt chain); when every model has been attempted the FULL chain is
        restarted (RCA C6 "all attempted -> start over") so an episode always has
        a candidate and the zai strike/cooldown bookkeeping stays live. When a
        reputation tracker is supplied and more than one candidate remains, they
        are reordered so healthy (live) channels come first - the 429-storming
        primary drops behind a working fallback or zai.
        ``zai_allowed`` is ``None`` (callers that predate the gate) or ``True``
        (gate passed) -> zai is offered as today; ``False`` excludes zai from the
        candidate list *before* the reorder, so a reputation-driven reorder can
        never promote a gated-out zai channel to first. If there is no active
        model, or nothing usable remains, a single ``None`` candidate keeps the
        standard non-overridden path.
        """
        active = getattr(llm_client, 'model', None) if llm_client is not None else None
        if not active:
            # No active model -> multi-model failover is meaningless; the
            # standard non-overridden path handles the episode.
            return [None]
        ordered: list[str] = [active]
        # Feature B: the zai overflow channel (separate provider quota). Only
        # offered while the channel is configured, enabled, not on a broken
        # cool-down AND not excluded by the overflow backlog gate - so a dead
        # zai proxy never delays an episode and zai stays a strict overflow
        # (drain) channel instead of a routine candidate.
        zai_model = self._zai_channel_model()
        if (
            zai_allowed is not False
            and zai_model
            and self._zai_enabled()
            and zai_model not in ordered
        ):
            ordered.append(zai_model)
        for fallback in self.resilience.model_fallbacks or []:
            if fallback and fallback not in ordered:
                ordered.append(fallback)
        candidates = [m for m in ordered if m not in attempted]
        if not candidates:
            # Every channel has already been attempted for this row: RESTART the
            # full chain (RCA C6 "if all attempted - start over"). This both
            # guarantees the episode keeps cycling through every worker (no
            # deadlock on an exhausted set) and keeps reaching the zai overflow
            # channel on repeated attempts, so its strike/cooldown bookkeeping
            # stays live (the zai proxy-loss auto-rollback depends on it).
            candidates = list(ordered)
        else:
            # Zai-first balance (2026-10-22): when the active ozon channel is
            # degrading (429-unhealthy by reputation OR a steady stream of recent
            # timeout / stream-timeout / connection errors), the zai overflow
            # channel jumps to the FRONT of the chain - drain through zai instead
            # of hammering a line that keeps timeouting. In the healthy case the
            # legacy per-model ordering is preserved exactly (active first).
            candidates = self._prefer_zai_first(candidates, active, reputation)
        return candidates

    def _prefer_zai_first(
        self,
        candidates: list[str | None],
        active: str | None,
        reputation: ModelReputationTracker | None,
    ) -> list[str | None]:
        """Reorder candidates: zai-first when the active channel is degrading.

        Falls through to the legacy ``reorder_by_health`` (healthy channels
        first, configured order otherwise) when the zai-first condition is not
        met, so the non-storming / healthy path behaves exactly as before.
        """
        if len(candidates) <= 1:
            return list(candidates)
        zai_model = self._zai_channel_model()
        if (
            zai_model in candidates
            and active is not None
            and self._active_channel_degraded(active, reputation)
        ):
            rest = [m for m in candidates if m != zai_model]
            return [zai_model] + rest
        if reputation is not None and len(candidates) > 1:
            return reputation.reorder_by_health(candidates)
        return list(candidates)

    def _active_channel_degraded(
        self,
        active: str,
        reputation: ModelReputationTracker | None,
    ) -> bool:
        """Whether the active channel looks degraded enough to flip to zai-first.

        Two independent signals, OR-ed: (1) the reputation tracker marks the
        429-heavy active channel unhealthy (existing per-model health), and
        (2) ``zai_first_timeout_threshold``+ recent timeout / stream-timeout /
        connection events for the active channel inside
        ``zai_first_timeout_window_seconds`` (the 429-blind case this balance
        was added for - a slow/timing-out ozon line that 429-reputation cannot
        see because the calls never come back at all).
        """
        if reputation is not None and not reputation.model_is_healthy(active):
            return True
        window = float(
            getattr(self.resilience, 'zai_first_timeout_window_seconds', 300.0) or 300.0
        )
        threshold = int(
            getattr(self.resilience, 'zai_first_timeout_threshold', 2) or 2
        )
        if threshold <= 0:
            return False
        now = self._timeout_time_fn()
        cutoff = now - window
        with self._timeout_lock:
            items = self._channel_timeouts.get(active)
        if not items:
            return False
        recent = 0
        for ts in reversed(items):
            if ts <= cutoff:
                break  # deque is time-ordered (append), older items are at the left
            recent += 1
            if recent >= threshold:
                return True
        return recent >= threshold

    # --- Feature B: zai overflow channel helpers ---------------------------------

    def _zai_channel_model(self) -> str | None:
        """The provider-qualified zai channel id, or None when not configured."""
        if self._graphiti_client_zai is None:
            return None
        model = getattr(self.resilience, 'zai_model', 'zai/glm-5.3') or 'zai/glm-5.3'
        return str(model)

    def _is_zai_channel(self, candidate: str | None) -> bool:
        """Whether a candidate is the zai overflow channel (vs. the ozon client)."""
        if candidate is None:
            return False
        return candidate == self._zai_channel_model() and self._graphiti_client_zai is not None

    def _zai_broken(self) -> bool:
        """Whether the zai channel is currently in a broken cool-down."""
        return self._zai_time_fn() < self._zai_broken_until

    def _zai_enabled(self) -> bool:
        """Whether the zai overflow channel is active for failover right now."""
        if self._graphiti_client_zai is None:
            return False
        if not bool(getattr(self.resilience, 'zai_fallback_enabled', False)):
            return False
        return not self._zai_broken()

    async def _zai_overflow_allowed(self) -> bool:
        """Backlog gate: whether the zai channel may join failover candidates now.

        With ``resilience.zai_overflow_pending_threshold > 0`` the zai channel
        participates only while the journal is backed up: ``unfinished``
        (pending+processing) >= threshold. That is the "default cannot keep up
        and the queue is piling up" signal the drain channel exists for. A
        threshold of 0 (the default) keeps the legacy behaviour (zai always
        offered while enabled); a missing journal (legacy in-memory path) is
        never gated. The sampled backlog is cached in ``_zai_backlog_unfinished``
        for /health.
        """
        threshold = int(getattr(self.resilience, 'zai_overflow_pending_threshold', 0) or 0)
        if threshold <= 0:
            return True
        if self._journal is None:
            return True
        unfinished = await self._journal.count_unfinished()
        self._zai_backlog_unfinished = unfinished
        return unfinished >= threshold

    def _zai_channel_success(self, candidate: str | None) -> None:
        """A zai-channel success heals the channel (resets the strike counter)."""
        if not self._is_zai_channel(candidate):
            return
        self._zai_consecutive_losses = 0
        self._zai_broken_until = 0.0
        self._zai_last_error = None

    def _zai_channel_failure(self, candidate: str | None, exc: BaseException) -> None:
        """Track a zai-channel failure; N consecutive proxy losses mark it broken.

        Only real proxy outages (502 / 5xx / connection / authorization) advance
        the strike counter - a 429 rate limit means the proxy is alive, so it
        neither counts toward nor heals the strike-out (a later success does).
        """
        if not self._is_zai_channel(candidate):
            return
        if not self._is_zai_proxy_loss(exc):
            return
        self._zai_consecutive_losses += 1
        self._zai_last_error = f'{type(exc).__name__}: {exc}'
        strikes = max(1, int(getattr(self.resilience, 'zai_loss_strikes', 3)))
        if self._zai_consecutive_losses >= strikes:
            self._zai_broken_until = self._zai_time_fn() + ZAI_BROKEN_COOLDOWN_SECONDS
            logger.warning(
                'zai overflow channel marked unavailable for %.0fs after %d '
                'consecutive proxy loss(es): %s',
                ZAI_BROKEN_COOLDOWN_SECONDS,
                self._zai_consecutive_losses,
                self._zai_last_error,
            )

    @staticmethod
    def _is_zai_proxy_loss(exc: BaseException) -> bool:
        """Whether a zai failure is a proxy outage (not mere rate limiting).

        A dead / wedged local proxy surfaces as 5xx (notably 502), auth errors
        (401/403) and transport/connection failures. A 429 (RateLimitError or a
        429 httpx status) means the proxy is alive and should NOT count toward
        the strike-out.
        """
        if isinstance(exc, RateLimitError):
            return False
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        if isinstance(exc, asyncio.TimeoutError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429:
                return False
            return 401 <= status < 500 or 500 <= status < 600
        # RCA C7: an openai SDK status error carried by the zai proxy line - 5xx
        # (notably 502 Bad Gateway) means the proxy is wedged; at the same time a
        # 429 rate limit proves it is alive.
        status = _openai_api_status_code(exc)
        if status is not None:
            if status == 429:
                return False
            return status >= 500 or 401 <= status < 500
        return False

    def _zai_health(self) -> dict[str, Any]:
        """JSON-serializable zai channel block for /health (read-only)."""
        configured = self._graphiti_client_zai is not None
        enabled = bool(getattr(self.resilience, 'zai_fallback_enabled', False))
        return {
            'enabled': enabled and configured,
            'configured': configured,
            'healthy': configured and not self._zai_broken(),
            'model': self._zai_channel_model() if configured else None,
            # Overflow backlog gate: the configured threshold and the last
            # sampled unfinished backlog. When threshold>0 and backlog_unfinished
            # < threshold the zai channel is gated OUT of failover candidates.
            'overflow_threshold': int(
                getattr(self.resilience, 'zai_overflow_pending_threshold', 0) or 0
            ),
            'backlog_unfinished': self._zai_backlog_unfinished,
            'last_error': self._zai_last_error,
            'consecutive_losses': self._zai_consecutive_losses,
            'broker_cooldown_remaining': (
                round(max(0.0, self._zai_broken_until - self._zai_time_fn()), 1)
                if self._zai_broken()
                else 0.0
            ),
        }

    def _record_reputation(self, model: str | None, exc: BaseException | None) -> None:
        """Record a per-model reputation outcome for a failover attempt.

        ``exc`` None means the attempt succeeded; otherwise the transient error
        class drives the event kind (429 / empty / transient). Permanent errors
        are not recorded (they are not provider trouble). 429 outcomes also feed
        the per-model StormThrottle (with any Retry-After hint and the
        reputation health signal as the OR-entry condition).
        """
        if model is None:
            return
        if exc is None:
            self._reputation.record_success(model)
            self._storm.record_success(model)
            return
        kind = self._reputation_kind(exc)
        if kind == '429':
            self._reputation.record_429(model)
            self._storm.record_failure(
                model,
                retry_after=_retry_after_from_exc(exc),
                reputation_unhealthy=not self._reputation.model_is_healthy(model),
            )
        elif kind == 'empty':
            self._reputation.record_empty(model)
        elif kind == 'transient':
            self._reputation.record_transient(model)
            if self._is_channel_timeout_exc(exc):
                self._record_channel_timeout(model)

    @staticmethod
    def _is_channel_timeout_exc(exc: BaseException) -> bool:
        """Whether a transient failure is a timeout / stream / connection problem.

        ``Request timed out`` and ``stream timeout`` surface as httpx timeouts
        / read errors; ``Connection error`` as transport or OSError classes.
        These are exactly the "formally healthy but actually timing out" line
        that the zai-first balance keys on (a 429-reputation tracker cannot see
        them - the call never completes).
        """
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        if isinstance(exc, asyncio.TimeoutError):
            return True
        # The openai SDK wraps httpx timeouts / connection failures ("Request
        # timed out." from llm-gateway-proton) into APITimeoutError /
        # APIConnectionError - the same "formally healthy but actually timing
        # out" signal, so these must feed the zai-first timeout ring too.
        if is_openai_timeout_error(exc):
            return True
        return bool(isinstance(exc, httpx.TransportError))

    def _record_channel_timeout(self, model: str | None) -> None:
        """Record one recent timeout/connection event for the channel (bounded ring)."""
        if model is None:
            return
        ts = self._timeout_time_fn()
        with self._timeout_lock:
            ring = self._channel_timeouts.get(model)
            if ring is None:
                ring = deque(maxlen=self._channel_timeout_budget)
                self._channel_timeouts[model] = ring
            ring.append(ts)

    @staticmethod
    def _is_rate_limit_error(exc: BaseException) -> bool:
        """Whether a failure is a rate-limit (429) - a recoverable storm signal.

        A storm of identical 429s is legitimate: the exponential backoff is
        exactly the mechanism designed to ride it out. Unlike timeouts / 5xx /
        permanent classes, a repeating 429 must NOT trigger the poison
        quarantine (RCA C2).
        """
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429
        status = _openai_api_status_code(exc)
        return status == 429

    @staticmethod
    def _reputation_kind(exc: BaseException) -> str:
        """Classify a transient failure into a reputation event kind."""
        if isinstance(exc, RateLimitError):
            return '429'
        if isinstance(exc, EmptyResponseError):
            return 'empty'
        # A truncated / malformed JSON body is provider trouble, not a
        # validation error: it counts as a generic transient for reputation.
        if isinstance(exc, json.JSONDecodeError):
            return 'transient'
        if is_transient_error(exc):
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                return '429'
            return 'transient'
        return 'permanent'

    @staticmethod
    def _model_override_setter(llm_client: Any) -> Callable[[str | None], None] | None:
        """Return the client's model-override hook, or None when unavailable."""
        setter = getattr(llm_client, 'set_model_override', None)
        return setter if callable(setter) else None

    @staticmethod
    def _failover_eligible(
        setter: Callable[[str | None], None] | None,
        model: str | None,
        exc: BaseException,
    ) -> bool:
        """Whether a transient failure on ``model`` may fall over to the next one."""
        if setter is None or model is None:
            return False
        return is_transient_error(exc)

    async def _storm_pace(self, model: str) -> None:
        """Enforce the min inter-request delay on a storming model.

        Non-storming models skip this entirely (per-model isolation). Uses the
        real monotonic clock: the pacing is a wall-clock rate limiter, so it must
        match actual request cadence even when the storm detector clock is a
        test hook.
        """
        interval = float(self.resilience.storm_min_request_interval_seconds)
        if interval <= 0:
            return
        now = time.monotonic()
        with self._storm_lock:
            last = self._storm_last_call.get(model, 0.0)
            elapsed = now - last
            wait = interval - elapsed if elapsed < interval else 0.0
            # Mark the start of this call slot immediately so a concurrent
            # worker on the same storming model also sees the reservation.
            self._storm_last_call[model] = now
        if wait > 0:
            await asyncio.sleep(wait)

    async def _storm_try_acquire(self, model: str) -> bool:
        """Acquire a storming-model concurrency slot; False when at the ceiling."""
        ceiling = max(1, int(self.resilience.storm_concurrency_ceiling))
        with self._storm_lock:
            inflight = self._storm_inflight.get(model, 0)
            if inflight >= ceiling:
                return False
            self._storm_inflight[model] = inflight + 1
        return True

    def _storm_release(self, model: str) -> None:
        """Release a storming-model concurrency slot."""
        with self._storm_lock:
            inflight = self._storm_inflight.get(model, 0) - 1
            if inflight > 0:
                self._storm_inflight[model] = inflight
            else:
                self._storm_inflight.pop(model, None)

    # --- Always-active per-channel pacing (2026-10-21) -------------------------

    def _channel_for_candidate(self, candidate: str) -> str:
        """The (provider, model) channel key a failover candidate maps to.

        Must match the LLM client's own ``channel_key()`` (base_url|model) so
        the failover gate and the client-side limiter agree on what is
        "saturated". The ozon channel uses the ozon base_url + candidate model
        id (the override is pinned inside add_episode); the zai channel uses
        the zai base_url + the bare wire model (the zai client has no override,
        its ``self.model`` is the stripped id).
        """
        if self._is_zai_channel(candidate):
            client = self._graphiti_client_zai
            wire = candidate.split('/', 1)[1] if candidate.startswith('zai/') else candidate
            base = (
                getattr(getattr(getattr(client, 'llm_client', None), 'config', None), 'base_url', None)
                or 'zai'
            )
            return f'{base}|{wire}'
        client = self._graphiti_client
        base = (
            getattr(getattr(getattr(client, 'llm_client', None), 'config', None), 'base_url', None)
            or 'ozon'
        )
        return f'{base}|{candidate}'

    async def _channel_wait_or_skip(self, candidate: str | None) -> bool:
        """Whether the worker may proceed on ``candidate`` channel right now.

        Honest, always-active pacing gate: when the per-channel limiter is
        enabled and the channel is saturated for the current window (the
        estimated slot wait would exceed ``_channel_wait_abort_seconds``) the
        caller should SKIP to the next failover candidate. Returns True to
        proceed (slot free / short wait / limiter OFF / no candidate), False to
        skip. Actual per-call smoothing is done by the client's own limiter
        hook inside ``add_episode``.
        """
        if candidate is None or not self._rate_limiter.enabled:
            return True
        wait = await self._rate_limiter.estimated_wait_seconds(
            self._channel_for_candidate(candidate)
        )
        return wait <= self._channel_wait_abort_seconds

    def _active_channel_count(self) -> int:
        """Unique (provider, model) channels the pool may currently use.

        Drives the inter-episode pause multiplier: the active model, the
        configured fallbacks and the zai overflow channel (when configured).
        """
        channels: set[str] = set()
        llm = getattr(self._graphiti_client, 'llm_client', None)
        active = getattr(llm, 'model', None)
        if active:
            channels.add(str(active))
        for fb in self.resilience.model_fallbacks or []:
            if fb:
                channels.add(str(fb))
        zai = self._zai_channel_model()
        if zai:
            channels.add(zai)
        return max(1, len(channels))

    async def _inter_episode_pause(self) -> None:
        """Pause between episodes of ONE worker (storm_min_request_interval_seconds).

        Activated globally (2026-10-21), not only in storm: after finishing an
        episode a worker sleeps ``interval * active_channel_count`` before its
        next claim, so a pool that just finished a batch does not immediately
        start a whole new batch at once (that instant batch-start is exactly
        what burst the gateway into 429). 0 keeps the legacy behaviour.
        """
        interval = float(
            getattr(self.resilience, 'storm_min_request_interval_seconds', 0.0) or 0.0
        )
        if interval <= 0:
            return
        await asyncio.sleep(interval * self._active_channel_count())

    def _pacing_health(self) -> dict[str, Any]:
        """JSON-serializable pacing state for /health (read-only)."""
        return {
            'enabled': self._rate_limiter.enabled,
            'max_calls_per_minute': self._rate_limiter.max_calls_per_minute,
            'window_seconds': round(self._rate_limiter.window_seconds, 1),
            'inter_episode_interval_seconds': float(
                getattr(self.resilience, 'storm_min_request_interval_seconds', 0.0) or 0.0
            ),
            'active_channels': self._active_channel_count(),
            'per_channel': self._rate_limiter.snapshot(),
            # Separate embedder limiter (2026-10-22): embeddings get their own
            # sliding-window ceiling so an embedding storm cannot burn the shared
            # gateway bucket and throttle chat. Surfaced here to watch total_waits.
            'embedding': self._embed_rate_limiter.snapshot(),
        }

    async def _journal_pool_worker(self, worker_id: str) -> None:
        """Serve journal groups from the global worker pool.

        Each worker alternates two roles for a group:

        - **Leader** (one per group, holds the group lock): claims the oldest
          pending row regardless of ``requires_serial`` and runs it to
          completion. Serial rows (auto-previous context / saga chains /
          ``update_communities``) are only ever taken by the leader, so
          within-group serial order stays strictly FIFO.
        - **Follower** (several per group, no lock): only after a group's lock
          is busy (someone is already leading) may a worker claim a
          ``requires_serial=0`` row and process it in parallel. Followers never
          touch serial rows, and a group is given at most
          ``_follower_limit`` concurrent follower rows, so serial zones are
          never parallelized while the parallel-safe backlog drains faster.

        The pool size therefore adds cross-group concurrency AND within-group
        concurrency for parallel-safe rows only.

        ``worker_id`` is this worker's unique claim-owner tag (RCA C10). With
        it, a row's lease/heartbeat/watchdog are tied to the exact worker that
        claimed it, so two workers of the same process are never
        indistinguishable for the claim/steal guards.
        """
        try:
            while True:
                if self._stopping:
                    return
                if self._episode_builder is None:
                    await self._journal_wait_global()
                    continue
                # Leader role: only a group with a free lock is eligible.
                group_id = await self._pick_group(leader=True)
                if group_id is not None:
                    lock = self._group_locks.setdefault(group_id, asyncio.Lock())
                    if await lock.acquire():
                        try:
                            outcome = await self._process_journal_once(
                                group_id, leader=True, worker_id=worker_id
                            )
                            if outcome is True:
                                # Inter-episode pause: keep the pool from
                                # immediately starting the next batch (this
                                # instant restart is what bursts the gateway).
                                await self._inter_episode_pause()
                                continue
                            if outcome == 'deadline':
                                # Stale-row surrender: cheap bookkeeping, no LLM
                                # bound - continue WITHOUT the inter-episode pause
                                # so hundreds of stale rows drain in seconds, not
                                # hours (RCA-C3 regression).
                                continue
                        finally:
                            lock.release()
                    # The lock was busy or the leader had nothing to claim: fall
                    # through to the follower check instead of idling.
                # Follower role: parallel-safe rows, no group lock needed; only
                # eligible while a leader holds the lock (serial FIFO active).
                group_id = await self._pick_group(leader=False)
                if group_id is not None:
                    outcome = await self._process_journal_once(
                        group_id, leader=False, worker_id=worker_id
                    )
                    if outcome is True:
                        await self._inter_episode_pause()
                        continue
                    if outcome == 'deadline':
                        continue
                await self._journal_wait_global()
        except asyncio.CancelledError:
            logger.info('Journal pool worker cancelled')
            raise
        except Exception as e:
            logger.error('Unexpected error in journal pool worker: %s', e)

    async def _pick_group(self, leader: bool = True) -> str | None:
        """Return a group this worker may claim a row from right now.

        Leader mode (``leader=True``): the first group with due rows whose lock
        is currently free - the worker would become that group's serial leader.

        Follower mode (``leader=False``): the first group that (a) has a due
        ``requires_serial=0`` row AND (b) is not already at the per-group
        follower cap. The group's lock may be HELD by its leader - that is
        exactly when followers are allowed to parallelize non-serial rows.
        Serial-only groups are never offered to followers.
        """
        digest = await self._journal.due_digest()
        for group_id in digest:
            lock = self._group_locks.get(group_id)
            if leader:
                if lock is None or not lock.locked():
                    return group_id
            else:
                if self._active_followers.get(group_id, 0) >= self._follower_limit:
                    continue
                if await self._journal.has_due_serial_safe(group_id):
                    return group_id
        return None

    async def _has_due_work(self) -> bool:
        """Whether any group currently has due rows this worker could claim.

        Considers either the leader path (a free group lock) or the follower
        path (parallel-safe rows within the per-group cap), so a worker only
        idles when there is genuinely nothing to do by either role.
        """
        return (
            await self._pick_group(leader=True) is not None
            or await self._pick_group(leader=False) is not None
        )

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

    async def _journal_heartbeat_loop(
        self, row_id: int, worker_id: str | None = None
    ) -> None:
        """Renew a claimed row's lease while its add_episode is in flight.

        Runs as a side task per in-flight row and is cancelled once the row is
        finished; the lease outlives a long LLM call, so the steward cannot
        reclaim it and no two workers process the same episode.
        """
        worker_id = worker_id or self._worker_id
        interval = max(0.1, self._journal.lease_seconds / 2.0)
        try:
            while True:
                await asyncio.sleep(interval)
                alive = await self._journal.heartbeat(row_id, worker_id)
                if not alive:
                    logger.warning(
                        'Lease lost on journal row %s (worker %s); stopping heartbeat',
                        row_id,
                        worker_id,
                    )
                    return
        except asyncio.CancelledError:
            return

    async def _journal_watchdog_loop(
        self,
        row_id: int,
        worker_id: str | None = None,
        created_at: str | None = None,
        attempt: int = 0,
        started_at: float | None = None,
    ) -> None:
        """Heal THIS worker's own processing row when its lease goes stale.

        The steward's steal guard deliberately never touches our own rows (a long
        in-flight LLM call must not be double-processed), so a live worker has to
        clean up its own mess: if the heartbeat task died, or add_episode hangs
        past reason without renewing the lease, this watchdog requeues the row
        back to ``pending`` (attempt+1) instead of stranding it in ``processing``
        until a restart. Runs as a side task next to the heartbeat task and is
        cancelled with it once the row is finished.

        It never fights a live heartbeat: the sleep window is
        ``lease_seconds + grace`` and the requeue is gated on an expired lease,
        so while the heartbeat renews the lease the check is a cheap no-op and
        the loop keeps watching. The claim is re-checked by
        ``requeue_own_stale`` (id + worker_id + status), so a row that was
        concurrently finished or reclaimed by another worker is never touched.

        RCA C3: on every wake, before the lease check, the watchdog also honours
        the per-episode wall-clock deadline - a row that has blown it is
        surrendered to ``failed`` even while its heartbeat is alive (the
        heartbeat alone would otherwise renew the lease forever without progress).
        """
        worker_id = worker_id or self._worker_id
        grace = float(self.resilience.journal_grace_seconds)
        interval = max(1.0, float(self._journal.lease_seconds) + grace)
        # Wall-clock backstop for a worker that renews the lease but never makes
        # progress (a stuck failover chain). 2x the add_episode budget is the
        # upper bound a legitimate run may take; beyond it the row is force-
        # released so its claim/lease guards cannot dead-lock on a live-zombie.
        force_after = 2.0 * float(self._journal_processing_timeout)
        try:
            while True:
                await asyncio.sleep(interval)
                # Wall-clock progress backstop first (independent of lease state).
                if started_at is not None and (time.monotonic() - started_at) > force_after:
                    released = await self._journal.release_own_row(
                        row_id,
                        worker_id,
                        error=(
                            f'stalled processing {force_after:.0f}s without progress '
                            '(JP-backstop)'
                        ),
                    )
                    if released:
                        logger.warning(
                            'watchdog: force-released own row %s (no progress for '
                            '%.0fs; JP-backstop)',
                            row_id,
                            force_after,
                        )
                        self._notify_pool()
                        return
                # Deadline check next (does not depend on a stale lease).
                if (
                    self._episode_wall_deadline_seconds > 0
                    and attempt >= 1
                    and created_at
                ):
                    created = _parse_iso_or_none(created_at)
                    if created is not None:
                        age = datetime.now(timezone.utc) - created
                        if age.total_seconds() > self._episode_wall_deadline_seconds:
                            failed = await self._journal.mark_failed(
                                row_id,
                                (
                                    f'deadline exceeded {age} wall-clock '
                                    f'(RCA C3 watchdog)'
                                ),
                                worker_id=worker_id,
                            )
                            if failed:
                                logger.warning(
                                    'watchdog: surrendered own row %s to failed '
                                    '(wall-clock deadline exceeded)',
                                    row_id,
                                )
                                self._notify_pool()
                                return
                healed = await self._journal.requeue_own_stale(
                    row_id, worker_id, grace
                )
                if healed:
                    logger.warning(
                        'watchdog: requeued own stale row %s (lease expired '
                        'while this worker still ran; retrying)',
                        row_id,
                    )
                    self._notify_pool()
                    return
                if not await self._journal.row_owned(row_id, worker_id):
                    # The row finished / moved on; nothing left to watch.
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
                    self._worker_scope, self.resilience.journal_grace_seconds
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
            # RCA C10: every pool worker claims with its OWN unique id so the
            # per-row lease/heartbeat/watchdog and the steward steal-guard can
            # tell two workers of the same process apart.
            worker_id = f'{self._worker_base}-{i}'
            self._worker_scope.add(worker_id)
            task = asyncio.create_task(
                self._journal_pool_worker(worker_id),
                name=f'journal-pool-worker-{i}',
            )
            self._pool_tasks.append(task)
        logger.info('Started %d journal pool worker(s)', n)

    async def _handle_journal_failure(
        self,
        group_id: str,
        row: dict[str, Any],
        exc: BaseException,
        worker_id: str | None = None,
    ) -> None:
        """React to a failed journal-processed episode: trip the breaker, back off.

        Transient failures trip the breaker; the row is kept ``pending`` with an
        exponential ``next_retry_at`` (the JournalRetryer alarm wakes the worker
        when it is due). Exhausted rows are moved to ``failed`` like the spool.

        ``worker_id`` is the pool worker that owned the claim (RCA C10): every
        claim-aware UPDATE below is gated on it so the row's owner is respected.

        Retry classification (RCA C1/C2):
        - **Poison** (``repeat_count`` of the SAME error signature reaches
          ``journal_poison_repeat_threshold``) -> immediate ``failed``
          ("poison: repeated signature xN") instead of spinning through all
          ``max_spool_attempts``. Rate-limit repeats (429) are EXCLUDED: a
          storm of identical 429s is a legitimate, recoverable backoff loop, not
          a poison;
        - **Permanent** (``is_permanent_error``: pydantic ValidationError, neo4j
          client-side data errors, openai SDK 4xx-except-429, graphiti
          validation) -> surrender after ``permanent_fail_fast_attempts``
          (default 2), not 10 - malformed episodes stop burning worker-slots;
        - otherwise the standard exponential backoff retry up to
          ``max_spool_attempts``.
        """
        worker_id = worker_id or self._worker_id
        transient = is_transient_error(exc) if self._breaker is not None else False
        if transient and self._breaker is not None:
            await self._breaker.record_failure(exc)

        row_id = int(row['id'])
        attempt = int(row.get('attempt', 0)) + 1
        error_text = str(exc)
        # RCA C2: track the repeated-signature counter. A signature change resets
        # the counter to 1; the same signature N times in a row is a deterministic
        # loop that no extra retry will fix -> quarantine. Rate-limited repeats
        # (429) never quarantine - the storm backoff is designed to ride them out.
        if not self._is_rate_limit_error(exc):
            signature, repeat_count = self._poison_state(row, error_text)
            if repeat_count >= self._poison_repeat_threshold:
                await self._journal.mark_failed(
                    row_id,
                    error=f'poison: repeated signature x{repeat_count}: {error_text[:80]}',
                    worker_id=worker_id,
                )
                logger.error(
                    'Episode %s (name=%s) for group %s QUARANTINED after %d identical '
                    'failure(s) ("poison" signature): %s',
                    row.get('uuid'),
                    row.get('name'),
                    group_id,
                    repeat_count,
                    error_text[:80],
                )
                return
        else:
            signature, repeat_count = None, None

        if attempt >= self.resilience.max_spool_attempts:
            await self._journal.mark_failed(row_id, error=error_text, worker_id=worker_id)
            logger.error(
                'Episode %s (name=%s) for group %s gave up after %d attempts: %s',
                row.get('uuid'),
                row.get('name'),
                group_id,
                attempt,
                exc,
            )
            return

        # RCA C1: a permanent failure is surrendered fast (attempt budget =
        # permanent_fail_fast_attempts) instead of exhausting max_spool_attempts.
        permanent = is_permanent_error(exc)
        if permanent and attempt >= self._permanent_fail_fast_attempts:
            await self._journal.mark_failed(
                row_id,
                error=f'permanent: {error_text}',
                worker_id=worker_id,
            )
            logger.error(
                'Episode %s (name=%s) for group %s surrendered as PERMANENT after %d '
                'attempt(s) (fast-fail): %s',
                row.get('uuid'),
                row.get('name'),
                group_id,
                attempt,
                error_text,
            )
            return

        backoff = self.resilience.spool_backoff_base_seconds * (2 ** (attempt - 1))
        # Feature A: a storming model gets a longer release window so the storm
        # has time to subside instead of being re-hit right after the base
        # backoff. The failed model is carried on the exception by
        # _process_with_failover.
        storm_model = getattr(exc, '_storm_model', None)
        if storm_model is not None:
            if self._storm.is_throttled(storm_model) is not None:
                backoff *= 4
            elif self._storm.in_storm(storm_model):
                backoff *= 2
        # 2026-10-22 backoff-cap: without a cap a 30s base reaches ~1920s (32 min)
        # by attempt 7 and episodes "fall asleep" - done stays frozen for hours.
        # The cap (default 480s) returns episodes to work in minutes, while
        # StormThrottle / the per-channel rate limiter keep gating the actual
        # request rate. Applied AFTER the storm multiplier, so even a storming
        # channel can never sleep past the cap.
        cap = float(
            getattr(self.resilience, 'retry_backoff_cap_seconds', 480.0) or 480.0
        )
        if backoff > cap:
            backoff = cap
        next_retry_at = (
            datetime.now(timezone.utc) + timedelta(seconds=backoff)
        ).isoformat()
        await self._journal.bump_retry(
            row_id,
            attempt=attempt,
            error=error_text,
            next_retry_at=next_retry_at,
            worker_id=worker_id,
            repeat_signature=signature,
            repeat_count=repeat_count,
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

    async def last_done_episode_uuid(self, group_id: str) -> str | None:
        """Return the uuid of the newest materialized episode for a group.

        Backs the opt-in ``auto_previous_episode_uuid`` flow in ``add_memory``:
        a ``done`` journal row was fully processed, so its episode exists in the
        graph and is the safe single predecessor to hand to a follow-up episode
        (skips auto-retrieval, making the new episode parallel-safe). Returns
        ``None`` when the journal is disabled or the group has no done rows.
        """
        if self._journal is not None:
            journal_uuid = await self._journal.last_done_uuid(group_id)
            if journal_uuid:
                return journal_uuid
        # Journal fallback to the graph: journal rows carry no ``uuid`` (the
        # uuid lives only on Episodic nodes), so a journal with no explicit
        # uuids can never answer. Ask Neo4j directly for the newest
        # materialized episode of the group - the graph is the source of truth
        # for the uuid of a fully processed episode.
        return await self._last_done_uuid_from_graph(group_id)

    async def _last_done_uuid_from_graph(self, group_id: str) -> str | None:
        """Return the uuid of the newest Episodic node for ``group_id`` from the
        graph (Neo4j), or ``None`` when the graph has none / is unreachable.

        The query filters on ``group_id`` and orders by ``created_at`` so the
        newest *materialized* episode wins. Because an Episodic node is written
        only after ``add_episode`` fully completes, this is the safe single
        predecessor for a follow-up parallel-safe episode - never a still
        in-flight row.
        """
        client = self._graphiti_client
        if client is None:
            return None
        driver = getattr(client, 'driver', None)
        if driver is None or not hasattr(driver, 'execute_query'):
            return None
        try:
            records, _, _ = await driver.execute_query(
                """
                MATCH (e:Episodic)
                WHERE e.group_id = $group_id
                RETURN e.uuid AS uuid
                ORDER BY e.created_at DESC
                LIMIT 1
                """,
                params={'group_id': group_id},
                routing_='r',
            )
            if not records:
                return None
            return str(records[0]['uuid'])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                'Neo4j last_done_uuid fallback failed for group %s: %s',
                group_id,
                exc,
            )
            return None

    async def journal_search_raw(
        self, query: str, *, limit: int = 20, include_done: bool = False
    ) -> list[dict[str, Any]]:
        """Search raw (not-yet-materialized) journal episodes by name/body.

        Backs the ``search_raw_episodes`` MCP tool. Returns ``[]`` when the
        journal is disabled - there is nothing to search - instead of raising,
        so the read path never breaks a journal-off server. Passes the journal
        rows through unchanged, so entries also carry the in-queue
        ``verified_by``/``verified_at`` fact-check markers from ``search_raw``.
        """
        if self._journal is None:
            return []
        return await self._journal.search_raw(
            query, limit=limit, include_done=include_done
        )

    async def initialize(
        self,
        graphiti_client: Any,
        episode_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        llm_semaphore: asyncio.Semaphore | None = None,
        journal_max_pending: int | None = None,
        graphiti_client_zai: Any = None,
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
            journal_max_pending: Optional soft ceiling for durable-journal intake
                (pending+processing), overriding whatever the constructor/resilience
                config resolved. ``None`` keeps the resolved value.
            graphiti_client_zai: Optional second Graphiti instance whose LLM client is
                the zai overflow provider. When set (and ``zai_fallback_enabled``), the
                journal worker accepts a zai channel between the active ozon model and
                model_fallbacks; one episode is processed entirely on a single channel.
        """
        self._graphiti_client = graphiti_client
        self._graphiti_client_zai = graphiti_client_zai
        self._episode_builder = episode_builder
        if journal_max_pending is not None:
            self._journal_max_pending = journal_max_pending
        if llm_semaphore is not None:
            self._llm_semaphore = llm_semaphore
        # Attach the always-active rate limiter to every LLM client (ozon + zai)
        # so EACH chat-completions call goes through the per-channel ceiling.
        for client in (graphiti_client, graphiti_client_zai):
            llm = getattr(client, 'llm_client', None)
            if llm is not None and hasattr(type(llm), 'set_channel_rate_limiter'):
                llm.set_channel_rate_limiter(self._rate_limiter)
        # Attach the separate embedder limiter to every graph client's embedder
        # (ozon + zai share the same embedding channel) so every embedding call is
        # paced under its own ceiling - the chat limiter and this one never borrow
        # slots from each other.
        for client in (graphiti_client, graphiti_client_zai):
            embedder = getattr(client, 'embedder', None)
            if embedder is not None and hasattr(type(embedder), 'set_channel_rate_limiter'):
                embedder.set_channel_rate_limiter(self._embed_rate_limiter)

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
                # Soft ceiling for journal intake (pending+processing), decoupled
                # from the legacy in-memory max_queue_depth (20).
                'max_pending': self._journal_max_pending,
                **stats,
                'processed_1h': metrics['processed'],
                'avg_processing_seconds': metrics['avg_processing_seconds'],
                # Active multi-model failover list (resilience.model_fallbacks).
                'fallbacks': list(self.resilience.model_fallbacks),
                # Raw (not-yet-materialized) episode search is available while
                # the journal is on; /health surfaces this so clients can gate
                # the search_raw_episodes tool.
                'search': True,
                # Phase 4: per-model reputation (429 / empty / transient /
                # success counters + healthy flag) for the /health block.
                'per_model': self._reputation.per_model_stats(),
            }
        else:
            snapshot['queue_depth'] = self._queue_depth
            snapshot['pending_episodes'] = self._spool.count_pending() if self._spool is not None else 0
            snapshot['journal'] = {'enabled': False, 'search': False}
        snapshot['stopping'] = self._stopping
        # Feature A: per-model storm block for /health -> storm.per_model
        # (which models are storming, their hard Retry-After remaining).
        snapshot['storm'] = self._storm.snapshot()
        # Feature B: zai overflow channel status (enabled/healthy/last_error/model).
        snapshot['zai'] = self._zai_health()
        # Always-active pacing state (2026-10-21) for /health -> pacing:
        # per-channel limits, current in-window call counts and limiter waits.
        snapshot['pacing'] = self._pacing_health()
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
    ) -> int | None:
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
            The journal ``row_id`` of the queued episode when the durable
            journal is enabled; ``None`` when it is disabled (legacy in-memory
            path). The row id lets the caller hand a validator the ``journal_id``
            to fact-check/verify without waiting for graph materialization.

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
        # Journal-backed path returns (row_id, size); the caller needs the
        # durable row id to surface journal_id. The legacy path returns the
        # in-memory position and has no journal row - map it to None.
        result = await self.add_episode_task(group_id, process_episode, plan=plan)
        if isinstance(result, tuple):
            return result[0]
        return None
