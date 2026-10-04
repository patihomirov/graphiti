"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from config.schema import ResilienceConfig
from services.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    QueueCapacityExceeded,
    is_transient_error,
)
from services.episode_spool import EpisodeSpool

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

        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Worker task handles per group_id, used to cancel in-flight episodes
        # during a graceful drain.
        self._worker_tasks: dict[str, asyncio.Task] = {}
        # True while a worker is awaiting its process_func (episode in flight).
        self._busy: dict[str, bool] = {}
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
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time. Failed episodes are spooled to disk so they
        can be recovered by the background retryer.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                # Get the next episode processing tuple from the queue
                # This will wait if the queue is empty
                process_func, plan = await self._episode_queues[group_id].get()

                try:
                    # Mark the worker busy so the graceful drain knows this
                    # episode is in flight (not merely pending in the queue).
                    self._busy[group_id] = True
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
                    self._busy[group_id] = False
                    # Decrement the shared depth invariant and mark done regardless
                    # of success/failure.
                    self._queue_depth = max(0, self._queue_depth - 1)
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

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
        """Get the current queue size for a group_id."""
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
        return any(self._busy.get(g, False) for g in self._episode_queues)

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
                if self._busy.get(group_id, False) and not task.done()
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
        """Check if a worker is running for a group_id."""
        return self._queue_workers.get(group_id, False)

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """The circuit breaker guarding the write path, if resilience is enabled."""
        return self._breaker

    @property
    def spool(self) -> EpisodeSpool | None:
        """The episode spool persisting failed writes, if spooling is enabled."""
        return self._spool

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def get_resilience_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the resilience state."""
        if self._breaker is not None:
            snapshot = await self._breaker.get_snapshot()
            snapshot['queue_depth'] = self._queue_depth
        else:
            snapshot = {
                'state': 'disabled',
                'failure_count': 0,
                'retry_after_seconds': 0.0,
                'last_failure_ts': None,
                'queue_depth': self._queue_depth,
            }
        snapshot['max_queue_depth'] = self.resilience.max_queue_depth
        snapshot['spool_enabled'] = self._spool is not None
        snapshot['pending_episodes'] = self._spool.count_pending() if self._spool is not None else 0
        snapshot['stopping'] = self._stopping
        return snapshot

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
