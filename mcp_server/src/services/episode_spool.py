"""Disk spool for episodes that failed to process through the LLM.

When the queue worker hits a transient failure (or a permanent one we do not want
to swallow), the episode "plan" - a flat, JSON-serializable dict of primitives -
is written to ``<spool_dir>/<uuid|hash>.json`` before the task is acknowledged.
A background retryer later replays these files, rebuilding the complex entities
(``entity_types``/``edge_types``/``edge_type_map`` and the ``EpisodeType`` enum)
from the live service at retry time. Because everything is on disk, episodes
survive process restarts and are never lost silently.

Idempotency note: a replay is only guaranteed idempotent when the episode has an
explicit ``uuid`` (graphiti-core upserts on that uuid). An episode spooled without
a uuid will be re-ingested as a new node on each replay, so callers should prefer
to pass a ``uuid`` to add_memory when they need dedup safety across retries.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.circuit_breaker import is_transient_error

logger = logging.getLogger(__name__)

SPOOL_SUFFIX = '.json'
KEEP_KEYS = (
    'group_id',
    'name',
    'episode_body',
    'source_description',
    'source',
    'uuid',
    'reference_time',
    'excluded_entity_types',
    'previous_episode_uuids',
    'custom_extraction_instructions',
    'update_communities',
    'saga',
    'saga_previous_episode_uuid',
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeSpool:
    """Persists episode plans to disk and tracks retry attempts."""

    def __init__(self, spool_dir: str | Path, backoff_base_seconds: float = 30.0):
        self.root = Path(spool_dir).expanduser()
        self.failed_dir = self.root / 'failed'
        self.failed_receipts = self.failed_dir / 'failed.jsonl'
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def _filename(self, plan: dict[str, Any]) -> str:
        ep_uuid = plan.get('uuid')
        if ep_uuid:
            return f'{ep_uuid}{SPOOL_SUFFIX}'
        digest = hashlib.sha256(
            str(plan.get('name', '')).encode()
            + b'\x00'
            + str(plan.get('episode_body', '')).encode()
            + b'\x00'
            + str(plan.get('reference_time', '')).encode()
        ).hexdigest()[:32]
        return f'{digest}{SPOOL_SUFFIX}'

    def _path_for(self, plan: dict[str, Any]) -> Path:
        return self.root / self._filename(plan)

    def list_pending(self) -> list[Path]:
        """Return top-level spool files (not the failed/ directory)."""
        if not self.root.exists():
            return []
        return [p for p in sorted(self.root.glob(f'*{SPOOL_SUFFIX}')) if p.is_file()]

    def count_pending(self) -> int:
        return len(self.list_pending())

    # ------------------------------------------------------------------ io

    def save(self, plan: dict[str, Any], reason: str) -> Path:
        """Persist a fresh episode plan (first failure). Idempotent per file."""
        record = {k: plan.get(k) for k in KEEP_KEYS if k in plan}
        record['attempt'] = round(int(plan.get('attempt', 0)))
        record['first_failure_ts'] = plan.get('first_failure_ts') or _now_iso()
        record['last_attempt_ts'] = _now_iso()
        record['last_reason'] = reason

        path = self._path_for(plan)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
        logger.warning(
            'Spooled episode %s (name=%s) after failure: %s', record.get('uuid'), record.get('name'), reason
        )
        return path

    def load(self, path: Path) -> dict[str, Any]:
        with path.open('r', encoding='utf-8') as fh:
            return json.load(fh)

    def update_attempt(self, plan: dict[str, Any], reason: str) -> Path:
        """Increment the attempt counter and stamp the last attempt time."""
        plan['attempt'] = int(plan.get('attempt', 0)) + 1
        plan['last_attempt_ts'] = _now_iso()
        plan['last_reason'] = reason
        path = self._path_for(plan)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return path

    def delete(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.error('Failed to remove spool file %s: %s', path, e)

    def mark_failed(self, path: Path, plan: dict[str, Any], reason: str) -> None:
        """Move an exhausted spool file to failed/ and append a manual-readd receipt."""
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        plan = dict(plan)
        plan['last_attempt_ts'] = _now_iso()
        plan['last_reason'] = reason

        dest = self.failed_dir / path.name
        tmp = self.failed_dir / (path.name + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        tmp.replace(dest)
        self.delete(path)

        receipt = {
            'uuid': plan.get('uuid'),
            'name': plan.get('name'),
            'group_id': plan.get('group_id'),
            'first_failure_ts': plan.get('first_failure_ts'),
            'last_attempt_ts': plan.get('last_attempt_ts'),
            'reason': reason,
            'ready_for_manual_readd': True,
        }
        with self.failed_receipts.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False) + '\n')
        logger.error(
            'Spool episode %s (name=%s) gave up after %d attempts, moved to failed/',
            plan.get('uuid'),
            plan.get('name'),
            plan.get('attempt'),
        )


class EpisodeRetryer:
    """Background retryer that replays spooled episodes.

    Runs an asyncio task that periodically scans the spool directory and attempts
    to ingest each pending episode via the graphiti client, rebuilding complex
    parameters through ``episode_builder``. Attempts are backed off exponentially
    (``backoff_base_seconds * 2^attempt``) and, when the circuit breaker is open,
    no attempts are made until it allows a request again. Episodes that exceed
    ``max_attempts`` are moved to ``failed/`` with a manual-readd receipt.
    """

    def __init__(
        self,
        spool: EpisodeSpool,
        breaker: Any,
        episode_builder: Callable[[dict[str, Any]], dict[str, Any]],
        graphiti_client: Any,
        interval_seconds: float = 15.0,
        max_attempts: int = 10,
        backoff_base_seconds: float = 30.0,
    ):
        self.spool = spool
        self.breaker = breaker
        self.episode_builder = episode_builder
        self.graphiti_client = graphiti_client
        self.interval_seconds = float(interval_seconds)
        self.max_attempts = int(max_attempts)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the background retry loop. Safe to call once."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name='episode-retryer')
            logger.info('Episode retryer started (interval=%ss, max_attempts=%d)', self.interval_seconds, self.max_attempts)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error('Episode retryer tick failed: %s', e)
        except asyncio.CancelledError:
            logger.info('Episode retryer cancelled')

    async def _tick(self) -> None:
        # Collect episodes that are actually due for a replay BEFORE probing the
        # breaker: allow_request() consumes the single half-open probe slot, and a
        # probe granted with nothing to replay would never be resolved by
        # record_success/record_failure, leaving the breaker stuck in half_open
        # and rejecting every new submission until a restart.
        due: list[tuple[Path, dict[str, Any]]] = []
        for path in self.spool.list_pending():
            try:
                plan = self.spool.load(path)
            except (OSError, json.JSONDecodeError) as e:
                logger.error('Skipping unreadable spool file %s: %s', path, e)
                continue
            if self._due_for(plan):
                due.append((path, plan))
        if not due:
            return
        # Do not burn attempts while the breaker is open.
        if self.breaker is not None and not await self.breaker.allow_request():
            return
        for path, plan in due:
            await self._process_one(path, plan)

    def _due_for(self, plan: dict[str, Any]) -> bool:
        attempt = int(plan.get('attempt', 0))
        if attempt <= 0:
            return True
        last_ts = plan.get('last_attempt_ts')
        if not last_ts:
            return True
        try:
            last_dt = datetime.fromisoformat(last_ts)
        except (TypeError, ValueError):
            return True
        delay = self.backoff_base_seconds * (2 ** (attempt - 1))
        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
        return elapsed >= delay

    async def _process_one(self, path: Path, plan: dict[str, Any]) -> None:
        try:
            kwargs = self.episode_builder(plan)
            await self.graphiti_client.add_episode(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            transient = is_transient_error(e)
            if self.breaker is not None:
                if transient:
                    await self.breaker.record_failure(e)
                else:
                    # A permanent failure is not evidence of provider trouble,
                    # but it must still resolve the probe slot this tick may have
                    # consumed, otherwise the breaker sticks in half_open.
                    await self.breaker.record_success()
            attempt = int(plan.get('attempt', 0)) + 1
            plan['attempt'] = attempt
            logger.warning(
                'Retry attempt %d/%d for spooled episode %s (name=%s) failed: %s',
                attempt,
                self.max_attempts,
                plan.get('uuid'),
                plan.get('name'),
                e,
            )
            if attempt >= self.max_attempts:
                self.spool.mark_failed(path, plan, reason=str(e))
            else:
                self.spool.update_attempt(plan, reason=str(e))
            return

        # The replay succeeded — resolve the probe this tick may have consumed so
        # the breaker closes instead of sticking in half_open with the probe slot
        # held forever.
        if self.breaker is not None:
            await self.breaker.record_success()
        self.spool.delete(path)
        logger.info('Retried spooled episode %s (name=%s) successfully', plan.get('uuid'), plan.get('name'))
