"""Durable SQLite write-ahead journal for the episode write path.

When ``resilience.journal_enabled`` is true every episode "plan" - a flat,
JSON-serializable dict of primitives (same shape as the disk spool) - is written
atomically to a WAL-mode SQLite table *before* it is processed, making the
journal the source of truth for the episode queue. Group workers claim rows
(``UPDATE ... RETURNING``, FIFO by ``(group_id, status, id)``) instead of living
in an in-memory queue, so episodes survive hard kills (SIGKILL) and are
replayed after a restart via ``requeue_stale()``.

Design notes (see ~/docs/research/graphiti-sqlite-queue-journal.md):

- **D2 (claim semantics):** the table is the single source of truth; claim is
  an atomic ``UPDATE ... RETURNING`` (sqlite >= 3.35) with a ``BEGIN IMMEDIATE``
  fallback for older builds. Every claim stamps ``worker_id`` (``p<pid>``) and
  ``lease_until`` as a hook for phase-2 multi-worker leases.
- **D3 (retry):** the ``JournalRetryer`` is a pure alarm - it never ingests an
  episode itself and never calls ``breaker.allow_request()`` (probing would
  consume and strand the single half-open probe slot; cf. the 680a069 fix). On
  each tick it wakes the workers of groups holding due rows.
- **D4 (idempotency):** ``dedup_key`` = explicit ``uuid`` or
  ``sha256(name+episode_body+reference_time)[:32]`` surrogate; ``UNIQUE`` index
  + ``INSERT OR IGNORE`` gives quiet idempotency for replays.
- **D5 (fail-closed):** the journal is opened eagerly; a failure to open it
  when ``journal_enabled`` is set raises at construction time instead of a
  silent fallback to the lossy in-memory queue.

asyncio-safety: ``check_same_thread=False`` plus a single ``asyncio.Lock`` and
``asyncio.to_thread`` around every DB operation; short transactions only;
``journal_mode=WAL``, ``synchronous=NORMAL``, ``busy_timeout=5000``.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Every column we expose on a claimed row (mirrors the episode_queue schema).
_ROW_COLUMNS = (
    'id',
    'dedup_key',
    'uuid',
    'group_id',
    'name',
    'episode_body',
    'plan_json',
    'status',
    'attempt',
    'worker_id',
    'lease_until',
    'first_failure_ts',
    'last_attempt_ts',
    'next_retry_at',
    'error',
    'requires_serial',
    'created_at',
    'updated_at',
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episode_queue (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key      TEXT NOT NULL,
    uuid           TEXT,
    group_id       TEXT NOT NULL,
    name           TEXT NOT NULL,
    episode_body   TEXT NOT NULL,
    plan_json      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    attempt        INTEGER NOT NULL DEFAULT 0,
    worker_id      TEXT,
    lease_until    TEXT,
    first_failure_ts TEXT,
    last_attempt_ts  TEXT,
    next_retry_at  TEXT,
    error          TEXT,
    requires_serial INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_episode_queue_dedup ON episode_queue(dedup_key);
CREATE INDEX IF NOT EXISTS idx_episode_queue_due ON episode_queue(status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_episode_queue_fifo ON episode_queue(group_id, status, id);
"""

_SELECT_ROW = f"""
SELECT {', '.join(_ROW_COLUMNS)} FROM episode_queue WHERE id = ?
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _later_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def compute_dedup_key(plan: dict[str, Any]) -> str:
    """Return the dedup key for a plan.

    An explicit ``uuid`` is used verbatim; otherwise a surrogate digest of
    ``name + episode_body + reference_time`` mirrors the disk spool's filename
    scheme so both layers agree on what a "duplicate" is.
    """
    ep_uuid = plan.get('uuid')
    if ep_uuid:
        return str(ep_uuid)
    digest = hashlib.sha256(
        str(plan.get('name', '')).encode('utf-8')
        + b'\x00'
        + str(plan.get('episode_body', '')).encode('utf-8')
        + b'\x00'
        + str(plan.get('reference_time', '')).encode('utf-8')
    ).hexdigest()[:32]
    return f'sha256:{digest}'


def plan_requires_serial(plan: dict[str, Any]) -> bool:
    """Whether a plan depends on strictly serial (in-order) processing.

    A row is serial - only the group leader may claim it - when it performs
    auto-retrieval of previous episodes (no ``previous_episode_uuids``), starts
    or forks a saga without an explicit ``saga_previous_episode_uuid``, or asks
    for a community refresh (``update_communities``). Parallel-safe is only a
    plan with an explicit, already-written ``previous_episode_uuids`` list and,
    when it has a saga, an explicit ``saga_previous_episode_uuid``.
    """
    if not plan.get('previous_episode_uuids'):
        return True
    if plan.get('saga') and not plan.get('saga_previous_episode_uuid'):
        return True
    return bool(plan.get('update_communities'))


class QueueJournal:
    """Durable SQLite journal backing the episode write path."""

    def __init__(self, db_path: str | Path, lease_seconds: float = 300.0):
        self.db_path = str(Path(db_path).expanduser())
        self.lease_seconds = float(lease_seconds)
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA synchronous=NORMAL')
            self._conn.execute('PRAGMA busy_timeout=5000')
            self._conn.executescript(_SCHEMA)
            self._apply_migrations()
            self._conn.commit()
            # sqlite_version >= 3.35 supports UPDATE ... RETURNING (claimed via
            # a single statement); older builds get the BEGIN IMMEDIATE fallback.
            self._has_returning = sqlite3.sqlite_version_info >= (3, 35)
        except Exception:
            # Fail-closed (D5): do not leak a half-open connection.
            with contextlib.suppress(Exception):
                self._conn.close()  # type: ignore[has-type]
            raise
        logger.info('SQLite journal ready: %s (lease=%ss)', self.db_path, self.lease_seconds)

    def _apply_migrations(self) -> None:
        """Idempotently bring an older journal up to the current schema.

        ``ALTER TABLE ADD COLUMN`` is not idempotent, so each additive step is
        gated on ``PRAGMA table_info`` (rather than a try/except swallows
        errors). Runs once per connection, synchronously on startup.
        """
        existing = {row[1] for row in self._conn.execute('PRAGMA table_info(episode_queue)')}
        if 'requires_serial' not in existing:
            # Rows created before the column existed default to the serial-safe
            # group FIFO (0 = parallel-safe is only set going forward by enqueue).
            self._conn.execute(
                'ALTER TABLE episode_queue '
                'ADD COLUMN requires_serial INTEGER NOT NULL DEFAULT 0'
            )
        # Created here (after any ALTER) so it is valid on both fresh and
        # migrated journals; an early CREATE with the missing column would fail.
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_episode_queue_serial '
            'ON episode_queue(group_id, status, requires_serial, id)'
        )

    # ------------------------------------------------------------- plumbing

    async def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run one SQL statement on the journal connection (async-safe).

        Commits the transaction after the statement; only safe for statements
        whose cursor is fully consumed inside ``_exec_commit_sync`` (plain
        UPDATE/DELETE without ``RETURNING``).
        """
        async with self._lock:
            return await asyncio.to_thread(self._exec_commit_sync, self._conn, sql, params)

    @staticmethod
    def _exec_commit_sync(conn: sqlite3.Connection, sql: str, params: tuple) -> sqlite3.Cursor:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur

    def _exec_sync(self, sql: str, params: tuple) -> sqlite3.Cursor:
        """Execute without committing (caller fetches then commits)."""
        return self._conn.execute(sql, params)

    async def _commit(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.commit)

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        async with self._lock:
            if self._conn is None:
                return
            try:
                await asyncio.to_thread(self._conn.close)
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.error('Error closing journal connection %s: %s', self.db_path, e)
            finally:
                self._conn = None  # type: ignore[assignment]

    # ------------------------------------------------------------ lifecycle

    async def enqueue(self, plan: dict[str, Any]) -> tuple[int, bool]:
        """Atomically persist a plan, returning ``(row_id, inserted)``.

        ``INSERT OR IGNORE`` on the dedup key gives quiet idempotency: a
        duplicate plan returns the existing row id with ``inserted=False``.
        """
        dedup_key = compute_dedup_key(plan)
        now = _now_iso()
        plan_json = json.dumps(plan, ensure_ascii=False, sort_keys=True)
        created_at = now
        serial = int(plan_requires_serial(plan))
        params = (
            dedup_key,
            plan.get('uuid'),
            plan['group_id'],
            plan['name'],
            plan['episode_body'],
            plan_json,
            serial,
            created_at,
            created_at,
        )
        sql = (
            'INSERT OR IGNORE INTO episode_queue '
            '(dedup_key, uuid, group_id, name, episode_body, plan_json, '
            ' requires_serial, created_at, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) '
            'RETURNING id'
        )
        if not self._has_returning:
            sql = (
                'INSERT OR IGNORE INTO episode_queue '
                '(dedup_key, uuid, group_id, name, episode_body, plan_json, '
                ' requires_serial, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
            )
        async with self._lock:
            cur = await asyncio.to_thread(self._exec_sync, sql, params)
            row = cur.fetchone()
            await asyncio.to_thread(self._conn.commit)
            if row is not None:
                return int(row[0]), True
            # Duplicate: retrieve the existing id so callers can react.
            found = await asyncio.to_thread(self._find_id_sync, dedup_key)
            return found, False

    def _find_id_sync(self, dedup_key: str) -> int:
        cur = self._conn.execute(
            'SELECT id FROM episode_queue WHERE dedup_key = ?', (dedup_key,)
        )
        row = cur.fetchone()
        self._conn.commit()
        if row is None:  # pragma: no cover - race between insert and select
            raise sqlite3.IntegrityError(f'Missing duplicate row for dedup_key {dedup_key}')
        return int(row[0])

    def _claim_where(self, leader: bool) -> str:
        """Claim WHERE fragment; non-leaders may only take serial-safe rows.

        A serial row depends on context ordering (auto-retrieve of previous
        episodes, a saga without its predecessor, or ``update_communities``),
        so only the group leader may claim it; parallel-safe rows (explicit
        already-written ``previous_episode_uuids`` / ``saga_previous_episode_uuid``)
        can go to any worker.
        """
        if leader:
            return "(status = 'pending' AND (next_retry_at IS NULL OR next_retry_at <= ?))"
        return (
            "(status = 'pending' AND requires_serial = 0 "
            "AND (next_retry_at IS NULL OR next_retry_at <= ?))"
        )

    async def claim_next(
        self, group_id: str, worker_id: str, leader: bool = True
    ) -> dict[str, Any] | None:
        """Atomically claim the oldest pending row for a group (FIFO).

        Uses a single ``UPDATE ... RETURNING`` (sqlite >= 3.35); for older SQLite
        builds the same atomicity is reached with ``BEGIN IMMEDIATE`` under the
        journal lock (one writer at the WAL level).
        """
        now = _now_iso()
        lease_until = _later_iso(self.lease_seconds)
        where = self._claim_where(leader)
        if self._has_returning:
            sql = f"""
                UPDATE episode_queue
                SET status='processing', worker_id=?, lease_until=?, updated_at=?
                WHERE id = (
                    SELECT id FROM episode_queue
                    WHERE group_id = ? AND {where}
                    ORDER BY id LIMIT 1
                )
                RETURNING {', '.join(_ROW_COLUMNS)}
            """
            params = (worker_id, lease_until, now, group_id, now)
            async with self._lock:
                cur = await asyncio.to_thread(self._exec_sync, sql, params)
                row = cur.fetchone()
                await asyncio.to_thread(self._conn.commit)
            return dict(row) if row is not None else None

        # Fallback for sqlite < 3.35: BEGIN IMMEDIATE + SELECT + UPDATE.
        select_sql = (
            'SELECT id FROM episode_queue '
            f'WHERE group_id = ? AND {where} '
            'ORDER BY id LIMIT 1'
        )
        update_sql = (
            'UPDATE episode_queue SET status=\'processing\', worker_id=?, '
            'lease_until=?, updated_at=? WHERE id=?'
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_next_fallback_sync,
                select_sql,
                update_sql,
                group_id,
                now,
                worker_id,
                lease_until,
            )

    def _claim_next_fallback_sync(
        self, select_sql: str, update_sql: str, group_id: str, now: str, worker_id: str, lease_until: str
    ) -> dict[str, Any] | None:
        self._conn.execute('BEGIN IMMEDIATE')
        try:
            cur = self._conn.execute(select_sql, (group_id, now))
            row = cur.fetchone()
            if row is None:
                self._conn.commit()
                return None
            row_id = int(row[0])
            self._conn.execute(update_sql, (worker_id, lease_until, now, row_id))
            self._conn.commit()
            return dict(self._conn.execute(_SELECT_ROW, (row_id,)).fetchone())
        except Exception:
            self._conn.rollback()
            raise

    async def mark_done(self, row_id: int) -> None:
        await self._execute(
            "UPDATE episode_queue SET status='done', updated_at=? WHERE id=?",
            (_now_iso(), row_id),
        )

    async def bump_retry(
        self,
        row_id: int,
        attempt: int,
        error: str,
        next_retry_at: str,
    ) -> None:
        """Keep a failed row pending for a later retry (exponential backoff)."""
        await self._execute(
            "UPDATE episode_queue SET status='pending', attempt=?, error=?, "
            "next_retry_at=?, worker_id=NULL, lease_until=NULL, "
            "first_failure_ts=COALESCE(first_failure_ts, ?), last_attempt_ts=?, updated_at=? "
            'WHERE id=?',
            (attempt, error, next_retry_at, _now_iso(), _now_iso(), _now_iso(), row_id),
        )

    async def mark_failed(self, row_id: int, error: str) -> None:
        """Move an exhausted row to failed/ for manual re-add (analogous to the spool)."""
        await self._execute(
            "UPDATE episode_queue SET status='failed', error=?, worker_id=NULL, "
            'lease_until=NULL, updated_at=? WHERE id=?',
            (error, _now_iso(), row_id),
        )

    async def requeue_stale(self) -> int:
        """Requeue processing rows whose lease has expired (e.g. after a hard kill).

        Returns the number of rows moved back to ``pending``.
        """
        sql = (
            'UPDATE episode_queue SET status=\'pending\', worker_id=NULL, '
            'lease_until=NULL, '
            'error=COALESCE(error, \'requeued stale claim\'), updated_at=? '
            "WHERE status='processing' AND (lease_until IS NULL OR lease_until <= ?)"
        )
        async with self._lock:
            cur = await asyncio.to_thread(
                self._exec_commit_sync,
                self._conn,
                sql,
                (_now_iso(), _now_iso()),
            )
            return cur.rowcount

    async def heartbeat(self, row_id: int, worker_id: str) -> bool:
        """Renew the lease on a row owned by ``worker_id`` (long LLM calls).

        Returns ``False`` when the row no longer belongs to this worker (it was
        reclaimed, deleted, or finished), in which case the caller must stop
        heartbeating.
        """
        async with self._lock:
            cur = await asyncio.to_thread(
                self._exec_commit_sync,
                self._conn,
                "UPDATE episode_queue SET lease_until=?, updated_at=? "
                "WHERE id=? AND worker_id=? AND status='processing'",
                (_later_iso(self.lease_seconds), _now_iso(), row_id, worker_id),
            )
            return cur.rowcount > 0

    async def requeue_stale_except(self, worker_id: str, grace_seconds: float) -> int:
        """Requeue processing rows whose owner vanished, never stealing our own.

        Only rows whose lease expired more than ``grace_seconds`` ago AND that
        are owned by a different worker (or none) are taken back to ``pending``.
        The current worker's rows are never stolen - even after their lease - so
        a long in-flight LLM call on this process can never be double-processed;
        such rows are healed by ``requeue_stale()`` on the next boot instead.
        """
        cutoff = _later_iso(-grace_seconds)
        sql = (
            'UPDATE episode_queue SET status=\'pending\', worker_id=NULL, '
            'lease_until=NULL, error=\'requeued stale claim\', updated_at=? '
            "WHERE status='processing' AND (lease_until IS NULL OR lease_until <= ?) "
            'AND (worker_id IS NULL OR worker_id <> ?)'
        )
        async with self._lock:
            cur = await asyncio.to_thread(
                self._exec_commit_sync,
                self._conn,
                sql,
                (_now_iso(), cutoff, worker_id),
            )
            return cur.rowcount

    # ------------------------------------------------------------- counts

    async def count_pending(self) -> int:
        return await self._count_by_status("status='pending'")

    async def count_processing(self) -> int:
        return await self._count_by_status("status='processing'")

    async def count_failed(self) -> int:
        return await self._count_by_status("status='failed'")

    async def count_done(self) -> int:
        return await self._count_by_status("status='done'")

    async def _count_by_status(self, where: str) -> int:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                f'SELECT COUNT(*) FROM episode_queue WHERE {where}',
            )
            return int(cur.fetchone()[0])

    async def count_unfinished(self) -> int:
        """Rows that still need work: pending + processing.

        Single atomic query: splitting it into ``count_pending()`` +
        ``count_processing()`` could double-count a row claimed between the two
        reads (backpressure would then over-reject).
        """
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT COUNT(*) FROM episode_queue WHERE status IN ('pending', 'processing')",
            )
            return int(cur.fetchone()[0])

    async def count_pending_group(self, group_id: str) -> int:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT COUNT(*) FROM episode_queue WHERE group_id = ? AND status = 'pending'",
                (group_id,),
            )
            return int(cur.fetchone()[0])

    async def count_unfinished_group(self, group_id: str) -> int:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT COUNT(*) FROM episode_queue WHERE group_id = ? "
                "AND status IN ('pending', 'processing')",
                (group_id,),
            )
            return int(cur.fetchone()[0])

    async def distinct_groups_pending(self) -> list[str]:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT DISTINCT group_id FROM episode_queue WHERE status IN ('pending', 'processing')",
            )
            return [str(r[0]) for r in cur.fetchall()]

    async def group_unfinished(self) -> dict[str, int]:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT group_id, COUNT(*) FROM episode_queue "
                "WHERE status IN ('pending', 'processing') GROUP BY group_id",
            )
            return {str(r[0]): int(r[1]) for r in cur.fetchall()}

    async def due_digest(self) -> dict[str, int]:
        """Return ``group_id -> count`` of pending rows due for (re)processing.

        Rows with ``next_retry_at`` in the future (backed off) are excluded, so
        the alarm never wakes a group before its rows are ready.
        """
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "SELECT group_id, COUNT(*) FROM episode_queue "
                "WHERE status = 'pending' AND (next_retry_at IS NULL OR next_retry_at <= ?) "
                'GROUP BY group_id',
                (_now_iso(),),
            )
            return {str(r[0]): int(r[1]) for r in cur.fetchall()}

    async def stats(self) -> dict[str, int]:
        """One-shot counts of every status bucket plus the total."""
        pending = await self.count_pending()
        processing = await self.count_processing()
        failed = await self.count_failed()
        done = await self.count_done()
        return {
            'pending': pending,
            'processing': processing,
            'failed': failed,
            'done': done,
            'unfinished': pending + processing,
            'total': pending + processing + failed + done,
        }

    async def processing_stats(self, window_seconds: float = 3600.0) -> dict[str, float | int]:
        """Aggregate timing stats over ``done`` rows updated within the window.

        Returns ``{'processed': int, 'avg_processing_seconds': float}`` - the
        number of rows finished inside the window and their average wall-clock
        processing time (``updated_at - created_at``). Pure SELECTs over the
        existing ``created_at``/``updated_at`` columns (no breaker/spool
        involvement), used by ``/health`` and the parallel-drain bench.
        """
        window_start = _later_iso(-window_seconds)
        sql = (
            "SELECT COUNT(*) AS processed, "
            "COALESCE(AVG(julianday(updated_at) - julianday(created_at)) * 86400.0, 0.0) "
            'AS avg_seconds '
            "FROM episode_queue WHERE status = 'done' AND updated_at >= ?"
        )
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, sql, (window_start,))
            row = cur.fetchone()
        return {
            'processed': int(row['processed']),
            'avg_processing_seconds': round(float(row['avg_seconds']), 3),
        }

    async def cleanup_done(self, retention_seconds: float = 7 * 24 * 3600) -> int:
        """Delete rows that are ``done`` and older than the retention window."""
        cutoff = _later_iso(-retention_seconds)
        async with self._lock:
            cur = await asyncio.to_thread(
                self._exec_commit_sync,
                self._conn,
                "DELETE FROM episode_queue WHERE status = 'done' AND updated_at <= ?",
                (cutoff,),
            )
            return cur.rowcount


class JournalRetryer:
    """Alarm that wakes group workers when rows become due for (re)processing.

    Unlike the legacy spool ``EpisodeRetryer`` this class never ingests an
    episode itself and never calls ``breaker.allow_request()``. Probing the
    breaker would consume the single half-open probe slot with nothing to replay
    and could strand the breaker in ``half_open`` (the 680a069 fix). It only
    computes which groups hold due rows and wakes their workers.
    """

    def __init__(
        self,
        journal: QueueJournal,
        wake: Callable[[str], Awaitable[None]],
        interval_seconds: float = 15.0,
    ):
        self.journal = journal
        self.wake = wake
        self.interval_seconds = float(interval_seconds)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name='journal-retryer')
            logger.info('Journal retryer started (interval=%ss)', self.interval_seconds)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
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
                    logger.error('Journal retryer tick failed: %s', e)
        except asyncio.CancelledError:
            logger.info('Journal retryer cancelled')

    async def _tick(self) -> None:
        digest = await self.journal.due_digest()
        for group_id in digest:
            try:
                await self.wake(group_id)
            except Exception as e:
                logger.error('Journal retryer failed to wake group %s: %s', group_id, e)
