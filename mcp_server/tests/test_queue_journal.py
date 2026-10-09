"""Unit tests for the durable SQLite queue journal (write path source of truth).

Covers the QueueJournal table contract (enqueue/claim/done, idempotency,
status transitions, FIFO, group isolation, stale requeue, due selection, WAL)
and the QueueService journal-backed behaviour (process exactly once, restart
without duplicates, journal-sourced backpressure and snapshot). Everything runs
against a tmp_path SQLite DB with no network and no graphiti-core data.
"""

import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock

import pytest
from graphiti_core import Graphiti

from config.schema import ResilienceConfig
from services.queue_journal import QueueJournal, compute_dedup_key, plan_requires_serial
from services.queue_service import (
    CircuitOpenError,
    QueueCapacityExceeded,
    QueueService,
)


def make_plan(**over):
    plan = {
        'group_id': 'g1',
        'name': 'ep',
        'episode_body': 'some body',
        'source_description': 'desc',
        'source': 'text',
        'uuid': 'u-1',
        'reference_time': '2024-06-01T00:00:00+00:00',
        'excluded_entity_types': None,
        'previous_episode_uuids': None,
        'custom_extraction_instructions': None,
        'update_communities': False,
        'saga': None,
        'saga_previous_episode_uuid': None,
    }
    plan.update(over)
    return plan


def make_config(db_path) -> ResilienceConfig:
    return ResilienceConfig(
        journal_enabled=True,
        journal_path=str(db_path),
        journal_lease_seconds=300.0,
        spool_enabled=False,
    )


@pytest.fixture
async def journal(tmp_path):
    j = QueueJournal(tmp_path / 'journal.db', lease_seconds=300.0)
    yield j
    await j.close()


class TestDedupKey:
    @pytest.mark.unit
    def test_uuid_used_verbatim(self):
        assert compute_dedup_key(make_plan(uuid='abc-123')) == 'abc-123'

    @pytest.mark.unit
    def test_surrogate_sha256_prefix(self):
        p1 = make_plan(uuid=None)
        p2 = make_plan(uuid=None)
        key = compute_dedup_key(p1)
        assert key.startswith('sha256:')
        assert key == compute_dedup_key(p2)
        assert len(key) == len('sha256:') + 32

    @pytest.mark.unit
    def test_body_change_changes_surrogate(self):
        k1 = compute_dedup_key(make_plan(uuid=None, episode_body='a'))
        k2 = compute_dedup_key(make_plan(uuid=None, episode_body='b'))
        assert k1 != k2


class TestQueueJournal:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueue_claim_done(self, journal):
        row_id, inserted = await journal.enqueue(make_plan())
        assert inserted is True

        row = await journal.claim_next('g1', 'p123')
        assert row is not None
        assert row['id'] == row_id
        assert row['status'] == 'processing'
        assert row['worker_id'] == 'p123'
        assert row['lease_until'] is not None
        assert json.loads(row['plan_json'])['name'] == 'ep'

        await journal.mark_done(row_id)
        assert await journal.count_done() == 1
        assert await journal.count_unfinished() == 0
        # Already done -> the group has nothing left to claim.
        assert await journal.claim_next('g1', 'p123') is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_uuid_idempotency(self, journal):
        plan = make_plan(uuid='dup-uuid')
        row_id, inserted = await journal.enqueue(plan)
        assert inserted is True
        same_id, dup_inserted = await journal.enqueue(plan)
        assert dup_inserted is False
        assert same_id == row_id
        assert await journal.count_pending() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_surrogate_idempotency(self, journal):
        plan = make_plan(uuid=None, name='n', episode_body='b', reference_time='2024-06-01T00:00:00+00:00')
        row_id, inserted = await journal.enqueue(plan)
        assert inserted is True
        same_id, dup_inserted = await journal.enqueue(plan)
        assert dup_inserted is False
        assert same_id == row_id
        assert await journal.count_pending() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_status_transitions(self, journal):
        row_id, _ = await journal.enqueue(make_plan(uuid='t-1'))
        assert await journal.count_pending() == 1

        await journal.claim_next('g1', 'w')
        assert await journal.count_pending() == 0
        assert await journal.count_processing() == 1

        await journal.mark_failed(row_id, 'boom')
        assert await journal.count_processing() == 0
        assert await journal.count_failed() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_second_claim_same_group_none(self, journal):
        # Two rows, single worker: the second claim must take the FIFO-next row,
        # and a third claim on an exhausted group returns None.
        await journal.enqueue(make_plan(uuid='a'))
        await journal.enqueue(make_plan(uuid='b'))

        first = await journal.claim_next('g1', 'w')
        second = await journal.claim_next('g1', 'w')
        assert {first['uuid'], second['uuid']} == {'a', 'b'}
        third = await journal.claim_next('g1', 'w')
        assert third is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fifo_order_by_id(self, journal):
        ids = []
        for uuid in ('f1', 'f2', 'f3'):
            row_id, _ = await journal.enqueue(make_plan(uuid=uuid))
            ids.append(row_id)
        claimed = [await journal.claim_next('g1', 'w') for _ in range(3)]
        assert [c['id'] for c in claimed] == ids
        assert [c['uuid'] for c in claimed] == ['f1', 'f2', 'f3']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_group_isolation(self, journal):
        await journal.enqueue(make_plan(group_id='g1', uuid='g1-a'))
        await journal.enqueue(make_plan(group_id='g2', uuid='g2-a'))
        await journal.enqueue(make_plan(group_id='g2', uuid='g2-b'))

        # Claiming group g1 does not touch g2 rows.
        row = await journal.claim_next('g1', 'w')
        assert row['group_id'] == 'g1'
        assert await journal.count_unfinished_group('g2') == 2
        assert await journal.count_pending_group('g1') == 0  # g1-a claimed (processing)

        # Example of the per-group digest used to wake workers.
        digest = await journal.due_digest()
        assert digest == {'g1': 0, 'g2': 2} or digest.get('g2') == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_requeue_stale_heals_killed_inflight(self, journal):
        await journal.enqueue(make_plan(uuid='k-1'))
        row = await journal.claim_next('g1', 'w')

        # Simulate a hard kill: the row is stuck in 'processing'. A fresh
        # requeue_stale (with a past/absent lease) must move it back to pending.
        n = await journal.requeue_stale()
        # Lease is still fresh (300s), so nothing is requeued yet.
        assert n == 0
        # Force the lease into the past (a fake "killed 10 minutes ago" claim).
        await journal._execute(
            "UPDATE episode_queue SET lease_until='2000-01-01T00:00:00+00:00' WHERE id=?",
            (row['id'],),
        )

        n = await journal.requeue_stale()
        assert n == 1
        assert await journal.count_pending() == 1
        assert await journal.count_processing() == 0
        reclaimed = await journal.claim_next('g1', 'w2')
        assert reclaimed['id'] == row['id']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_due_selection_filters_backed_off(self, journal):
        row_id, _ = await journal.enqueue(make_plan(uuid='due-1'))
        await journal.claim_next('g1', 'w')
        # Failure marks the row pending with a future next_retry_at.
        await journal.bump_retry(row_id, attempt=1, error='boom', next_retry_at='2999-01-01T00:00:00+00:00')

        assert await journal.due_digest() == {}
        assert await journal.claim_next('g1', 'w') is None  # not due
        assert await journal.count_pending() == 1
        # Once due, it becomes claimable again.
        await journal._execute(
            'UPDATE episode_queue SET next_retry_at=NULL WHERE id=?', (row_id,)
        )
        assert await journal.due_digest() == {'g1': 1}
        assert await journal.claim_next('g1', 'w') is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_failed_exhausted(self, journal):
        row_id, _ = await journal.enqueue(make_plan(uuid='ex-1'))
        await journal.mark_failed(row_id, 'gave up')
        assert await journal.count_failed() == 1
        assert await journal.count_pending() == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_unfinished_after_restart(self, tmp_path):
        """A fresh QueueJournal on the same file sees rows from a "dead" process."""
        db = tmp_path / 'restart.db'
        j1 = QueueJournal(db)
        row_a, _ = await j1.enqueue(make_plan(uuid='r-a'))
        row_b, _ = await j1.enqueue(make_plan(uuid='r-b'))
        claimed = await j1.claim_next('g1', 'old-pid')  # one is in flight (processing)
        # Simulate a process long dead: push the lease into the past.
        await j1._execute(
            "UPDATE episode_queue SET lease_until='2000-01-01T00:00:00+00:00' WHERE id=?",
            (claimed['id'],),
        )
        await j1.close()

        j2 = QueueJournal(db)
        assert await j2.count_unfinished() == 2
        # requeue_stale returns the abandoned processing row (no valid lease).
        n = await j2.requeue_stale()
        assert n == 1
        assert await j2.count_pending() == 2
        await j2.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_and_wal(self, tmp_path):
        db = tmp_path / 'wal.db'
        j = QueueJournal(db)
        await j.enqueue(make_plan(uuid='w-1'))

        conn = sqlite3.connect(str(db))
        mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
        conn.close()
        assert mode == 'wal'
        assert (tmp_path / 'wal.db-wal').exists() or (tmp_path / 'wal.db').exists()

        await j.close()
        # Close is idempotent.
        await j.close()


class TestQueueServiceJournal:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_episode_processed_exactly_once(self, tmp_path):
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_config(tmp_path / 'j.db'))
        builder = make_builder()
        await service.initialize(client, episode_builder=builder)
        try:
            await service.add_episode(
                group_id='g1',
                name='ep',
                content='body',
                source_description='desc',
                episode_type='text',
                entity_types=None,
                uuid='u-solo',
            )
            await wait_until(lambda: client.add_episode.call_count >= 1)
            await asyncio.sleep(0.05)
            client.add_episode.assert_awaited_once()
            assert await service.journal.count_done() == 1
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restart_requeues_without_duplicates(self, tmp_path):
        """A second service on the same journal path replays unfinished rows.

        Two rows are persisted by "process 1" without being processed (a fake
        hard kill), then "process 2" claims and processes both in id order.
        """
        db = tmp_path / 'restart-svc.db'
        # "Process 1": only persists, never processes (no workers started).
        s1 = QueueService(make_config(db))
        await s1.initialize(AsyncMock(spec=Graphiti))
        await s1.add_episode(group_id='g1', name='epA', content='bodyA',
                             source_description='d', episode_type='text',
                             entity_types=None, uuid='u-A')
        await s1.add_episode(group_id='g1', name='epB', content='bodyB',
                             source_description='d', episode_type='text',
                             entity_types=None, uuid='u-B')
        assert await s1.journal.count_pending() == 2
        await s1.close()

        # "Process 2": the same journal path, now with a real worker + builder.
        client = AsyncMock(spec=Graphiti)
        s2 = QueueService(make_config(db))
        await s2.initialize(client, episode_builder=make_builder())
        try:
            assert await s2.journal.count_pending() == 2
            await wait_until(lambda: client.add_episode.await_count >= 2)
            await asyncio.sleep(0.05)
            # No duplicates: exactly two awaits, in row-id (FIFO) order.
            assert client.add_episode.await_count == 2
            names = [c.kwargs['name'] for c in client.add_episode.await_args_list]
            assert names == ['epA', 'epB']
            assert await s2.journal.count_done() == 2
        finally:
            await s2.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backpressure_too_many_episodes(self, tmp_path):
        from graphiti_core.llm_client.errors import RateLimitError

        gate = asyncio.Event()
        client = AsyncMock(spec=Graphiti)

        async def hold(**kw):
            await gate.wait()

        client.add_episode.side_effect = hold
        service = QueueService(
            make_config(tmp_path / 'backpressure.db').model_copy(
                update={'journal_max_pending': 2}
            )
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await service.add_episode(group_id='g1', name='epA', content='A',
                                      source_description='d', episode_type='text',
                                      entity_types=None, uuid='u-bp-A')
            await service.add_episode(group_id='g1', name='epB', content='B',
                                      source_description='d', episode_type='text',
                                      entity_types=None, uuid='u-bp-B')
            # The worker is blocked, so both rows count against the limit.
            with pytest.raises(QueueCapacityExceeded):
                await service.add_episode(group_id='g1', name='epC', content='C',
                                          source_description='d', episode_type='text',
                                          entity_types=None, uuid='u-bp-C')
            assert await service.journal.count_unfinished() >= 2
            await service._breaker.record_failure(RateLimitError())
        finally:
            gate.set()
            for _ in range(200):
                if await service.journal.count_done() >= 2:
                    break
                await asyncio.sleep(0.05)
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_reflects_journal(self, tmp_path):
        gate = asyncio.Event()
        client = AsyncMock(spec=Graphiti)

        async def hold(**kw):
            await gate.wait()

        client.add_episode.side_effect = hold
        service = QueueService(make_config(tmp_path / 'snap.db'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            await service.add_episode(group_id='g1', name='ep', content='body',
                                      source_description='d', episode_type='text',
                                      entity_types=None, uuid='u-snap')
            snap = await service.get_resilience_snapshot()
            assert snap['journal']['enabled'] is True
            assert 'snap.db' in snap['journal']['path']
            # queue_depth/pending_episodes come from journal rows.
            assert snap['queue_depth'] >= 1
            assert snap['pending_episodes'] + snap['queue_depth'] >= 1
            assert 'pending' in snap['journal'] and 'processing' in snap['journal']
            assert snap['journal']['pending'] + snap['journal']['processing'] == snap['journal']['unfinished']
        finally:
            gate.set()
            for _ in range(200):
                if await service.journal.count_done() >= 1:
                    break
                await asyncio.sleep(0.05)
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_open_rejects_in_journal_mode(self, tmp_path):
        from graphiti_core.llm_client.errors import RateLimitError

        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_config(tmp_path / 'breaker.db'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for _ in range(3):
                await service._breaker.record_failure(RateLimitError())
            with pytest.raises(CircuitOpenError):
                await service.add_episode(group_id='g1', name='ep', content='body',
                                          source_description='d', episode_type='text',
                                          entity_types=None, uuid='u-breaker')
            assert await service.journal.count_pending() == 0
        finally:
            await service.close()


def make_builder():
    """Episode builder used by journal tests: forwards a subset of kwargs."""

    def _builder(plan: dict) -> dict:
        return {
            'name': plan['name'],
            'episode_body': plan['episode_body'],
            'source_description': plan['source_description'],
            'source': plan['source'],
            'group_id': plan['group_id'],
            'uuid': plan.get('uuid'),
        }

    return _builder


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError('Timed out waiting for condition')
        await asyncio.sleep(0.02)


class TestSerialPolicyAndLease:
    """Phase 2: serial-zone flag, leader claim guard, lease heartbeat/steward."""

    @pytest.mark.unit
    def test_plan_requires_serial_matrix(self):
        # No previous episodes -> auto-retrieve context -> serial.
        assert plan_requires_serial(make_plan(previous_episode_uuids=None)) is True
        assert plan_requires_serial(make_plan(previous_episode_uuids=[])) is True
        # Explicit already-written previous episodes -> parallel-safe.
        assert plan_requires_serial(make_plan(previous_episode_uuids=['u-1', 'u-2'])) is False
        # Saga starting/forking without its predecessor -> serial.
        assert plan_requires_serial(make_plan(saga='saga', saga_previous_episode_uuid=None)) is True
        # Saga with an explicit predecessor -> parallel-safe.
        assert (
            plan_requires_serial(
                make_plan(
                    previous_episode_uuids=['u-1'],
                    saga='saga',
                    saga_previous_episode_uuid='u-1',
                )
            )
            is False
        )
        # Community refresh -> serial.
        assert plan_requires_serial(make_plan(update_communities=True)) is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_claim_leader_picks_serial_first(self, journal):
        # Group has a serial row followed by a parallel-safe row.
        await journal.enqueue(make_plan(uuid='lp-serial'))
        await journal.enqueue(
            make_plan(uuid='lp-para', previous_episode_uuids=['u-written'])
        )
        # The leader claims the oldest pending row regardless of the flag.
        first = await journal.claim_next('g1', 'w-leader', leader=True)
        assert first is not None
        assert first['uuid'] == 'lp-serial'
        assert first['requires_serial'] == 1
        second = await journal.claim_next('g1', 'w-leader', leader=True)
        assert second is not None
        assert second['uuid'] == 'lp-para'
        assert second['requires_serial'] == 0
        await journal.mark_done(first['id'])
        await journal.mark_done(second['id'])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_claim_non_leader_skips_serial(self, journal):
        await journal.enqueue(make_plan(uuid='ns-serial'))
        assert await journal.claim_next('g1', 'w-follower', leader=False) is None
        serial = await journal.claim_next('g1', 'w-leader', leader=True)
        assert serial is not None and serial['requires_serial'] == 1
        await journal.mark_done(serial['id'])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_heartbeat_renews_lease_and_guards_owner(self, journal):
        await journal.enqueue(make_plan(uuid='hb-1'))
        claimed = await journal.claim_next('g1', 'w1')
        row_id = claimed['id']
        old_lease = claimed['lease_until']
        # Owner renews; the lease moves forward (ISO strings compare lexically).
        assert await journal.heartbeat(row_id, 'w1') is True
        cur = sqlite3.connect(journal.db_path)
        try:
            lease = cur.execute(
                'SELECT lease_until FROM episode_queue WHERE id=?', (row_id,)
            ).fetchone()[0]
        finally:
            cur.close()
        assert lease > old_lease
        # A different worker cannot renew it.
        assert await journal.heartbeat(row_id, 'w2') is False
        await journal.mark_done(row_id)
        assert await journal.heartbeat(row_id, 'w1') is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_requeue_stale_except_steal_guard(self, journal):
        my_worker = 'p-me'
        await journal.enqueue(make_plan(uuid='st-1'))
        claimed = await journal.claim_next('g1', my_worker)
        row_id = claimed['id']
        # Fresh lease: no one steals.
        assert await journal.requeue_stale_except(my_worker, grace_seconds=1) == 0
        assert await journal.requeue_stale_except('p-other', grace_seconds=1) == 0
        # Expire the lease out-of-band.
        cur = sqlite3.connect(journal.db_path)
        cur.execute(
            "UPDATE episode_queue SET lease_until='2000-01-01T00:00:00+00:00' WHERE id=?",
            (row_id,),
        )
        cur.commit()
        cur.close()
        # Our own row is never stolen, even when its lease is ancient.
        assert await journal.requeue_stale_except(my_worker, grace_seconds=1) == 0
        # A vanished owner's row is reclaimable.
        assert await journal.requeue_stale_except('p-other', grace_seconds=1) == 1
        assert await journal.count_pending() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_processing_stats_window(self, journal):
        await journal.enqueue(make_plan(uuid='ps-1'))
        row = await journal.claim_next('g1', 'w1')
        await journal.mark_done(row['id'])
        stats = await journal.processing_stats(window_seconds=3600.0)
        assert stats['processed'] >= 1
        assert stats['avg_processing_seconds'] >= 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_migration_adds_requires_serial(self, tmp_path):
        # A phase-1 journal has no requires_serial column; opening it with the
        # current QueueJournal must migrate it idempotently and serve claims.
        db = tmp_path / 'legacy.db'
        conn = sqlite3.connect(db)
        conn.executescript(
            'CREATE TABLE episode_queue ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, dedup_key TEXT NOT NULL, uuid TEXT, '
            'group_id TEXT NOT NULL, name TEXT NOT NULL, episode_body TEXT NOT NULL, '
            "plan_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending' "
            "CHECK (status IN ('pending', 'processing', 'done', 'failed')), "
            'attempt INTEGER NOT NULL DEFAULT 0, worker_id TEXT, lease_until TEXT, '
            'first_failure_ts TEXT, last_attempt_ts TEXT, next_retry_at TEXT, '
            'error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)'
        )
        conn.commit()
        conn.close()

        j = QueueJournal(db, lease_seconds=300.0)
        try:
            cols = {
                r[1]
                for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')
            }
            assert 'requires_serial' in cols
            # Enqueue + leader/non-leader claim work on the migrated schema.
            row_id, inserted = await j.enqueue(
                make_plan(uuid='mig-1', previous_episode_uuids=['u-x'])
            )
            assert inserted is True and row_id > 0
            claimed = await j.claim_next('g1', 'w1', leader=False)
            assert claimed is not None and claimed['requires_serial'] == 0
            await j.mark_done(row_id)
        finally:
            await j.close()


class TestSearchRaw:
    """Mini-phase A: raw-episode search over the journal (visibility of
    not-yet-materialized episodes). An episode must be findable right after
    enqueue, regardless of extraction."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_finds_by_name(self, journal):
        await journal.enqueue(
            make_plan(uuid='sr-name', name='Alpha ingress report', episode_body='body one')
        )
        await journal.enqueue(
            make_plan(uuid='sr-other', name='beta doc', episode_body='unrelated')
        )
        hits = await journal.search_raw('alpha')
        assert len(hits) == 1
        assert hits[0]['name'] == 'Alpha ingress report'
        assert hits[0]['status'] == 'pending'
        assert hits[0]['group_id'] == 'g1'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_finds_by_body_case_insensitive(self, journal):
        await journal.enqueue(
            make_plan(uuid='sr-b1', name='n-one', episode_body='The QUICK brown fox')
        )
        await journal.enqueue(
            make_plan(uuid='sr-b2', name='n-two', episode_body='lazy dog')
        )
        hits = await journal.search_raw('quick')
        assert len(hits) == 1
        assert hits[0]['name'] == 'n-one'
        assert hits[0]['snippet'] == 'The QUICK brown fox'
        # Case-insensitive in both directions (stored value and query).
        hits = await journal.search_raw('QUICK')
        assert len(hits) == 1 and hits[0]['name'] == 'n-one'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_entry_shape_and_materialized_flag(self, journal):
        await journal.enqueue(
            make_plan(uuid='sr-k', name='keys', episode_body='x' * 300)
        )
        hits = await journal.search_raw('keys')
        assert len(hits) == 1
        entry = hits[0]
        assert {
            'id', 'status', 'group_id', 'name', 'snippet',
            'created_at', 'updated_at', 'materialized',
        } <= set(entry)
        assert entry['materialized'] is False
        # Snippet is capped to the first ~200 characters of episode_body.
        assert entry['snippet'] == 'x' * 200

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_escapes_wildcards(self, journal):
        await journal.enqueue(
            make_plan(uuid='sr-w1', name='100% profit plan', episode_body='plain')
        )
        await journal.enqueue(
            make_plan(uuid='sr-w2', name='plain document', episode_body='plain')
        )
        await journal.enqueue(
            make_plan(uuid='sr-w3', name='under_score_note', episode_body='plain')
        )
        # '%' must be a literal, not a wildcard: only the row that literally
        # contains '100%' matches.
        hits = await journal.search_raw('100%')
        assert [h['name'] for h in hits] == ['100% profit plan']
        # '_' must be a literal, not a single-char wildcard.
        hits = await journal.search_raw('under_')
        assert [h['name'] for h in hits] == ['under_score_note']
        # A bare '%' query must not match every row: escaping turns it into a
        # literal '%', so only rows that literally contain one match.
        hits = await journal.search_raw('%')
        assert [h['name'] for h in hits] == ['100% profit plan']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_excludes_done_unless_include_done(self, journal):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='sr-d1', name='done name', episode_body='body')
        )
        await journal.mark_done(row_id)
        await journal.enqueue(
            make_plan(uuid='sr-d2', name='pending name', episode_body='body')
        )
        # Done rows are hidden by default.
        hits = await journal.search_raw('name')
        assert [h['name'] for h in hits] == ['pending name']
        # ... and included with include_done=True.
        hits = await journal.search_raw('name', include_done=True)
        assert {h['name'] for h in hits} == {'done name', 'pending name'}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_includes_failed_and_processing(self, journal):
        failed_id, _ = await journal.enqueue(
            make_plan(uuid='sr-f1', name='failed row', episode_body='body')
        )
        await journal.mark_failed(failed_id, 'boom')
        await journal.enqueue(
            make_plan(uuid='sr-p1', name='processing row', episode_body='body')
        )
        await journal.claim_next('g1', 'w')  # claims the pending row -> processing
        hits = await journal.search_raw('row')
        assert {h['name'] for h in hits} == {'failed row', 'processing row'}
        assert {h['status'] for h in hits} == {'failed', 'processing'}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_respects_limit_and_newest_first(self, journal):
        for i in range(5):
            await journal.enqueue(
                make_plan(uuid=f'sr-l{i}', name='limit match', episode_body='x')
            )
        hits = await journal.search_raw('limit match', limit=2)
        assert len(hits) == 2
        assert hits[0]['id'] > hits[1]['id']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_empty_query_and_bad_limit(self, journal):
        await journal.enqueue(
            make_plan(uuid='sr-e1', name='some ep', episode_body='body')
        )
        assert await journal.search_raw('') == []
        assert await journal.search_raw('   ') == []
        assert await journal.search_raw('some ep', limit=0) == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_queue_service_journal_search_raw_empty_when_journal_off(self):
        # MCP-level: with the journal disabled (default config) the read path
        # returns [] instead of raising, so search_raw_episodes never breaks a
        # journal-off server.
        service = QueueService(ResilienceConfig(journal_enabled=False, spool_enabled=False))
        try:
            assert await service.journal_search_raw('anything') == []
        finally:
            await service.close()


class TestPhase4ConfigDefaults:
    """Phase 4 config: bounded breakthrough + per-model reputation defaults.

    All new fields default to behaviour-preserving values (backwards
    compatible: an empty YAML / a stock ResilienceConfig leaves the write path
    exactly as it was before Phase 4).
    """

    @pytest.mark.unit
    def test_phase4_defaults_preserve_current_behaviour(self):
        cfg = ResilienceConfig()
        # Bounded breakthrough intake is off by default: open breaker rejects.
        assert cfg.enqueue_breakthrough_max_pending == 0
        # Per-model reputation defaults (300s window, 5 429/min, 60s cooldown).
        assert cfg.model_reputation_window_seconds == 300.0
        assert cfg.model_reputation_429_per_min_threshold == 5.0
        assert cfg.model_reputation_429_cooldown_seconds == 60.0

    @pytest.mark.unit
    def test_phase4_constraints(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ResilienceConfig(enqueue_breakthrough_max_pending=-1)
        with pytest.raises(pydantic.ValidationError):
            ResilienceConfig(model_reputation_window_seconds=0.5)
        with pytest.raises(pydantic.ValidationError):
            ResilienceConfig(model_reputation_429_per_min_threshold=-1.0)
        with pytest.raises(pydantic.ValidationError):
            ResilienceConfig(model_reputation_429_cooldown_seconds=-1.0)

    @pytest.mark.unit
    def test_phase4_values_roundtrip(self):
        cfg = ResilienceConfig(
            enqueue_breakthrough_max_pending=7,
            model_reputation_window_seconds=120.0,
            model_reputation_429_per_min_threshold=10.0,
            model_reputation_429_cooldown_seconds=15.0,
        )
        assert cfg.enqueue_breakthrough_max_pending == 7
        assert cfg.model_reputation_window_seconds == 120.0
        assert cfg.model_reputation_429_per_min_threshold == 10.0
        assert cfg.model_reputation_429_cooldown_seconds == 15.0


class TestPhase4BreakthroughIntake:
    """Phase 4: bounded breakthrough enqueue while the circuit breaker is OPEN."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_open_breaker_breakthrough_allows_bounded_enqueue(self, tmp_path):
        from graphiti_core.llm_client.errors import RateLimitError

        cfg = make_config(tmp_path / 'brk.db').model_copy(
            update=dict(journal_max_pending=5, enqueue_breakthrough_max_pending=3)
        )
        # NOT initialized: no pool workers, so row counts stay deterministic.
        service = QueueService(cfg)
        try:
            for _ in range(3):
                await service._breaker.record_failure(RateLimitError())
            assert service._breaker.state == 'open'
            # Both zones fill while the breaker stays open (no CircuitOpenError).
            for i in range(5):
                await service._add_episode_task_journal('g1', make_plan(uuid=f'brk-{i}'))
            assert await service.journal.count_pending() == 5
            # Breakthrough zone: 3 more accepted past journal_max_pending.
            for i in range(3):
                await service._add_episode_task_journal('g1', make_plan(uuid=f'boom-{i}'))
            assert await service.journal.count_pending() == 8
            # The breakthrough ceiling is journal_max_pending + breakthrough =
            # 5 + 3 = 8: one more row would exceed it -> capacity reject, not
            # backpressure from the (now irrelevant) legacy max_queue_depth.
            with pytest.raises(QueueCapacityExceeded):
                await service._add_episode_task_journal('g1', make_plan(uuid='overflow'))
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_open_breaker_breakthrough_default_rejects(self, tmp_path):
        """breakthrough=0 (default): an open breaker still rejects - regression."""
        from graphiti_core.llm_client.errors import RateLimitError

        service = QueueService(make_config(tmp_path / 'brk0.db'))
        try:
            for _ in range(3):
                await service._breaker.record_failure(RateLimitError())
            assert service._breaker.state == 'open'
            with pytest.raises(CircuitOpenError):
                await service._add_episode_task_journal('g1', make_plan(uuid='u-brk0'))
            assert await service.journal.count_pending() == 0
        finally:
            await service.close()


class TestJournalMaxPending:
    """Soft ceiling for durable-journal intake (journal_max_pending).

    Decoupled from the legacy in-memory max_queue_depth (20): the journal path
    backpressures at pending+processing >= journal_max_pending, while the legacy
    in-memory path keeps its own max_queue_depth unchanged.
    """

    @pytest.mark.unit
    def test_default_is_500(self):
        cfg = ResilienceConfig()
        assert cfg.journal_max_pending == 500
        # The legacy in-memory ceiling is untouched.
        assert cfg.max_queue_depth == 20

    @pytest.mark.unit
    def test_override_in_config(self):
        assert ResilienceConfig(journal_max_pending=3).journal_max_pending == 3
        assert ResilienceConfig(journal_max_pending=0).journal_max_pending == 0
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ResilienceConfig(journal_max_pending=-1)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_intake_works_up_to_the_ceiling(self, tmp_path):
        """With journal_max_pending=3, rows 1..3 are accepted, the 4th rejects."""
        service = QueueService(
            make_config(tmp_path / 'cap.db').model_copy(update={'journal_max_pending': 3})
        )
        try:
            for i in range(3):
                await service._add_episode_task_journal('g1', make_plan(uuid=f'cap-{i}'))
            assert await service.journal.count_unfinished() == 3
            with pytest.raises(QueueCapacityExceeded):
                await service._add_episode_task_journal('g1', make_plan(uuid='cap-overflow'))
            # The rejected row was not persisted.
            assert await service.journal.count_unfinished() == 3
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_journal_off_preserves_legacy_max_queue_depth_20(self):
        """Regression: journal disabled -> the in-memory path still uses
        max_queue_depth (default 20), NOT journal_max_pending."""
        service = QueueService(ResilienceConfig(spool_enabled=False))
        assert service._journal is None
        gate = asyncio.Event()

        async def block():
            await gate.wait()

        try:
            for _ in range(20):
                await service.add_episode_task('g1', block)
            assert service._queue_depth == 20
            with pytest.raises(QueueCapacityExceeded):
                await service.add_episode_task('g1', block)
            assert service._queue_depth == 20
        finally:
            gate.set()
            await service._episode_queues['g1'].join()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_exposes_max_pending(self, tmp_path):
        """/health journal block surfaces the effective journal ceiling."""
        service = QueueService(
            make_config(tmp_path / 'mp.db').model_copy(update={'journal_max_pending': 7})
        )
        try:
            snap = await service.get_resilience_snapshot()
            assert snap['journal']['enabled'] is True
            assert snap['journal']['max_pending'] == 7
        finally:
            await service.close()


class TestMarkVerified:
    """In-queue fact-check marker: a validator stamps verified_by/verified_at
    directly on a raw journal row, without waiting for graph materialization."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_verified_sets_by_and_at(self, journal):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='mv-1', name='verify me', episode_body='raw body')
        )
        assert (
            await journal.mark_verified(
                row_id, by='validator-hard', at='2026-10-09T10:00:00+00:00'
            )
            is True
        )
        hits = await journal.search_raw('verify me')
        assert len(hits) == 1
        entry = hits[0]
        assert entry['verified_by'] == 'validator-hard'
        assert entry['verified_at'] == '2026-10-09T10:00:00+00:00'
        # The fact-check marker does not touch the row status.
        assert entry['status'] == 'pending'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_verified_defaults_at_to_now(self, journal):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='mv-now', name='now stamp', episode_body='b')
        )
        assert await journal.mark_verified(row_id, by='validator') is True
        hits = await journal.search_raw('now stamp')
        assert hits[0]['verified_by'] == 'validator'
        assert hits[0]['verified_at'] is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_verified_second_call_updates_to_latest(self, journal):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='mv-2', name='recheck', episode_body='b')
        )
        first = '2026-10-09T10:00:00+00:00'
        second = '2026-10-09T11:30:00+00:00'
        assert await journal.mark_verified(row_id, by='validator', at=first) is True
        assert await journal.mark_verified(row_id, by='validator-hard', at=second) is True
        hits = await journal.search_raw('recheck')
        assert hits[0]['verified_by'] == 'validator-hard'
        assert hits[0]['verified_at'] == second
        # Still pending after every fact-check stamp.
        assert hits[0]['status'] == 'pending'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_verified_unknown_row_returns_false(self, journal):
        assert await journal.mark_verified(999_999, by='validator') is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mark_verified_does_not_change_status(self, journal):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='mv-st', name='status stable', episode_body='b')
        )
        await journal.mark_verified(row_id, by='validator', at='2026-10-09T10:00:00+00:00')
        # Still pending, still claimable by a worker.
        assert await journal.count_pending() == 1
        claimed = await journal.claim_next('g1', 'w')
        assert claimed is not None and claimed['id'] == row_id
        # The verified marker survives the claim (fact-check provenance kept).
        assert claimed['verified_by'] == 'validator'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_migration_adds_verified_columns(self, tmp_path):
        # A pre-verified journal has no verified_by/verified_at columns; opening
        # it with the current QueueJournal must add them idempotently, and
        # search/claim must work on the migrated schema.
        db = tmp_path / 'legacy-verified.db'
        conn = sqlite3.connect(db)
        conn.executescript(
            'CREATE TABLE episode_queue ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, dedup_key TEXT NOT NULL, uuid TEXT, '
            'group_id TEXT NOT NULL, name TEXT NOT NULL, episode_body TEXT NOT NULL, '
            "plan_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending' "
            "CHECK (status IN ('pending', 'processing', 'done', 'failed')), "
            'attempt INTEGER NOT NULL DEFAULT 0, worker_id TEXT, lease_until TEXT, '
            'first_failure_ts TEXT, last_attempt_ts TEXT, next_retry_at TEXT, '
            'error TEXT, requires_serial INTEGER NOT NULL DEFAULT 0, '
            'created_at TEXT NOT NULL, updated_at TEXT NOT NULL)'
        )
        conn.commit()
        conn.close()

        j = QueueJournal(db, lease_seconds=300.0)
        try:
            cols = {
                r[1]
                for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')
            }
            assert 'verified_by' in cols
            assert 'verified_at' in cols
            # Fresh writes + fact-check + search work on the migrated schema.
            row_id, inserted = await j.enqueue(
                make_plan(uuid='mv-mig', name='migrated row', episode_body='body')
            )
            assert inserted is True and row_id > 0
            assert (
                await j.mark_verified(
                    row_id, by='validator', at='2026-10-09T10:00:00+00:00'
                )
                is True
            )
            hits = await j.search_raw('migrated row')
            assert hits[0]['verified_by'] == 'validator'
            # Claim uses the full _ROW_COLUMNS on the migrated table.
            claimed = await j.claim_next('g1', 'w')
            assert claimed is not None and claimed['id'] == row_id
            assert claimed['verified_by'] == 'validator'
        finally:
            await j.close()
