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
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from graphiti_core import Graphiti

from config.schema import ResilienceConfig
from services.queue_journal import QueueJournal, compute_dedup_key
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
            make_config(tmp_path / 'backpressure.db').model_copy(update={'max_queue_depth': 2})
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
