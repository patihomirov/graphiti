"""Phase-2 tests: the global journal worker pool.

Covers parallel cross-group processing (exactly-once), strict per-group FIFO
(serial zones are never parallelized even with a larger pool), the global LLM
semaphore cap shared by the pool, and the graceful-drain cancellation of
in-flight pool workers.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from graphiti_core import Graphiti

from config.schema import ResilienceConfig
from services.queue_service import QueueService


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


def make_pool_config(db_path, workers: int = 2, **over) -> ResilienceConfig:
    cfg = ResilienceConfig(
        journal_enabled=True,
        journal_path=str(db_path),
        journal_lease_seconds=300.0,
        journal_workers=workers,
        spool_enabled=False,
        max_queue_depth=100,
        journal_steward_interval_seconds=60.0,
        journal_grace_seconds=60.0,
    )
    return cfg.model_copy(update=over)


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError('Timed out waiting for condition')
        await asyncio.sleep(0.02)


class TestWorkerPool:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pool_processes_all_groups_exactly_once(self, tmp_path):
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_pool_config(tmp_path / 'pool.db', workers=2))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(8):
                group = f'g{(i % 2) + 1}'
                await service.add_episode(
                    group_id=group, name=f'ep{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'pool-{i}',
                )
            await wait_until(lambda: client.add_episode.await_count >= 8)
            await asyncio.sleep(0.05)
            assert client.add_episode.await_count == 8
            assert await service.journal.count_done() == 8
            assert await service.journal.count_unfinished() == 0
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pool_single_group_stays_fifo(self, tmp_path):
        # Two workers, one group: the per-group lock must serialize processing
        # in claim (row-id) order - no overlap even with a bigger pool.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0
        seen = []

        async def serial(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen.append(kwargs.get('uuid'))
            await asyncio.sleep(0.01)
            active -= 1

        client.add_episode.side_effect = serial
        service = QueueService(make_pool_config(tmp_path / 'fifo.db', workers=2))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(6):
                await service.add_episode(
                    group_id='g1', name=f'e{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'fifo-{i}',
                )
            await wait_until(lambda: client.add_episode.await_count >= 6)
            assert max_active == 1
            assert seen == [f'fifo-{i}' for i in range(6)]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pool_parallel_groups_are_concurrent(self, tmp_path):
        # Two independent groups, two workers: processing overlaps.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0

        async def concurrent(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1

        client.add_episode.side_effect = concurrent
        service = QueueService(make_pool_config(tmp_path / 'par.db', workers=2))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(6):
                group = f'g{(i % 2) + 1}'
                await service.add_episode(
                    group_id=group, name=f'e{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'par-{i}',
                )
            await wait_until(lambda: client.add_episode.await_count >= 6)
            assert max_active >= 2
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_global_semaphore_caps_concurrency(self, tmp_path):
        # Four workers on four groups, but semaphore_limit=2: at most two
        # add_episode calls in flight at any time.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0

        async def capped(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1

        client.add_episode.side_effect = capped
        service = QueueService(
            make_pool_config(tmp_path / 'sem.db', workers=4, semaphore_limit=2)
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(8):
                group = f'g{(i % 4) + 1}'
                await service.add_episode(
                    group_id=group, name=f'e{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'sem-{i}',
                )
            await wait_until(lambda: client.add_episode.await_count >= 8)
            assert 1 <= max_active <= 2
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pool_drain_cancels_in_flight(self, tmp_path):
        # An in-flight row past the grace window is cancelled by drain(); the
        # row remains 'processing' (healed on next boot) and is not lost.
        gate = asyncio.Event()
        client = AsyncMock(spec=Graphiti)

        async def hold(**kwargs):
            await gate.wait()

        client.add_episode.side_effect = hold
        service = QueueService(make_pool_config(tmp_path / 'drain.db', workers=1))
        await service.initialize(client, episode_builder=make_builder())
        try:
            await service.add_episode(
                group_id='g1', name='ep', content='body', source_description='d',
                episode_type='text', entity_types=None, uuid='drain-1',
            )
            await asyncio.sleep(0.2)  # let the pool claim it
            assert await service.journal.count_processing() == 1
            await service.drain(wait_current_seconds=0.1)
            assert await service.journal.count_processing() == 1
        finally:
            gate.set()
            await asyncio.sleep(0.2)
            await service.close()


class TestParallelFollowers:
    """Phase 5: serial-leader + parallel-follower within a single group.

    The first worker to take a group's lock is its serial leader (strict FIFO,
    claims every row); once the lock is busy, the other pool workers become
    followers that drain only ``requires_serial=0`` rows in parallel. The
    global semaphore still bounds total in-flight across both roles.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_followers_parallelize_non_serial_rows(self, tmp_path):
        # One group, 3 workers: one serial row (leader) then 4 parallel-safe
        # rows. The non-serial rows must overlap in flight (max_active >= 2),
        # everything is processed exactly once, and nothing is double-claimed.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0
        seen = []

        async def work(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen.append(kwargs.get('uuid'))
            await asyncio.sleep(0.02)
            active -= 1

        client.add_episode.side_effect = work
        service = QueueService(make_pool_config(tmp_path / 'fol.db', workers=3))
        await service.initialize(client, episode_builder=make_builder())
        try:
            await service.add_episode(
                group_id='g1', name='s0', content='body', source_description='d',
                episode_type='text', entity_types=None, uuid='serial-0',
            )
            for i in range(4):
                await service.add_episode(
                    group_id='g1', name=f'p{i}', content='body', source_description='d',
                    episode_type='text', entity_types=None, uuid=f'para-{i}',
                    previous_episode_uuids=['u-done'],
                )
            await wait_until(lambda: client.add_episode.await_count >= 5)
            await asyncio.sleep(0.05)  # let late claims settle
            assert client.add_episode.await_count == 5
            assert await service.journal.count_done() == 5
            assert await service.journal.count_unfinished() == 0
            # Each uuid claimed exactly once (no watchdog/steal double-claim).
            assert sorted(seen) == [f'para-{i}' for i in range(4)] + ['serial-0']
            # Non-serial rows ran in parallel while the serial row was led.
            assert max_active >= 2
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_serial_rows_stay_strictly_fifo_with_followers(self, tmp_path):
        # 4 workers, one group: 5 serial rows (no explicit previous). Followers
        # must NEVER touch them - only the leader runs, strictly by row id.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0
        seen = []

        async def serial(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen.append(kwargs.get('uuid'))
            await asyncio.sleep(0.01)
            active -= 1

        client.add_episode.side_effect = serial
        service = QueueService(make_pool_config(tmp_path / 'sf.db', workers=4))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(5):
                await service.add_episode(
                    group_id='g1', name=f'e{i}', content='body', source_description='d',
                    episode_type='text', entity_types=None, uuid=f'ser-{i}',
                )
            await wait_until(lambda: client.add_episode.await_count >= 5)
            assert max_active == 1
            assert seen == [f'ser-{i}' for i in range(5)]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_follower_concurrency_capped_by_semaphore(self, tmp_path):
        # 4 workers, all non-serial rows, but semaphore_limit=1: the global
        # LLM semaphore must still cap in-flight add_episode to 1 even though
        # followers would happily parallelize the non-serial backlog.
        client = AsyncMock(spec=Graphiti)
        active = 0
        max_active = 0

        async def capped(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1

        client.add_episode.side_effect = capped
        service = QueueService(
            make_pool_config(tmp_path / 'semf.db', workers=4, semaphore_limit=1)
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(6):
                await service.add_episode(
                    group_id='g1', name=f'p{i}', content='body', source_description='d',
                    episode_type='text', entity_types=None, uuid=f'pf-{i}',
                    previous_episode_uuids=['u-done'],
                )
            await wait_until(lambda: client.add_episode.await_count >= 6)
            assert max_active == 1
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_backlog_drains_exactly_once(self, tmp_path):
        # 3 workers, one group with a mix of serial and parallel-safe rows:
        # every uuid is processed exactly once and the backlog drains fully.
        client = AsyncMock(spec=Graphiti)
        done_uuids = []
        lock = asyncio.Lock()

        async def record(**kwargs):
            u = kwargs.get('uuid')
            await asyncio.sleep(0.005)
            async with lock:
                done_uuids.append(u)

        client.add_episode.side_effect = record
        service = QueueService(make_pool_config(tmp_path / 'mix.db', workers=3))
        await service.initialize(client, episode_builder=make_builder())
        expected = []
        try:
            for i in range(8):
                if i % 3 == 0:
                    # serial row
                    await service.add_episode(
                        group_id='g1', name=f'e{i}', content='body', source_description='d',
                        episode_type='text', entity_types=None, uuid=f'mix-s{i}',
                    )
                    expected.append(f'mix-s{i}')
                else:
                    await service.add_episode(
                        group_id='g1', name=f'e{i}', content='body', source_description='d',
                        episode_type='text', entity_types=None, uuid=f'mix-p{i}',
                        previous_episode_uuids=['u-done'],
                    )
                    expected.append(f'mix-p{i}')
            await wait_until(lambda: client.add_episode.await_count >= 8)
            await asyncio.sleep(0.05)
            assert client.add_episode.await_count == 8
            assert await service.journal.count_done() == 8
            assert await service.journal.count_unfinished() == 0
            assert sorted(done_uuids) == sorted(expected)
        finally:
            await service.close()


class TestLastDoneEpisodeUuidGraphFallback:
    """Throughput fix 2026-10-21: ``last_done_episode_uuid`` must not silently
    return ``None`` when journal rows carry no uuid (uuid lives only on Episodic
    nodes in the graph). The Neo4j fallback answers with the newest materialized
    episode of the group, keeping ``auto_previous_episode_uuid`` alive.
    """

    def _service(self, tmp_path, **over) -> QueueService:
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_pool_config(tmp_path, workers=1, **over))
        return client, service

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_journal_uuid_wins_without_graph_touch(self, tmp_path):
        # A done journal row WITH an explicit uuid answers directly; the graph
        # fallback must not even be reached (driver left unmocked would raise).
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_pool_config(tmp_path / 'j.db', workers=1))
        await service.initialize(client, episode_builder=make_builder())
        try:
            await service.add_episode(
                group_id='g1', name='x', content='body', source_description='d',
                episode_type='text', entity_types=None, uuid='journal-uuid',
            )
            await wait_until(lambda: client.add_episode.await_count >= 1)
            await asyncio.sleep(0.05)
            assert await service.last_done_episode_uuid('g1') == 'journal-uuid'
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_falls_back_to_graph_when_journal_has_no_uuid(self, tmp_path):
        # Journal has done rows but none carrying a uuid (the production
        # reality) -> the Neo4j fallback returns the newest materialized
        # episode filtered by group_id.
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_pool_config(tmp_path / 'g.db', workers=1))
        await service.initialize(client, episode_builder=make_builder())
        try:
            driver = AsyncMock()
            future_result = [None]

            async def fake_execute_query(query, **kwargs):
                return future_result[0]

            driver.execute_query = AsyncMock(side_effect=fake_execute_query)
            client.driver = driver

            future_result[0] = ([{'uuid': 'graph-uuid-1'}], None, None)
            assert await service.last_done_episode_uuid('main') == 'graph-uuid-1'
            call = driver.execute_query.call_args
            assert call.kwargs.get('params', {}).get('group_id') == 'main'
            assert 'e.group_id' in call.args[0]
            assert 'ORDER BY e.created_at DESC' in call.args[0]

            # No materialized episodes -> None (auto_previous falls back to serial).
            future_result[0] = ([], None, None)
            assert await service.last_done_episode_uuid('main') is None
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_graph_fallback_survives_driver_errors(self, tmp_path):
        # A broken/unreachable driver must not leak an exception upward; the
        # opt-in flow just falls back to serial processing (returns None).
        client = AsyncMock(spec=Graphiti)
        service = QueueService(make_pool_config(tmp_path / 'e.db', workers=1))
        await service.initialize(client, episode_builder=make_builder())
        try:
            driver = AsyncMock()

            async def boom(query, **kwargs):
                raise RuntimeError('neo4j down')

            driver.execute_query = AsyncMock(side_effect=boom)
            client.driver = driver
            assert await service.last_done_episode_uuid('main') is None
        finally:
            await service.close()

