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
