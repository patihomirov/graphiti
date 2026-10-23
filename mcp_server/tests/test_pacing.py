"""Integration tests for always-active pacing inside the journal worker (2026-10-21).

Covers the failover gate ``_channel_wait_or_skip`` (a saturated primary channel
routes the episode to the next candidate instead of queueing), the inter-episode
pause between claims (a worker that just finished an episode sleeps before
picking up the next one, so a pool never restarts a whole batch at once), and
the /health ``pacing`` block. Everything runs against a tmp_path SQLite journal
with fake graphiti/LLM clients - no network.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from graphiti_core import Graphiti

from config.schema import ResilienceConfig
from services.queue_service import QueueService
from services.rate_limiter import ChannelRateLimiter


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


def make_config(db_path, **over) -> ResilienceConfig:
    cfg = dict(
        journal_enabled=True,
        journal_path=str(db_path),
        journal_lease_seconds=300.0,
        journal_workers=1,
        spool_enabled=False,
        spool_backoff_base_seconds=0.01,
    )
    cfg.update(over)
    return ResilienceConfig(**cfg)


def make_builder():
    """Forward a subset of kwargs, mirroring the production episode_builder."""

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


class FakeLLM:
    """Fake LLM client exposing the model-override hook and recording calls."""

    def __init__(self, model: str = 'model-A'):
        self.model = model
        self.seen: list[str | None] = []
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self.seen.append(model)
        self._override = model


class PacedGraphiti:
    """Fake graphiti client whose add_episode records the winning channel."""

    def __init__(self, llm: FakeLLM):
        self.llm_client = llm
        self.calls: list[tuple[str | None, dict]] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append((model, kwargs))
        return None


async def wait_done(service: QueueService, count: int = 1, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await service.journal.count_done() >= count:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'Timed out waiting for {count} done rows')


async def enqueue(service: QueueService, uuid: str) -> None:
    await service.add_episode(
        group_id='g1',
        name=f'ep-{uuid}',
        content='body',
        source_description='d',
        episode_type='text',
        entity_types=None,
        uuid=uuid,
    )


class TestFailoverGate:
    """``_channel_wait_or_skip`` decides proceed-vs-skip honestly."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_skips_saturated_channel(self, tmp_path):
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(
            make_config(tmp_path / 'g1.db', channel_max_calls_per_minute=2)
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            # Replace with a limiter whose window is huge, then fill the channel.
            service._rate_limiter = ChannelRateLimiter(
                max_calls_per_minute=1, window_seconds=3600.0
            )
            service._channel_wait_abort_seconds = 60.0
            await service._rate_limiter.acquire('ozon|model-A')
            # model-A is saturated (would wait ~3600s) -> gate says SKIP.
            assert await service._channel_wait_or_skip('model-A') is False
            # A fresh channel has a free slot -> gate says PROCEED.
            assert await service._channel_wait_or_skip('model-B') is True
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_proceeds_when_limiter_off_or_no_candidate(self, tmp_path):
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(make_config(tmp_path / 'g2.db'))  # limiter OFF (0)
        await service.initialize(client, episode_builder=make_builder())
        try:
            assert service._rate_limiter.enabled is False
            assert await service._channel_wait_or_skip('model-A') is True
            assert await service._channel_wait_or_skip(None) is True
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_short_wait_proceeds(self, tmp_path):
        """A bounded wait (<= abort window) still proceeds - the client paces."""
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(
            make_config(tmp_path / 'g3.db', channel_max_calls_per_minute=1)
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            service._rate_limiter = ChannelRateLimiter(
                max_calls_per_minute=10, window_seconds=1.0
            )
            service._channel_wait_abort_seconds = 60.0
            for _ in range(9):
                await service._rate_limiter.acquire('ozon|model-A')
            # 9/10 slots used, next would wait ~1s (< 60s abort) -> proceed.
            assert await service._channel_wait_or_skip('model-A') is True
        finally:
            await service.close()


class TestFailoverE2E:
    """A saturated primary routes the episode to a free fallback channel."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_saturated_primary_routes_to_fallback(self, tmp_path):
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(
            make_config(
                tmp_path / 'e2e.db',
                model_fallbacks=['model-B'],
                channel_max_calls_per_minute=5,
            )
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            service._channel_wait_abort_seconds = 1.0
            # Fill the primary channel to saturation (5/5 slots in the window).
            for _ in range(5):
                await service._rate_limiter.acquire('ozon|model-A')
            await enqueue(service, 'u-gate')
            await wait_done(service)
            # The episode was processed on the free fallback, not the saturated primary.
            assert client.calls[-1][0] == 'model-B'
            assert await service.journal.count_done() == 1
        finally:
            await service.close()


class TestInterEpisodePause:
    """A worker that just finished an episode pauses before its next claim."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_worker_pauses_between_consecutive_claims(self, tmp_path):
        client = AsyncMock(spec=Graphiti)
        starts: list[float] = []

        async def work(**kwargs):
            starts.append(time.monotonic())
            await asyncio.sleep(0.01)

        client.add_episode.side_effect = work
        service = QueueService(
            make_config(
                tmp_path / 'pause.db',
                storm_min_request_interval_seconds=0.2,
            )
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(2):
                await service.add_episode(
                    group_id='g1', name=f'e{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'pause-{i}',
                )
            deadline = asyncio.get_running_loop().time() + 5.0
            while client.add_episode.await_count < 2:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError('timed out waiting for 2 claims')
                await asyncio.sleep(0.02)
            assert len(starts) == 2
            gap = starts[1] - starts[0]
            # interval (0.2s) * active_channels (>=1) - never instant restart.
            assert gap >= 0.15
            # Wait for the second claim to actually finish before checking done.
            while await service.journal.count_done() < 2:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError('timed out waiting for 2 done rows')
                await asyncio.sleep(0.02)
            assert await service.journal.count_done() == 2
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_interval_means_no_pause(self, tmp_path):
        client = AsyncMock(spec=Graphiti)
        starts: list[float] = []

        async def work(**kwargs):
            starts.append(time.monotonic())
            await asyncio.sleep(0.001)

        client.add_episode.side_effect = work
        service = QueueService(make_config(tmp_path / 'nopause.db'))  # interval 0
        await service.initialize(client, episode_builder=make_builder())
        try:
            for i in range(2):
                await service.add_episode(
                    group_id='g1', name=f'e{i}', content='body',
                    source_description='d', episode_type='text',
                    entity_types=None, uuid=f'np-{i}',
                )
            deadline = asyncio.get_running_loop().time() + 5.0
            while client.add_episode.await_count < 2:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError('timed out waiting for 2 claims')
                await asyncio.sleep(0.005)
            assert len(starts) == 2
            gap = starts[1] - starts[0]
            assert gap < 0.05  # no forced pause
        finally:
            await service.close()


class TestHealthPacing:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_exposes_pacing_block(self, tmp_path):
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(
            make_config(
                tmp_path / 'h.db',
                model_fallbacks=['model-B'],
                channel_max_calls_per_minute=60,
                storm_min_request_interval_seconds=5.0,
            )
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            snap = await service.get_resilience_snapshot()
            pacing = snap['pacing']
            assert pacing['enabled'] is True
            assert pacing['max_calls_per_minute'] == 60
            assert pacing['inter_episode_interval_seconds'] == 5.0
            assert pacing['active_channels'] >= 1
            assert 'per_channel' in pacing
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_pacing_disabled_by_default(self, tmp_path):
        llm = FakeLLM('model-A')
        client = PacedGraphiti(llm)
        service = QueueService(make_config(tmp_path / 'hd.db'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            snap = await service.get_resilience_snapshot()
            assert snap['pacing']['enabled'] is False
            assert snap['pacing']['max_calls_per_minute'] == 0
        finally:
            await service.close()
