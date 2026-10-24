"""Unit + e2e tests for the zai overflow extraction channel (Feature B).

Covers the second ``OpenAIGenericClient`` built from the ``providers.zai`` slot
(secret resolved from a ``{file:...}`` reference, never logged), the
``(provider, model)`` channel keys used by failover candidates, the routing of a
whole episode through the zai Graphiti instance when the ozon primary is in a
429 storm, and the auto-rollback that marks the zai channel broken after
``zai_loss_strikes`` consecutive proxy losses. Everything runs against a
tmp_path SQLite journal + fake graphiti/LLM clients - no network, no
graphiti-core data.
"""

import json
import sqlite3

import httpx
import pytest
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig, ResilienceConfig
from services.factories import LLMClientFactory, resolve_api_key
from services.queue_service import QueueService


def make_config(db_path, **over) -> ResilienceConfig:
    cfg = dict(
        journal_enabled=True,
        journal_path=str(db_path),
        journal_lease_seconds=300.0,
        spool_enabled=False,
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


class FakeClock:
    """Monotonic clock shared by the reputation tracker and the zai channel."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeLLM:
    """Fake LLM exposing the model-override hook like the real client."""

    def __init__(self, model: str):
        self.model = model
        self.seen: list[str | None] = []
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self.seen.append(model)
        self._override = model


class FakeGraphiti:
    """Fake graphiti client whose add_episode fails the first N calls."""

    def __init__(self, llm: FakeLLM, fail_calls: int = 0, fail_error=RateLimitError):
        self.llm_client = llm
        self.fail_remaining = fail_calls
        self.fail_error = fail_error
        self.calls: list[tuple[str | None, dict]] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append((model, kwargs))
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            if isinstance(self.fail_error, type):
                raise self.fail_error(f'transient failure on {model}')
            raise self.fail_error
        return None


def attempted_for(journal, uuid: str) -> list[str]:
    conn = sqlite3.connect(journal.db_path)
    try:
        row = conn.execute(
            'SELECT attempted_models FROM episode_queue WHERE uuid=?', (uuid,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return []
    return json.loads(row[0])


async def wait_done(service: QueueService, count: int = 1, timeout: float = 5.0) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await service.journal.count_done() >= count:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'Timed out waiting for {count} done rows')


async def wait_cond(cond, timeout: float = 10.0, msg: str = 'condition') -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'Timed out waiting for {msg}')


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


class TestSecretResolution:
    """{file:...} secret references resolve without logging the value."""

    @pytest.mark.unit
    def test_literal_key_passes_unchanged(self):
        assert resolve_api_key('plain-key') == 'plain-key'

    @pytest.mark.unit
    def test_file_reference_resolves(self, tmp_path):
        token = tmp_path / 'token'
        token.write_text('secret-token\n')
        assert resolve_api_key(f'{{file:{token}}}') == 'secret-token'

    @pytest.mark.unit
    def test_env_expansion_inside_file_reference(self, tmp_path, monkeypatch):
        token = tmp_path / 'token'
        token.write_text('env-secret')
        monkeypatch.setenv('TEST_HOME', str(tmp_path))
        assert resolve_api_key('{file:{env:TEST_HOME}/token}') == 'env-secret'

    @pytest.mark.unit
    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            resolve_api_key('')

    @pytest.mark.unit
    def test_empty_file_raises(self, tmp_path):
        token = tmp_path / 'empty-token'
        token.write_text('   \n')
        with pytest.raises(ValueError):
            resolve_api_key(f'{{file:{token}}}')


class TestCreateZaiFactory:
    """The factory builds a zai OpenAIGenericClient from the providers.zai slot."""

    def _config(
        self, api_key: str, api_url: str = 'http://127.0.0.4:3128/zai/openai/v1'
    ) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model='active-model',
            max_tokens=2048,
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='k', api_url='http://ox/v1'),
                zai=OpenAIProviderConfig(api_key=api_key, api_url=api_url),
            ),
        )

    @pytest.mark.unit
    def test_builds_generic_client_from_zai_slot(self, tmp_path):
        token = tmp_path / 'token'
        token.write_text('zai-vendor-token')
        config = self._config(f'{{file:{token}}}')
        client = LLMClientFactory.create_zai(config, zai_model='zai/glm-5.3')
        assert isinstance(client, OpenAIGenericClient)
        # The channel key stays provider-qualified; the wire model (what the zai
        # proxy expects) drops the provider prefix.
        assert client.model == 'glm-5.3'
        assert client.structured_output_mode == 'json_object'
        # Headroom above the active budget for zai reasoning models.
        assert client.max_tokens == 8192
        assert client.config.api_key == 'zai-vendor-token'
        assert client.config.base_url == 'http://127.0.0.4:3128/zai/openai/v1'

    @pytest.mark.unit
    def test_max_tokens_uses_active_budget_with_headroom(self, tmp_path):
        token = tmp_path / 'token'
        token.write_text('t')
        config = self._config(f'{{file:{token}}}')
        config.max_tokens = 32768
        client = LLMClientFactory.create_zai(config)
        assert client.max_tokens == 32768

    @pytest.mark.unit
    def test_missing_zai_slot_raises(self):
        config = LLMConfig(
            provider='openai',
            model='m',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='k', api_url='http://ox/v1')
            ),
        )
        with pytest.raises(ValueError):
            LLMClientFactory.create_zai(config)


class TestChannelCandidates:
    """_failover_candidates emits (provider, model) channels: ozon, then zai, then fallbacks."""

    async def _service(self, tmp_path, **over) -> QueueService:
        return QueueService(make_config(tmp_path / 'c.db', **over))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_channel_inserted_between_active_and_fallbacks(self, tmp_path):
        llm = FakeLLM('model-A')
        service = await self._service(
            tmp_path, zai_fallback_enabled=True, model_fallbacks=['model-B']
        )
        service._graphiti_client_zai = object()
        try:
            assert service._failover_candidates([], llm) == [
                'model-A',
                'zai/glm-5.3',
                'model-B',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_excluded_when_disabled(self, tmp_path):
        llm = FakeLLM('model-A')
        service = await self._service(tmp_path, zai_fallback_enabled=False)
        service._graphiti_client_zai = object()
        try:
            assert service._failover_candidates([], llm) == ['model-A']
            assert service._zai_enabled() is False
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_excluded_without_client(self, tmp_path):
        llm = FakeLLM('model-A')
        service = await self._service(tmp_path, zai_fallback_enabled=True)
        try:
            assert service._failover_candidates([], llm) == ['model-A']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_excluded_when_broken(self, tmp_path):
        llm = FakeLLM('model-A')
        service = await self._service(tmp_path, zai_fallback_enabled=True)
        service._graphiti_client_zai = object()
        service._zai_broken_until = 10 ** 12
        try:
            assert service._failover_candidates([], llm) == ['model-A']
            assert service._zai_enabled() is False
            assert service._zai_broken() is True
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_in_attempted_never_reprobed(self, tmp_path):
        llm = FakeLLM('model-A')
        service = await self._service(tmp_path, zai_fallback_enabled=True)
        service._graphiti_client_zai = object()
        try:
            # RCA C6: inside ONE chain an already-attempted channel is never
            # re-probed...
            assert service._failover_candidates(['zai/glm-5.3'], llm) == ['model-A']
            # ...but once EVERY channel was attempted the full chain restarts
            # ("all attempted -> start over"), never a single-None deadlock.
            assert service._failover_candidates(['model-A', 'zai/glm-5.3'], llm) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_active_model_keeps_none_path(self, tmp_path):
        service = await self._service(tmp_path, zai_fallback_enabled=True)
        service._graphiti_client_zai = object()
        try:
            assert service._failover_candidates([], None) == [None]
        finally:
            await service.close()


class TestReputationByChannel:
    """Reputation is keyed by (provider, model): the zai channel is independent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_channel_tracks_independently_of_ozon(self):
        clock = FakeClock()
        service = QueueService(
            ResilienceConfig(
                zai_fallback_enabled=True,
                model_reputation_window_seconds=60.0,
                model_reputation_429_per_min_threshold=5.0,
                model_reputation_429_cooldown_seconds=0.0,
            )
        )
        service._reputation._time_fn = clock
        service._graphiti_client_zai = object()
        try:
            llm = FakeLLM('model-A')
            # The zai channel 429s; the ozon model stays clean.
            for _ in range(5):
                service._reputation.record_429('zai/glm-5.3')
            service._reputation.record_success('model-A')
            assert service._reputation.model_is_healthy('zai/glm-5.3') is False
            assert service._reputation.model_is_healthy('model-A') is True
            # A storming zai channel moves behind the healthy ozon primary.
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
            stats = service._reputation.per_model_stats()
            assert 'zai/glm-5.3' in stats
            assert 'model-A' in stats
        finally:
            await service.close()


class TestZaiFirstOnActiveTimeout:
    """2026-10-22 zai-first balance: a timeouting active ozon line flips zai first.

    The ModelReputationTracker only sees 429 storms, so a formally "healthy"
    but actually timeouting active channel keeps winning the candidate order.
    The zai-first balance promotes the zai overflow channel to the front once
    ``zai_first_timeout_threshold`` recent timeout/stream/connection events are
    recorded for the active channel - and restores the legacy order when the
    active channel is healthy or the timeouts age out of the window.
    """

    async def _service(self, tmp_path, clock, **over):
        service = QueueService(
            make_config(
                tmp_path / 'zai-first.db',
                zai_fallback_enabled=True,
                model_reputation_window_seconds=60.0,
                model_reputation_429_per_min_threshold=5.0,
                model_reputation_429_cooldown_seconds=0.0,
                **over,
            )
        )
        service._graphiti_client_zai = object()
        service._timeout_time_fn = clock
        return service

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_healthy_active_keeps_legacy_order(self, tmp_path):
        clock = FakeClock()
        service = await self._service(tmp_path, clock)
        llm = FakeLLM('model-A')
        try:
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_stream_flips_zai_first(self, tmp_path):
        clock = FakeClock()
        service = await self._service(tmp_path, clock)
        llm = FakeLLM('model-A')
        try:
            # Two recent transport timeouts on the active channel -> zai-first.
            service._record_channel_timeout('model-A')
            service._record_channel_timeout('model-A')
            assert service._failover_candidates([], llm, service._reputation) == [
                'zai/glm-5.3',
                'model-A',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_timeout_is_noise_keeps_legacy(self, tmp_path):
        clock = FakeClock()
        service = await self._service(tmp_path, clock)
        llm = FakeLLM('model-A')
        try:
            service._record_channel_timeout('model-A')
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_expired_timeouts_restore_legacy(self, tmp_path):
        clock = FakeClock()
        service = await self._service(
            tmp_path, clock, zai_first_timeout_window_seconds=60.0
        )
        llm = FakeLLM('model-A')
        try:
            service._record_channel_timeout('model-A')
            service._record_channel_timeout('model-A')
            assert service._failover_candidates([], llm, service._reputation) == [
                'zai/glm-5.3',
                'model-A',
            ]
            clock.advance(61.0)
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_first_keeps_fallbacks_after_and_respects_attempted(self, tmp_path):
        clock = FakeClock()
        service = await self._service(tmp_path, clock, model_fallbacks=['model-B'])
        llm = FakeLLM('model-A')
        try:
            service._record_channel_timeout('model-A')
            service._record_channel_timeout('model-A')
            assert service._failover_candidates([], llm, service._reputation) == [
                'zai/glm-5.3',
                'model-A',
                'model-B',
            ]
            # Anti-loop: zai already attempted -> skipped; active + fallback remain.
            assert service._failover_candidates(
                ['zai/glm-5.3'], llm, service._reputation
            ) == ['model-A', 'model-B']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_apitimeout_records_ring_and_flips_zai_first(self, tmp_path):
        """APITimeoutError flows through the real feed path into the zai-first ring.

        2026-10-23 fix: openai APITimeoutError is now classified as transient by
        is_transient_error, so _record_reputation records it and
        _is_channel_timeout_exc pushes it into _channel_timeouts. One timeout in
        the window is noise (legacy order kept); the second one flips zai first.
        """
        import httpx
        import openai

        clock = FakeClock()
        service = await self._service(tmp_path, clock)
        llm = FakeLLM('model-A')
        timeout = openai.APITimeoutError(request=httpx.Request('POST', 'http://x/v1'))
        try:
            assert service._is_channel_timeout_exc(timeout) is True
            # One timeout via the real _record_reputation path -> ring seeded,
            # still below the threshold -> legacy order preserved.
            service._record_reputation('model-A', timeout)
            assert list(service._channel_timeouts['model-A'])  # ring has the ts
            assert service._active_channel_degraded('model-A', service._reputation) is False
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
            # Second timeout inside the window -> degraded -> zai becomes first.
            service._record_reputation('model-A', timeout)
            assert service._active_channel_degraded('model-A', service._reputation) is True
            assert service._failover_candidates([], llm, service._reputation) == [
                'zai/glm-5.3',
                'model-A',
            ]
        finally:
            await service.close()


class TestE2ERouteToZaiOnStorm:
    """A 429-storming ozon primary is overflowed to the zai channel; it recovers."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_primary_storm_routes_episode_to_zai(self, tmp_path):
        clock = FakeClock()
        ozon_llm = FakeLLM('model-A')
        zai_llm = FakeLLM('zai/glm-5.3')
        ozon = FakeGraphiti(ozon_llm, fail_calls=99, fail_error=RateLimitError)  # always 429s
        zai = FakeGraphiti(zai_llm, fail_calls=0)  # healthy overflow channel
        service = QueueService(
            make_config(
                tmp_path / 'zai-e2e.db',
                zai_fallback_enabled=True,
                model_fallbacks=[],
                model_reputation_window_seconds=60.0,
                model_reputation_429_cooldown_seconds=5.0,
            ),
            storm_time_fn=clock,
            zai_time_fn=clock,
        )
        service._reputation._time_fn = clock
        try:
            await service.initialize(
                ozon, episode_builder=make_builder(), graphiti_client_zai=zai
            )
            # Fresh state: the healthy primary stays first (zai channel offered).
            assert service._failover_candidates([], ozon_llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
            await enqueue(service, 'uzai')
            await wait_done(service)
            # The episode was processed entirely on the zai channel.
            assert len(zai.calls) == 1
            assert zai.calls[0][0] == 'zai/glm-5.3'
            assert len(ozon.calls) == 1
            # attempted_models recorded the failed ozon primary only.
            assert attempted_for(service.journal, 'uzai') == ['model-A']
            # Reputation learned the primary is storming -> zai moves first.
            assert service._reputation.model_is_healthy('model-A') is False
            assert service._failover_candidates([], ozon_llm, service._reputation) == [
                'zai/glm-5.3',
                'model-A',
            ]
            # After a quiet window the primary heals and leads again.
            clock.advance(61.0)
            assert service._reputation.model_is_healthy('model-A') is True
            assert service._failover_candidates([], ozon_llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
            assert await service.journal.count_done() == 1
            assert await service.journal.claim_next('g1', 'w') is None
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_primary_api_timeout_routes_episode_to_zai(self, tmp_path):
        """An openai APITimeoutError on the primary now fails over to zai.

        2026-10-23 fix regression guard: before the fix _failover_eligible
        returned False for APITimeoutError and _process_with_failover raised on
        the first candidate, so zai was never called and the journal row
        eventually failed. After the fix the episode drains through zai and the
        timeout is recorded in the channel ring.
        """
        import httpx
        import openai

        clock = FakeClock()
        ozon_llm = FakeLLM('model-A')
        zai_llm = FakeLLM('zai/glm-5.3')
        timeout = openai.APITimeoutError(request=httpx.Request('POST', 'http://x/v1'))
        ozon = FakeGraphiti(ozon_llm, fail_calls=99, fail_error=timeout)
        zai = FakeGraphiti(zai_llm, fail_calls=0)  # healthy overflow channel
        service = QueueService(
            make_config(
                tmp_path / 'zai-timeout-e2e.db',
                zai_fallback_enabled=True,
                model_fallbacks=[],
                model_reputation_window_seconds=60.0,
                model_reputation_429_cooldown_seconds=5.0,
            ),
            storm_time_fn=clock,
            zai_time_fn=clock,
        )
        service._reputation._time_fn = clock
        try:
            await service.initialize(
                ozon, episode_builder=make_builder(), graphiti_client_zai=zai
            )
            await enqueue(service, 'utimeout')
            await wait_done(service)
            # The active APITimeoutError was classified transient and the
            # failover chain reached the zai channel, which completed the row.
            assert len(zai.calls) == 1
            assert zai.calls[0][0] == 'zai/glm-5.3'
            assert len(ozon.calls) == 1
            assert attempted_for(service.journal, 'utimeout') == ['model-A']
            # One timeout in the ring = below the threshold -> legacy order still
            # applies for the next candidate computation.
            assert service._failover_candidates([], ozon_llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()


class TestE2EAutoRollback:
    """N consecutive zai proxy losses mark the channel broken; ozon keeps working."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zai_502_strikes_auto_rollback(self, tmp_path):
        clock = FakeClock()
        ozon_llm = FakeLLM('model-A')
        zai_llm = FakeLLM('zai/glm-5.3')
        ozon = FakeGraphiti(ozon_llm, fail_calls=99, fail_error=RateLimitError)
        zai = FakeGraphiti(
            zai_llm, fail_calls=99, fail_error=httpx.ConnectError('zai proxy down')
        )
        service = QueueService(
            make_config(
                tmp_path / 'roll.db',
                zai_fallback_enabled=True,
                zai_loss_strikes=2,
                model_fallbacks=[],
                spool_backoff_base_seconds=0.01,
                retryer_interval_seconds=0.05,
                max_spool_attempts=6,
                journal_lease_seconds=30.0,
            ),
            storm_time_fn=clock,
            zai_time_fn=clock,
        )
        try:
            await service.initialize(
                ozon, episode_builder=make_builder(), graphiti_client_zai=zai
            )
            assert service._zai_enabled() is True
            await enqueue(service, 'uroll')
            # Both channels fail, so the row exhausts and retries; the second zai
            # 502 crosses zai_loss_strikes and marks the channel broken.
            await wait_cond(lambda: service._zai_broken(), msg='zai channel broken')
            assert service._zai_consecutive_losses >= 2
            assert service._zai_enabled() is False
            # The zai channel is absent from the candidate list (auto-rollback).
            assert service._failover_candidates([], ozon_llm, service._reputation) == [
                'model-A'
            ]
            # The zai client was hit (the 502s actually came from it), and the
            # fault is isolated: the zai channel is now on a cool-down.
            assert len(zai.calls) >= 2
            snap = await service.get_resilience_snapshot()
            assert snap['zai']['healthy'] is False
            assert snap['zai']['enabled'] is True
            assert snap['zai']['consecutive_losses'] >= 2
            assert snap['zai']['model'] == 'zai/glm-5.3'
            # Cooldown expiry + a fresh success re-offers the channel.
            clock.advance(360.0)
            service._zai_channel_success('zai/glm-5.3')
            assert service._zai_enabled() is True
            assert service._zai_consecutive_losses == 0
        finally:
            await service.close()


class TestHealthZaiBlock:
    """get_resilience_snapshot exposes the zai channel status for /health."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_reports_zai_off_when_not_configured(self, tmp_path):
        service = QueueService(make_config(tmp_path / 'h.db'))
        try:
            await service.initialize(
                FakeGraphiti(FakeLLM('model-A')), episode_builder=make_builder()
            )
            snap = await service.get_resilience_snapshot()
            assert snap['zai']['enabled'] is False
            assert snap['zai']['configured'] is False
            assert snap['zai']['healthy'] is False
            assert snap['zai']['model'] is None
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_reports_overflow_gate_fields(self, tmp_path):
        service = QueueService(
            make_config(tmp_path / 'h2.db', zai_overflow_pending_threshold=120)
        )
        try:
            snap = await service.get_resilience_snapshot()
            assert snap['zai']['overflow_threshold'] == 120
            assert snap['zai']['backlog_unfinished'] == 0
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_refreshes_stale_backlog_cache(self, tmp_path):
        # After a drain with no new episodes, _zai_overflow_allowed is no longer
        # called, so the cache would keep the last sampled value. /health must
        # report the LIVE backlog, not the stale snapshot.
        service = QueueService(
            make_config(tmp_path / 'h3.db', zai_overflow_pending_threshold=120)
        )
        try:
            service._zai_backlog_unfinished = 7  # stale leftover from the drain
            snap = await service.get_resilience_snapshot()
            assert snap['zai']['backlog_unfinished'] == 0  # live count, cache refreshed
        finally:
            await service.close()


class TestGatedOverflow:
    """Backlog gate keeps the zai channel a strict overflow (drain) channel.

    ``zai_overflow_pending_threshold`` (0 by default = legacy) : the zai channel
    joins failover candidates ONLY while the journal is backed up
    (unfinished >= threshold). When the gate is closed, zai is excluded from the
    candidate list BEFORE the per-model health reorder, so a reputation-driven
    reorder can never promote it - even when the active ozon model is unhealthy.
    """

    async def _service(self, tmp_path, **over) -> QueueService:
        return QueueService(
            make_config(
                tmp_path / 'gate.db',
                zai_fallback_enabled=True,
                **over,
            )
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_gate_off_by_default_keeps_legacy_behaviour(self, tmp_path):
        # Threshold default is 0: the gate is open and zai is offered exactly as
        # before this feature (18 pre-existing cases rely on that default).
        service = await self._service(tmp_path)
        service._graphiti_client_zai = object()
        try:
            assert service.resilience.zai_overflow_pending_threshold == 0
            assert await service._zai_overflow_allowed() is True
            llm = FakeLLM('model-A')
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-A',
                'zai/glm-5.3',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_threshold_above_backlog_excludes_zai_even_when_active_unhealthy(
        self, tmp_path
    ):
        # (i) Gate closed (threshold above the live backlog): zai is NOT a
        # candidate even though reputation would have promoted the healthy zai
        # channel ahead of the 429-storming active ozon model.
        clock = FakeClock()
        service = await self._service(
            tmp_path,
            zai_overflow_pending_threshold=10,
            model_reputation_window_seconds=60.0,
            model_reputation_429_per_min_threshold=5.0,
            model_reputation_429_cooldown_seconds=5.0,
        )
        service._reputation._time_fn = clock
        service._graphiti_client_zai = object()
        try:
            llm = FakeLLM('model-A')
            # The active ozon model is reputation-unhealthy; the zai channel is clean.
            for _ in range(5):
                service._reputation.record_429('model-A')
            service._reputation.record_success('zai/glm-5.3')
            assert service._reputation.model_is_healthy('model-A') is False
            assert service._reputation.model_is_healthy('zai/glm-5.3') is True
            # Without the gate (zai_allowed=None) the healthy zai is promoted first.
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=None
            ) == ['zai/glm-5.3', 'model-A']
            # Gate closed: live backlog (0) < threshold (10).
            assert await service._zai_overflow_allowed() is False
            # Excluded BEFORE reorder -> the reorder physically cannot promote zai.
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=False
            ) == ['model-A']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_threshold_met_offers_zai_after_active(self, tmp_path):
        # (ii) Gate open (backlog >= threshold): zai is offered after the active
        # model, as the drain channel it is meant to be.
        service = await self._service(tmp_path, zai_overflow_pending_threshold=5)
        service._graphiti_client_zai = object()
        try:
            # Seed the journal so unfinished >= threshold.
            for i in range(5):
                await service._journal.enqueue(
                    {
                        'group_id': 'g1',
                        'name': f'ep-{i}',
                        'episode_body': f'body-{i}',
                        'source_description': 'd',
                        'source': 'text',
                        'uuid': f'gate-{i}',
                    }
                )
            assert await service._zai_overflow_allowed() is True
            assert service._zai_backlog_unfinished == 5
            llm = FakeLLM('model-A')
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=True
            ) == ['model-A', 'zai/glm-5.3']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_e2e_backlog_growth_opens_gate_drain_closes_it(self, tmp_path):
        # (iii) Full gate path on a live journal: backlog grows past the threshold
        # -> zai appears in the chain; backlog drains below it -> zai disappears.
        # No initialize (episode_builder=None) so the pool idles and the journal
        # count is fully under test control.
        service = await self._service(
            tmp_path, zai_overflow_pending_threshold=3, model_fallbacks=[]
        )
        service._graphiti_client_zai = object()
        llm = FakeLLM('model-A')
        try:
            # Empty journal: gate closed, zai absent.
            assert await service._zai_overflow_allowed() is False
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=False
            ) == ['model-A']
            # Backlog grows to the threshold -> gate opens -> zai enters the chain.
            row_ids = []
            for i in range(3):
                row_id, inserted = await service._journal.enqueue(
                    {
                        'group_id': 'g1',
                        'name': f'ep-{i}',
                        'episode_body': f'body-{i}',
                        'source_description': 'd',
                        'source': 'text',
                        'uuid': f'e2e-{i}',
                    }
                )
                assert inserted
                row_ids.append(row_id)
            assert await service._zai_overflow_allowed() is True
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=True
            ) == ['model-A', 'zai/glm-5.3']
            # Backlog drains below the threshold -> gate closes -> zai leaves the chain.
            for row_id in row_ids:
                assert await service._journal.mark_done(row_id) is True
            assert await service._zai_overflow_allowed() is False
            assert service._failover_candidates(
                [], llm, service._reputation, zai_allowed=False
            ) == ['model-A']
        finally:
            await service.close()
