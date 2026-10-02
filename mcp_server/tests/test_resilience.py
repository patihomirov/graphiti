"""Unit tests for the resilience layer: circuit breaker, backpressure, spool.

Covers both phases of the write-path resilience work without touching graphiti
core / the network. AsyncMock(spec=Graphiti) is used so the worker never touches
a real database.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from graphiti_core import Graphiti
from graphiti_core.llm_client.errors import RateLimitError

from config.schema import ResilienceConfig
from services.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    QueueCapacityExceeded,
    is_transient_error,
)
from services.episode_spool import EpisodeRetryer, EpisodeSpool
from services.queue_service import QueueService


class FakeClock:
    """Monotonic clock we can advance manually to drive the breaker's cooldown."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


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


class TestCircuitBreaker:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_opens_after_consecutive_failures(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=3, open_timeout_seconds=30.0, time_fn=clock)

        assert await breaker.allow_request() is True
        for _ in range(2):
            await breaker.record_failure(RateLimitError())
            assert await breaker.allow_request() is True

        await breaker.record_failure(RateLimitError())
        assert await breaker.allow_request() is False
        snapshot = await breaker.get_snapshot()
        assert snapshot['state'] == 'open'
        assert snapshot['failure_count'] == 3
        assert snapshot['retry_after_seconds'] == pytest.approx(30.0)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_closes_on_probe_success(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock)
        await breaker.record_failure(RateLimitError())
        assert await breaker.allow_request() is False

        clock.advance(30.0)
        assert await breaker.allow_request() is True  # transparent transition to half_open
        assert (await breaker.get_snapshot())['state'] == 'half_open'

        await breaker.record_success()
        assert (await breaker.get_snapshot())['state'] == 'closed'
        assert (await breaker.get_snapshot())['failure_count'] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_reopens_on_probe_failure(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock)
        await breaker.record_failure(RateLimitError())
        clock.advance(30.0)
        assert await breaker.allow_request() is True

        await breaker.record_failure(RateLimitError())  # probe fails
        assert (await breaker.get_snapshot())['state'] == 'open'
        assert await breaker.allow_request() is False


class TestIsTransientError:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        'exc,expected',
        [
            (RateLimitError(), True),
            (httpx.ConnectError('c', request=httpx.Request('GET', 'http://x')), True),
            (httpx.ReadTimeout('t', request=httpx.Request('GET', 'http://x')), True),
            (
                httpx.HTTPStatusError(
                    'e',
                    request=httpx.Request('GET', 'http://x'),
                    response=httpx.Response(503, request=httpx.Request('GET', 'http://x')),
                ),
                True,
            ),
            (
                httpx.HTTPStatusError(
                    'e',
                    request=httpx.Request('GET', 'http://x'),
                    response=httpx.Response(429, request=httpx.Request('GET', 'http://x')),
                ),
                True,
            ),
            (TimeoutError('t'), True),
            (ConnectionError('c'), True),
            # Permanent errors are not transient.
            (ValueError('validation'), False),
            (
                httpx.HTTPStatusError(
                    'e',
                    request=httpx.Request('GET', 'http://x'),
                    response=httpx.Response(400, request=httpx.Request('GET', 'http://x')),
                ),
                False,
            ),
        ],
    )
    def test_transient_table(self, exc, expected):
        assert is_transient_error(exc) is expected


class TestBackpressure:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_episode_fails_fast_when_circuit_open(self):
        client = AsyncMock(spec=Graphiti)
        service = QueueService(ResilienceConfig(spool_enabled=False))
        await service.initialize(client)

        for _ in range(3):
            await service._breaker.record_failure(RateLimitError())

        with pytest.raises(CircuitOpenError):
            await service.add_episode(
                group_id='g1',
                name='e',
                content='b',
                source_description='d',
                episode_type='text',
                entity_types=None,
                uuid='u-open',
            )

        assert service.get_queue_size('g1') == 0
        client.add_episode.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_episode_fails_fast_on_queue_capacity(self):
        service = QueueService(ResilienceConfig(max_queue_depth=2, spool_enabled=False))
        gate = asyncio.Event()

        async def block():
            await gate.wait()

        await service.add_episode_task('g1', block)
        await service.add_episode_task('g1', block)
        assert service._queue_depth == 2

        with pytest.raises(QueueCapacityExceeded):
            await service.add_episode_task('g1', block)

        gate.set()
        await service._episode_queues['g1'].join()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_exposes_breaker_and_depth(self):
        service = QueueService(ResilienceConfig(spool_enabled=False))
        snap = await service.get_resilience_snapshot()
        assert snap['state'] in ('closed', 'disabled')
        assert snap['max_queue_depth'] == 20
        assert snap['queue_depth'] == 0

    @pytest.mark.unit
    def test_public_accessors_expose_resilience_internals(self, tmp_path):
        service = QueueService(ResilienceConfig(spool_dir=str(tmp_path / 'spool')))
        assert isinstance(service.circuit_breaker, CircuitBreaker)
        assert isinstance(service.spool, EpisodeSpool)

        service = QueueService(ResilienceConfig(spool_enabled=False))
        assert isinstance(service.circuit_breaker, CircuitBreaker)
        assert service.spool is None

        service = QueueService(ResilienceConfig(enabled=False))
        assert service.circuit_breaker is None
        assert service.spool is None


class TestSpool:
    @pytest.mark.unit
    def test_round_trip_save_load(self, tmp_path):
        spool = EpisodeSpool(tmp_path)
        plan = make_plan(uuid='u-roundtrip')
        path = spool.save(plan, 'boom')

        assert path.exists()
        loaded = spool.load(path)
        assert loaded['uuid'] == 'u-roundtrip'
        assert loaded['name'] == 'ep'
        assert loaded['source'] == 'text'
        assert loaded['first_failure_ts'] is not None
        assert loaded['last_reason'] == 'boom'

        # Stand-in for the server's episode_builder reconstructing parameters.
        rebuilt = {
            'name': loaded['name'],
            'episode_body': loaded['episode_body'],
            'source': loaded['source'],
            'uuid': loaded['uuid'],
        }
        assert rebuilt['source'] == 'text'

        spool.delete(path)
        assert not path.exists()

    @pytest.mark.unit
    def test_mark_failed_writes_receipt(self, tmp_path):
        spool = EpisodeSpool(tmp_path)
        plan = make_plan(uuid='u-fail')
        path = spool.save(plan, 'first err')

        spool.update_attempt(spool.load(path), 'again')
        spool.mark_failed(path, spool.load(path), 'gave up')

        assert not path.exists()
        assert (tmp_path / 'failed' / path.name).exists()

        receipts = (tmp_path / 'failed' / 'failed.jsonl').read_text().strip().splitlines()
        receipt = json.loads(receipts[0])
        assert receipt['uuid'] == 'u-fail'
        assert receipt['group_id'] == 'g1'
        assert receipt['ready_for_manual_readd'] is True
        assert receipt['reason'] == 'gave up'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retryer_success_removes_spool_file(self, tmp_path):
        spool = EpisodeSpool(tmp_path, backoff_base_seconds=1)
        client = AsyncMock(spec=Graphiti)
        plan = make_plan(uuid='u-retry-ok')
        path = spool.save(plan, 'first err')
        breaker = CircuitBreaker(failure_threshold=3, open_timeout_seconds=30)

        def builder(p):
            return {
                'name': p['name'],
                'episode_body': p['episode_body'],
                'source': 'text',
                'uuid': p['uuid'],
                'group_id': p['group_id'],
            }

        retryer = EpisodeRetryer(
            spool, breaker, builder, client, interval_seconds=1, max_attempts=3, backoff_base_seconds=1
        )
        await retryer._process_one(path, spool.load(path))

        assert not path.exists()
        client.add_episode.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retryer_moves_to_failed_after_max_attempts(self, tmp_path):
        spool = EpisodeSpool(tmp_path, backoff_base_seconds=1)
        client = AsyncMock(spec=Graphiti)
        client.add_episode.side_effect = RateLimitError()
        plan = make_plan(uuid='u-retry-fail')
        path = spool.save(plan, 'boom')
        breaker = CircuitBreaker(failure_threshold=3, open_timeout_seconds=30)

        def builder(p):
            return {
                'name': p['name'],
                'episode_body': p['episode_body'],
                'source': 'text',
                'uuid': p['uuid'],
                'group_id': p['group_id'],
            }

        retryer = EpisodeRetryer(
            spool, breaker, builder, client, interval_seconds=1, max_attempts=2, backoff_base_seconds=1
        )

        await retryer._process_one(path, spool.load(path))
        assert path.exists()  # first failure -> still pending

        await retryer._process_one(path, spool.load(path))
        assert not path.exists()  # second failure -> exhausted, moved to failed/
        assert (tmp_path / 'failed' / path.name).exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retryer_skips_when_breaker_open(self, tmp_path):
        spool = EpisodeSpool(tmp_path, backoff_base_seconds=1)
        client = AsyncMock(spec=Graphiti)
        plan = make_plan(uuid='u-open')
        path = spool.save(plan, 'boom')
        breaker = CircuitBreaker(failure_threshold=1, open_timeout_seconds=30)
        await breaker.record_failure(RateLimitError())
        assert await breaker.allow_request() is False

        retryer = EpisodeRetryer(
            spool, breaker, lambda p: {'name': p['name']}, client, interval_seconds=1, max_attempts=3
        )
        await retryer._tick()

        client.add_episode.assert_not_awaited()
        assert path.exists()
