"""Regression tests: the spool retryer must never strand the half-open probe.

Diagnosed twice on 2026-10-05: with the breaker open and the spool empty, the
retryer's ``_tick()`` probed the breaker FIRST (consuming the single half-open
probe slot), then found nothing to replay, so ``record_success``/``record_failure``
never arrived and the breaker stuck in half_open rejecting every ``add_memory``
until a restart. The same stranding happened when the replayed episode itself
succeeded (the success path never resolved the probe).

These tests pin the fix from both sides: the retryer no longer probes without
due work and always resolves a probe it consumed, and the breaker re-grants a
probe that stayed unresolved for ``probe_timeout_seconds``.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from graphiti_core import Graphiti
from graphiti_core.llm_client.errors import RateLimitError

from services.circuit_breaker import CircuitBreaker
from services.episode_spool import EpisodeRetryer, EpisodeSpool


class FakeClock:
    """Monotonic clock we can advance manually to drive breaker timers."""

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
    }
    plan.update(over)
    return plan


def make_retryer(tmp_path, breaker, client, **over):
    backoff = over.pop('backoff_base_seconds', 3600.0)
    spool = EpisodeSpool(tmp_path, backoff_base_seconds=backoff)

    def builder(p):
        return {
            'name': p['name'],
            'episode_body': p['episode_body'],
            'source': 'text',
            'uuid': p['uuid'],
            'group_id': p['group_id'],
        }

    return EpisodeRetryer(
        spool,
        breaker,
        builder,
        client,
        interval_seconds=1,
        max_attempts=over.pop('max_attempts', 3),
        backoff_base_seconds=backoff,
    )


class TestRetryerDoesNotStrandProbe:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_spool_tick_does_not_consume_probe(self, tmp_path):
        """The diagnosed case: open breaker + empty spool -> tick must not probe.

        A probe consumed with nothing to replay is never resolved by
        record_success/record_failure, so the breaker would stick in half_open.
        """
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock
        )
        await breaker.record_failure(RateLimitError())  # open @ t=1000
        clock.advance(30.0)  # cooldown elapsed

        retryer = make_retryer(tmp_path, breaker, AsyncMock(spec=Graphiti))
        await retryer._tick()  # spool is empty

        # The breaker was never probed: still open, probe slot untouched.
        assert (await breaker.get_snapshot())['state'] == 'open'

        # A real request can still take the probe and resolve it -> closed.
        assert await breaker.allow_request() is True
        await breaker.record_success()
        assert (await breaker.get_snapshot())['state'] == 'closed'
        assert await breaker.allow_request() is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backed_off_spool_tick_does_not_consume_probe(self, tmp_path):
        """Pending-but-not-due episodes must not burn the probe slot either."""
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock
        )
        await breaker.record_failure(RateLimitError())
        clock.advance(30.0)

        retryer = make_retryer(tmp_path, breaker, AsyncMock(spec=Graphiti))
        path = retryer.spool.save(make_plan(uuid='u-backoff'), 'boom')
        retryer.spool.update_attempt(retryer.spool.load(path), 'again')  # backoff starts

        await retryer._tick()  # episode exists but is not due yet

        assert (await breaker.get_snapshot())['state'] == 'open'  # never probed
        assert await breaker.allow_request() is True  # probe slot still available

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retryer_success_closes_half_open_breaker(self, tmp_path):
        """Replay consumes the probe; its success must close the breaker."""
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock
        )
        await breaker.record_failure(RateLimitError())  # open @ t=1000
        clock.advance(30.0)

        client = AsyncMock(spec=Graphiti)
        retryer = make_retryer(tmp_path, breaker, client)
        path = retryer.spool.save(make_plan(uuid='u-retry-probe'), 'boom')

        await retryer._tick()  # probes open->half_open, replays, succeeds

        client.add_episode.assert_awaited_once()
        assert not path.exists()
        assert (await breaker.get_snapshot())['state'] == 'closed'
        assert await breaker.allow_request() is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retryer_permanent_failure_releases_probe(self, tmp_path):
        """A permanent replay error is not provider trouble: resolve, don't strand."""
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, open_timeout_seconds=30.0, time_fn=clock
        )
        await breaker.record_failure(RateLimitError())
        clock.advance(30.0)

        client = AsyncMock(spec=Graphiti)
        client.add_episode.side_effect = ValueError('bad plan')
        retryer = make_retryer(tmp_path, breaker, client)
        path = retryer.spool.save(make_plan(uuid='u-perm'), 'boom')

        await retryer._tick()

        client.add_episode.assert_awaited_once()
        assert path.exists()  # attempt recorded, episode stays pending
        # attempt=2, not 1: pre-existing double increment (_process_one bumps,
        # then update_attempt bumps again) — unchanged by this fix.
        assert retryer.spool.load(path)['attempt'] == 2
        assert (await breaker.get_snapshot())['state'] == 'closed'  # probe resolved
        assert await breaker.allow_request() is True


class TestProbeTimeout:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stranded_probe_regranted_after_timeout(self):
        """Belt-and-suspenders: a probe unresolved for probe_timeout_seconds is re-granted."""
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1,
            open_timeout_seconds=30.0,
            probe_timeout_seconds=60.0,
            time_fn=clock,
        )
        await breaker.record_failure(RateLimitError())  # open @ t=1000
        clock.advance(30.0)
        assert await breaker.allow_request() is True  # probe granted @ t=1030

        # While the probe is fresh, parallel submissions are still rejected.
        results = await asyncio.gather(*[breaker.allow_request() for _ in range(3)])
        assert all(r is False for r in results)

        clock.advance(59.0)
        assert await breaker.allow_request() is False  # still held

        clock.advance(1.0)  # probe age hits the 60s timeout
        assert await breaker.allow_request() is True  # re-granted

        await breaker.record_success()  # the new probe resolves
        assert (await breaker.get_snapshot())['state'] == 'closed'
        assert await breaker.allow_request() is True
