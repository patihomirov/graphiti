"""Unit tests for the always-active per-channel rate limiter (2026-10-21).

Covers the strict sliding-window RPM guarantee (the number of calls in ANY
window-long interval never exceeds the configured ceiling), the async slot
wait, per-channel isolation, the OFF (0) legacy path, the failover-friendly
``estimated_wait_seconds`` and the "a 429 never loosens the limit" property
(the limiter has no branch that speeds the stream back up after a failure).

Uses a fake monotonic clock + fake sleep that advances the clock, making every
test deterministic and instant (no real time.sleep).
"""

import pytest

from services.rate_limiter import ChannelRateLimiter


class FakeClock:
    """Deterministic fake monotonic clock."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_limiter(clock, max_calls=0, window=60.0):
    """Limiter with a fake sleep that advances the clock by the requested delay."""

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)

    return ChannelRateLimiter(
        max_calls_per_minute=max_calls,
        window_seconds=window,
        time_fn=clock,
        sleep_fn=fake_sleep,
    )


class TestOff:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_limit_is_disabled_and_noop(self):
        limiter = ChannelRateLimiter(max_calls_per_minute=0)
        assert limiter.enabled is False
        assert await limiter.acquire('model-A') == 0.0
        assert await limiter.estimated_wait_seconds('model-A') == 0.0
        assert limiter.current_calls('model-A') == 0
        assert limiter.snapshot()['limit_per_minute'] == 0


class TestRpmCeiling:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_never_exceeds_rpm_per_channel(self):
        """100 back-to-back acquires never leave >max_calls calls inside the window."""
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=3, window=10.0)
        for _ in range(100):
            await limiter.acquire('ch')
            assert limiter.current_calls('ch') <= 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_until_oldest_slot_drops_out(self):
        """The 4th call (over a 3/10s ceiling) waits almost a full window."""
        clock = FakeClock()
        start = clock.t
        limiter = make_limiter(clock, max_calls=3, window=10.0)
        for _ in range(3):
            assert await limiter.acquire('ch') == 0.0
        waited = await limiter.acquire('ch')
        assert 9.999 <= waited <= 10.0
        assert clock.t - start >= 9.999

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_uniform_slotting_no_burst_after_wait(self):
        """After paying a wait, the next immediately-issued call may still need a slot."""
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=1, window=5.0)
        assert await limiter.acquire('ch') == 0.0
        # Slot is now occupied; the next caller must wait for it to drop out.
        waited = await limiter.acquire('ch')
        assert 4.999 <= waited <= 5.0


class TestChannelIsolation:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_channels_are_independent(self):
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=1, window=10.0)
        await limiter.acquire('model-A')
        # model-A is saturated, model-B has never been touched.
        assert await limiter.estimated_wait_seconds('model-A') > 0.0
        assert await limiter.estimated_wait_seconds('model-B') == 0.0
        assert await limiter.acquire('model-B') == 0.0


class TestEstimatedWait:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_estimated_wait_reflects_saturation(self):
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=2, window=10.0)
        assert await limiter.estimated_wait_seconds('ch') == 0.0
        await limiter.acquire('ch')
        assert await limiter.estimated_wait_seconds('ch') == 0.0  # slot still free
        await limiter.acquire('ch')
        # Window full: the next acquire would wait for the oldest to leave.
        est = await limiter.estimated_wait_seconds('ch')
        assert 9.999 <= est <= 10.0


class Test429DoesNotLoosen:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_429_is_just_another_call_that_holds_a_slot(self):
        """A failed (429) call still occupies its window slot - no loosening."""
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=1, window=10.0)
        # First call "returned 429" - it occupies the single slot.
        assert await limiter.acquire('ch') == 0.0
        # The follow-up attempt must still respect the ceiling: no speed-up.
        waited = await limiter.acquire('ch')
        assert waited >= 9.999
        # And the RPM invariant held throughout.
        assert limiter.current_calls('ch') <= 1


class TestSnapshot:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_exposes_pacing_state(self):
        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=4, window=60.0)
        await limiter.acquire('ch')
        snap = limiter.snapshot()
        assert snap['limit_per_minute'] == 4
        assert snap['window_seconds'] == 60.0
        assert snap['channels']['ch']['current_calls_in_window'] == 1
        assert snap['channels']['ch']['times_waited'] == 0


class TestClientIntegration:
    """The OpenAIGenericClient routes each call through the limiter (duck-typed)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_client_goes_through_limiter(self):
        from types import SimpleNamespace

        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.prompts.models import Message

        clock = FakeClock()
        limiter = make_limiter(clock, max_calls=2, window=10.0)

        calls = {'n': 0}

        class _FakeCompletions:
            async def create(self, **kwargs):
                calls['n'] += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))
                    ]
                )

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeOpenAIClient:
            chat = _FakeChat()

        cfg = LLMConfig(api_key='k', model='m', base_url='http://x/v1')
        client = OpenAIGenericClient(config=cfg, client=_FakeOpenAIClient())
        client.set_channel_rate_limiter(limiter)
        msg = [Message(role='user', content='hi')]

        # Two back-to-back single calls fit within the 2/10s ceiling instantly.
        assert await client.generate_response(list(msg)) == {'ok': 1}
        assert await client.generate_response(list(msg)) == {'ok': 1}
        assert calls['n'] == 2
        start = clock.t
        # The third call must be paced (waits for the oldest slot to leave).
        await client.generate_response(list(msg))
        assert calls['n'] == 3
        assert clock.t - start >= 9.999
        # The channel key was provider-qualified (base_url|model).
        assert 'http://x/v1|m' in limiter.snapshot()['channels']


if __name__ == '__main__':  # pragma: no cover
    import asyncio

    async def _selftest():
        clock = FakeClock()
        lim = make_limiter(clock, 3, 10.0)
        for _ in range(5):
            w = await lim.acquire('ch')
            print(f'waited={w:.2f} clock={clock.t} calls={lim.current_calls("ch")}')

    asyncio.run(_selftest())
