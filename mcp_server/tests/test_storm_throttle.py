"""Unit tests for the per-model StormThrottle (Feature A).

Covers storm entry hysteresis (a real burst >= threshold in the window, or the
reputation-unhealthy OR-entry), per-model isolation (model A storming does not
touch model B), the threshold never false-triggering on 1-2 isolated 429s,
quiet-success exit, the lazy silence self-heal and the Retry-After hard pause.
"""

import pytest

from services.storm_throttle import STORM_EXIT_SILENCE_SECONDS, StormThrottle


class StormClock:
    """Deterministic fake monotonic clock."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_throttle(clock, **over) -> StormThrottle:
    defaults = dict(window_seconds=60.0, threshold=6, time_fn=clock)
    defaults.update(over)
    return StormThrottle(**defaults)


class TestStormEntry:
    @pytest.mark.unit
    def test_does_not_storm_on_1_or_2_429s(self):
        """1-2 isolated 429s (even in one window) are NOT a storm."""
        clock = StormClock()
        t = make_throttle(clock)
        t.record_failure('model-A')
        assert t.in_storm('model-A') is False
        # A second lone 429 an hour later is still not a storm.
        clock.advance(3600)
        t.record_failure('model-A')
        assert t.in_storm('model-A') is False
        assert t.storm_models() == []

    @pytest.mark.unit
    def test_enters_storm_after_threshold_of_429s_in_window(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=6)
        for _ in range(5):
            t.record_failure('model-A')
        assert t.in_storm('model-A') is False  # below threshold
        t.record_failure('model-A')  # 6th -> storm
        assert t.in_storm('model-A') is True
        assert t.storm_models() == ['model-A']

    @pytest.mark.unit
    def test_per_model_isolation(self):
        """Model B stays normal while model A is storming."""
        clock = StormClock()
        t = make_throttle(clock, threshold=3)
        for _ in range(3):
            t.record_failure('model-A')
        assert t.in_storm('model-A') is True
        assert t.in_storm('model-B') is False
        assert t.storm_models() == ['model-A']
        # A success on B has no effect on A's storm either.
        t.record_success('model-B')
        assert t.in_storm('model-A') is True

    @pytest.mark.unit
    def test_enters_storm_when_reputation_unhealthy_signal(self):
        """The OR-entry: a reputation-unhealthy flag triggers storm even at <threshold."""
        clock = StormClock()
        t = make_throttle(clock, threshold=6)
        # Only 2 direct 429s, but reputation reports the model unhealthy.
        t.record_failure('model-A')
        t.record_failure('model-A', reputation_unhealthy=True)
        assert t.in_storm('model-A') is True

    @pytest.mark.unit
    def test_window_prunes_old_429s(self):
        """429s older than the window do not count towards the next storm."""
        clock = StormClock()
        t = make_throttle(clock, window_seconds=60.0, threshold=6)
        for _ in range(5):
            t.record_failure('model-A')
        # Let the window pass: old events fall out, no storm.
        clock.advance(61.0)
        t.record_failure('model-A')
        assert t.in_storm('model-A') is False


class TestStormExit:
    @pytest.mark.unit
    def test_quiet_success_exits_storm(self):
        """After 120s without a 429, one success clears the storm."""
        clock = StormClock()
        t = make_throttle(clock, threshold=3)
        for _ in range(3):
            t.record_failure('model-A')
        assert t.in_storm('model-A') is True
        # Success while still noisy: does NOT exit.
        t.record_success('model-A')
        assert t.in_storm('model-A') is True
        # Wait out the silence window, then a quiet success exits.
        clock.advance(STORM_EXIT_SILENCE_SECONDS + 1)
        t.record_success('model-A')
        assert t.in_storm('model-A') is False

    @pytest.mark.unit
    def test_lazy_silence_self_heals(self):
        """A storm that simply goes quiet releases on its own (no call needed)."""
        clock = StormClock()
        t = make_throttle(clock, threshold=3)
        for _ in range(3):
            t.record_failure('model-A')
        assert t.in_storm('model-A') is True
        clock.advance(STORM_EXIT_SILENCE_SECONDS + 5)
        # Just asking in_storm is enough; the next probe is un-throttled.
        assert t.in_storm('model-A') is False
        assert t.is_throttled('model-A') is None
        assert t.storm_models() == []

    @pytest.mark.unit
    def test_storm_reeenters_after_new_burst(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=3)
        for _ in range(3):
            t.record_failure('model-A')
        clock.advance(STORM_EXIT_SILENCE_SECONDS + 1)
        t.record_success('model-A')
        assert t.in_storm('model-A') is False
        # A fresh burst storms it again.
        for _ in range(3):
            t.record_failure('model-A')
        assert t.in_storm('model-A') is True


class TestRetryAfterThrottling:
    @pytest.mark.unit
    def test_throttled_returns_remaining_retry_after(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=3, respect_retry_after=True)
        for _ in range(3):
            t.record_failure('model-A', retry_after=10.0)
        assert t.in_storm('model-A') is True
        throttled = t.is_throttled('model-A')
        assert throttled is not None and abs(throttled - 10.0) < 0.01

    @pytest.mark.unit
    def test_throttled_decays_and_expires(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=3, respect_retry_after=True)
        for _ in range(3):
            t.record_failure('model-A', retry_after=10.0)
        assert t.is_throttled('model-A') is not None
        clock.advance(11.0)
        assert t.is_throttled('model-A') is None  # pause expired

    @pytest.mark.unit
    def test_no_retry_after_hint_means_not_hard_throttled(self):
        """Storm without a Retry-After hint -> soft pacing only (None)."""
        clock = StormClock()
        t = make_throttle(clock, threshold=3)
        for _ in range(3):
            t.record_failure('model-A')  # no retry_after
        assert t.in_storm('model-A') is True
        assert t.is_throttled('model-A') is None

    @pytest.mark.unit
    def test_respect_retry_after_off_never_throttles(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=3, respect_retry_after=False)
        for _ in range(3):
            t.record_failure('model-A', retry_after=10.0)
        assert t.in_storm('model-A') is True
        assert t.is_throttled('model-A') is None

    @pytest.mark.unit
    def test_not_storming_never_throttles(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=10)
        t.record_failure('model-A', retry_after=30.0)
        assert t.in_storm('model-A') is False
        assert t.is_throttled('model-A') is None


class TestSnapshot:
    @pytest.mark.unit
    def test_snapshot_exposes_per_model_state(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=2)
        t.record_success('model-B')
        t.record_failure('model-A', retry_after=7.0)
        t.record_failure('model-A')
        snap = t.snapshot()
        assert 'per_model' in snap
        entry = snap['per_model']['model-A']
        assert entry['in_storm'] is True
        assert entry['window_429_count'] == 2
        assert entry['throttled_retry_after'] is not None
        assert entry['last_429_ts'] is not None
        assert snap['per_model']['model-B']['in_storm'] is False

    @pytest.mark.unit
    def test_snapshot_uses_monotonic_time_by_default(self):
        t = StormThrottle()  # default time_fn
        t.record_failure('model-X')
        snap = t.snapshot()
        assert snap['per_model']['model-X']['window_429_count'] == 1

    @pytest.mark.unit
    def test_any_storm(self):
        clock = StormClock()
        t = make_throttle(clock, threshold=2)
        assert t.any_storm() is False
        t.record_failure('model-A')
        t.record_failure('model-A')
        assert t.any_storm() is True


class TestThreadSafety:
    @pytest.mark.unit
    def test_concurrent_recording_does_not_corrupt(self):
        """A quick thread burst of records must not raise or corrupt state."""
        import threading as _threading

        clock = StormClock()
        t = make_throttle(clock, threshold=100)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    t.record_failure('model-A')
                    t.record_success('model-B')
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [_threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert t.snapshot()['per_model']['model-A']['window_429_count'] >= 400
