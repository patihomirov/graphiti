"""Unit tests for the per-model reputation tracker (Phase 4).

Covers the sliding-window 429 / empty / transient / success counters, the
health predicate (429-rate threshold + fresh-429 cooldown), the candidate
reordering used by the journal worker's failover chain, and the /health
per-model stats export. All tests use a fake monotonic clock - no I/O.
"""

import pytest

from services.model_reputation import ModelReputationTracker


class FakeClock:
    """Monotonic clock we can advance manually to drive the sliding window."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_tracker(clock, **over) -> ModelReputationTracker:
    base = dict(window_seconds=60.0, per_min_threshold=5.0, cooldown_seconds=60.0)
    base.update(over)
    return ModelReputationTracker(time_fn=clock, **base)


class TestHealth:
    @pytest.mark.unit
    def test_empty_tracker_is_healthy(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        assert tracker.model_is_healthy('model-A') is True

    @pytest.mark.unit
    def test_rate_below_threshold_is_healthy(self):
        clock = FakeClock()
        # 4 429s over a 60s window = 4/min < 5/min -> healthy by rate.
        tracker = make_tracker(clock, cooldown_seconds=0.0)
        for _ in range(4):
            tracker.record_429('model-A')
        assert tracker.model_is_healthy('model-A') is True

    @pytest.mark.unit
    def test_rate_at_threshold_is_unhealthy(self):
        clock = FakeClock()
        # 5 429s over a 60s window = 5/min == threshold -> unhealthy.
        tracker = make_tracker(clock, cooldown_seconds=0.0)
        for _ in range(5):
            tracker.record_429('model-A')
        assert tracker.model_is_healthy('model-A') is False

    @pytest.mark.unit
    def test_fresh_429_cooldown_overrides_low_rate(self):
        clock = FakeClock()
        # 1 429 -> rate 1/min (healthy by rate), but a fresh 429 within the
        # 60s cooldown marks the model unhealthy regardless.
        tracker = make_tracker(clock)
        tracker.record_429('model-A')
        assert tracker.model_is_healthy('model-A') is False
        # After the cooldown elapses (no new 429) the model recovers.
        clock.advance(61.0)
        assert tracker.model_is_healthy('model-A') is True

    @pytest.mark.unit
    def test_empty_and_transient_do_not_affect_health(self):
        clock = FakeClock()
        tracker = make_tracker(clock, cooldown_seconds=0.0)
        tracker.record_empty('model-A')
        tracker.record_transient('model-A')
        tracker.record_success('model-A')
        assert tracker.model_is_healthy('model-A') is True

    @pytest.mark.unit
    def test_storm_over_then_silence_recovers(self):
        clock = FakeClock()
        tracker = make_tracker(clock, window_seconds=10.0, per_min_threshold=30.0)
        for _ in range(10):  # 10/10s = 60/min > 30/min -> unhealthy
            tracker.record_429('model-A')
        assert tracker.model_is_healthy('model-A') is False
        # Events fall out of the sliding window -> no 429s -> healthy again.
        clock.advance(11.0)
        assert tracker.model_is_healthy('model-A') is True


class TestReorderByHealth:
    @pytest.mark.unit
    def test_unhealthy_primary_moves_fallback_first(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        for _ in range(5):
            tracker.record_429('model-A')  # 5/min >= 5/min -> unhealthy
        assert tracker.reorder_by_health(['model-A', 'model-B']) == ['model-B', 'model-A']
        assert tracker.reorder_by_health(['model-A', 'model-B', 'model-C']) == [
            'model-B',
            'model-C',
            'model-A',
        ]

    @pytest.mark.unit
    def test_healthy_primary_keeps_priority(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        tracker.record_success('model-A')
        assert tracker.reorder_by_health(['model-A', 'model-B']) == ['model-A', 'model-B']

    @pytest.mark.unit
    def test_healthy_fallback_order_preserved(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        for _ in range(5):
            tracker.record_429('model-A')
        tracker.record_success('model-B')
        tracker.record_success('model-C')
        # Both fallbacks healthy -> keep their relative configured order.
        assert tracker.reorder_by_health(['model-A', 'model-B', 'model-C']) == [
            'model-B',
            'model-C',
            'model-A',
        ]

    @pytest.mark.unit
    def test_all_unhealthy_keeps_configured_order(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        for model in ('model-A', 'model-B', 'model-C'):
            for _ in range(5):
                tracker.record_429(model)
        # No healthy candidate: do NOT freeze - keep the configured order
        # (active first) so the breaker/backoff still throttle the storm.
        assert tracker.reorder_by_health(['model-A', 'model-B', 'model-C']) == [
            'model-A',
            'model-B',
            'model-C',
        ]

    @pytest.mark.unit
    def test_single_or_empty_candidate_unchanged(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        for _ in range(5):
            tracker.record_429('model-A')
        assert tracker.reorder_by_health([]) == []
        assert tracker.reorder_by_health(['model-A']) == ['model-A']


class TestPerModelStats:
    @pytest.mark.unit
    def test_stats_reflect_every_outcome(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        tracker.record_429('model-A')
        tracker.record_empty('model-A')
        tracker.record_transient('model-A')
        tracker.record_success('model-B')
        stats = tracker.per_model_stats()
        assert stats['model-A'] == {
            '429': 1,
            'empty': 1,
            'failures': 1,
            'successes': 0,
            'healthy': False,
            'last_429_ts': clock(),
        }
        assert stats['model-B']['successes'] == 1
        assert stats['model-B']['healthy'] is True
        assert stats['model-B']['last_429_ts'] is None

    @pytest.mark.unit
    def test_window_prunes_old_events(self):
        clock = FakeClock()
        tracker = make_tracker(clock)
        tracker.record_429('model-A')
        tracker.record_success('model-B')
        clock.advance(61.0)
        stats = tracker.per_model_stats()
        # Old 429 and success fall out of the 60s window.
        assert stats['model-A']['429'] == 0
        assert stats['model-A']['healthy'] is True
        assert stats['model-B']['successes'] == 0
