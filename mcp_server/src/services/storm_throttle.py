"""Per-model adaptive storm throttle for the journal worker pool.

While :class:`services.model_reputation.ModelReputationTracker` keeps a
sliding-window health signal for candidate *ordering*, this module implements
the *pacing* decision next to it: when a single model starts throwing 429s
close together, that model enters a per-model ``storm`` state and the pool
slows its call rate / caps its concurrency, while every other model keeps its
normal tempo. This is the per-model isolation the operator asked for: "у каждой
модели ozon свой rate-limit, можно обращаться с разной скоростью к разным
моделям" (model A storming must only slow model A).

Hysteresis, so a lone 429 (even a couple per hour) is never mistaken for a
storm:

- ENTER storm (per model): ``count(429)`` within the detection window
  (``storm_detection_429_window_seconds``) reaches
  ``storm_detection_429_threshold``, OR the reputation tracker reports the
  model unhealthy (its per-minute 429 rate meets the reputation threshold).
- EXIT storm: no 429 on that model for ``exit_silence_seconds`` (lazy, so a
  model that simply went quiet self-recovers without waiting for a new call),
  followed by a successful call on it — one quiet success clears the storm
  immediately (the requirement's probe-like exit).

No LLM, no graph DB, no network. All state is in-memory and thread-safe.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

STORM_EXIT_SILENCE_SECONDS = 120.0
STORM_RETRY_AFTER_CAP_SECONDS = 60.0


@dataclass
class _ModelStormState:
    """Sliding-window 429 history + storm state for a single model."""

    window: float
    _429: list[float] = field(default_factory=list)
    in_storm: bool = False
    storm_started_ts: float | None = None
    last_429_ts: float | None = None
    last_success_ts: float | None = None
    retry_after_until: float | None = None


class StormThrottle:
    """Thread-safe per-model storm detection with hysteresis + Retry-After.

    ``record_failure``/``record_success`` are fed from the journal worker's
    per-model reputation events; ``in_storm``/``is_throttled`` gate the call
    site in queue_service; ``snapshot`` feeds the /health ``storm.per_model``
    block.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        threshold: int = 6,
        exit_silence_seconds: float = STORM_EXIT_SILENCE_SECONDS,
        respect_retry_after: bool = True,
        time_fn=None,
    ):
        self._window = float(window_seconds)
        self._threshold = max(1, int(threshold))
        self._exit_silence = float(exit_silence_seconds)
        self._respect_retry_after = bool(respect_retry_after)
        self._time_fn = time_fn or time.monotonic
        self._lock = threading.Lock()
        self._models: dict[str, _ModelStormState] = {}

    def _stats_locked(self, model: str, now: float) -> _ModelStormState:
        """Return (creating if needed) window-pruned stats, lock held."""
        stats = self._models.setdefault(model, _ModelStormState(window=self._window))
        cutoff = now - self._window
        stats._429 = [t for t in stats._429 if t > cutoff]
        return stats

    def _enter_locked(self, stats: _ModelStormState, now: float) -> None:
        if not stats.in_storm:
            stats.in_storm = True
            stats.storm_started_ts = now

    def _exit_locked(self, stats: _ModelStormState) -> None:
        if stats.in_storm:
            stats.in_storm = False
            stats.storm_started_ts = None
            stats.retry_after_until = None

    def record_failure(
        self,
        model: str,
        retry_after: float | None = None,
        reputation_unhealthy: bool = False,
    ) -> None:
        """Record a 429 for a model; may enter storm (hysteresis entry)."""
        now = self._time_fn()
        with self._lock:
            stats = self._stats_locked(model, now)
            stats._429.append(now)
            stats.last_429_ts = now
            if (
                self._respect_retry_after
                and isinstance(retry_after, (int, float))
                and retry_after > 0
            ):
                stats.retry_after_until = now + min(
                    float(retry_after), STORM_RETRY_AFTER_CAP_SECONDS
                )
            if len(stats._429) >= self._threshold or reputation_unhealthy:
                self._enter_locked(stats, now)

    def record_success(self, model: str) -> None:
        """Record a successful call. A quiet success exits storm."""
        now = self._time_fn()
        with self._lock:
            stats = self._models.setdefault(
                model, _ModelStormState(window=self._window)
            )
            stats.last_success_ts = now
            if stats.in_storm:
                quiet = stats.last_429_ts is None or (
                    now - stats.last_429_ts >= self._exit_silence
                )
                if quiet:
                    self._exit_locked(stats)

    def in_storm(self, model: str) -> bool:
        """Whether a model is currently throttling (storm state)."""
        now = self._time_fn()
        with self._lock:
            stats = self._models.get(model)
            if stats is None or not stats.in_storm:
                return False
            # Lazy self-heal: a storm that has been quiet longer than the exit
            # window is over even before the next call lands a success on it.
            if stats.last_429_ts is not None and (
                now - stats.last_429_ts >= self._exit_silence
            ):
                self._exit_locked(stats)
                return False
            return True

    def is_throttled(self, model: str) -> float | None:
        """Remaining seconds of a hard Retry-After pause, or None.

        Only meaningful while the model is storming. When ``respect_retry_after``
        is off or no hint was recorded, returns None — the caller falls back to
        soft pacing (min inter-request interval / concurrency ceiling).
        """
        now = self._time_fn()
        with self._lock:
            stats = self._models.get(model)
            if (
                stats is None
                or not stats.in_storm
                or not self._respect_retry_after
                or stats.retry_after_until is None
            ):
                return None
            remaining = stats.retry_after_until - now
            if remaining <= 0:
                stats.retry_after_until = None
                return None
            return remaining

    def storm_models(self) -> list[str]:
        """Sorted list of models currently in storm state."""
        with self._lock:
            models = list(self._models)
        return [m for m in sorted(models) if self.in_storm(m)]

    def any_storm(self) -> bool:
        """Whether any known model is currently in storm state."""
        with self._lock:
            models = list(self._models)
        return any(self.in_storm(m) for m in models)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable per-model storm state for /health.

        Returns ``{'per_model': {model: {...}}}`` where each entry reports
        ``in_storm``, the hard ``throttled_retry_after`` remaining (if any) and
        the raw counters/timestamps.
        """
        now = self._time_fn()
        out: dict[str, Any] = {}
        with self._lock:
            for model in sorted(self._models):
                stats = self._models.get(model)
                if stats is None:
                    continue
                self._stats_locked(model, now)
                in_storm = bool(stats.in_storm)
                throttled: float | None = None
                if (
                    in_storm
                    and self._respect_retry_after
                    and stats.retry_after_until is not None
                ):
                    remaining = stats.retry_after_until - now
                    throttled = round(max(0.0, remaining), 1) if remaining > 0 else None
                if (
                    in_storm
                    and stats.last_429_ts is not None
                    and (now - stats.last_429_ts) >= self._exit_silence
                ):
                    # Report the lazy-heal view (silent storm counts as gone).
                    in_storm = False
                    throttled = None
                out[model] = {
                    'in_storm': in_storm,
                    'throttled_retry_after': throttled,
                    'window_429_count': len(stats._429),
                    'window_seconds': round(self._window, 1),
                    'last_429_ts': stats.last_429_ts,
                    'last_success_ts': stats.last_success_ts,
                    'storm_started_ts': stats.storm_started_ts if in_storm else None,
                }
        return {'per_model': out}
