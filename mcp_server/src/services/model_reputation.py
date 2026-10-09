"""In-memory per-model reputation tracking for the journal worker pool.

Phase 4 (per-model routing): while the active extraction model is in a 429
storm, the pool must prefer live fallback channels when claiming rows. This
module keeps a small sliding-window reputation per model (429 / empty /
transient / success) so the failover candidate ordering can put healthy
channels first and the /health snapshot can expose per-model health.

All state is in-memory and lost on restart: the window is short (default
300s), restarts are rare, and a fresh process treating every channel as
healthy is exactly the desired behaviour (the breaker and intake keep
working while the worker drains the journal on whatever channel is live).
No LLM, no graph DB, no network.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _PerModelStats:
    """Sliding-window event timestamps for a single model."""

    last_429_ts: float | None = None
    _429: list[float] = field(default_factory=list)
    _empty: list[float] = field(default_factory=list)
    _transient: list[float] = field(default_factory=list)
    _success: list[float] = field(default_factory=list)


class ModelReputationTracker:
    """Thread-safe in-memory reputation per LLM model over a sliding window.

    ``model_is_healthy`` drives candidate ordering in the journal worker: a
    model is healthy when its 429 rate over the window is below the threshold
    AND no 429 happened within the cooldown. Channels with no recorded events
    are healthy. Empty / transient / success events are kept for statistics
    only (``per_model_stats``) and do not affect health.
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        per_min_threshold: float = 5.0,
        cooldown_seconds: float = 60.0,
        time_fn=None,
    ):
        self._window = float(window_seconds)
        self._per_min_threshold = float(per_min_threshold)
        self._cooldown = float(cooldown_seconds)
        self._time_fn = time_fn or time.monotonic
        self._lock = threading.Lock()
        self._models: dict[str, _PerModelStats] = {}

    def _stats_locked(self, model: str, now: float) -> _PerModelStats:
        """Return (creating if needed) pruned stats for a model, lock held."""
        stats = self._models.setdefault(model, _PerModelStats())
        self._prune(stats, now)
        return stats

    def _prune(self, stats: _PerModelStats, now: float) -> None:
        """Drop events older than the window (lock held by caller)."""
        cutoff = now - self._window
        stats._429 = [t for t in stats._429 if t > cutoff]
        stats._empty = [t for t in stats._empty if t > cutoff]
        stats._transient = [t for t in stats._transient if t > cutoff]
        stats._success = [t for t in stats._success if t > cutoff]

    def _record(self, model: str, kind: str) -> None:
        ts = self._time_fn()
        with self._lock:
            stats = self._stats_locked(model, ts)
            if kind == '429':
                stats._429.append(ts)
                stats.last_429_ts = ts
            elif kind == 'empty':
                stats._empty.append(ts)
            elif kind == 'transient':
                stats._transient.append(ts)
            else:  # success
                stats._success.append(ts)

    def record_429(self, model: str) -> None:
        """Record a rate-limit (429) outcome for a model."""
        self._record(model, '429')

    def record_empty(self, model: str) -> None:
        """Record an empty-response outcome for a model."""
        self._record(model, 'empty')

    def record_transient(self, model: str) -> None:
        """Record a transient (5xx / timeout / transport) outcome for a model."""
        self._record(model, 'transient')

    def record_success(self, model: str) -> None:
        """Record a successful outcome for a model."""
        self._record(model, 'success')

    def _healthy_locked(self, stats: _PerModelStats, now: float) -> bool:
        """Health over pruned stats, lock held (see ``model_is_healthy``)."""
        if not stats._429:
            return True
        rate = len(stats._429) / self._window * 60.0
        if rate >= self._per_min_threshold:
            return False
        return stats.last_429_ts is None or (now - stats.last_429_ts) >= self._cooldown

    def model_is_healthy(self, model: str) -> bool:
        """Whether a model is considered live for candidate ordering.

        A model is unhealthy when its 429 rate over the window meets the
        threshold, or a 429 happened within the cooldown. No recorded 429s
        (fresh model, or past a silent period) means healthy.
        """
        now = self._time_fn()
        with self._lock:
            stats = self._models.get(model)
            if stats is None:
                return True
            self._prune(stats, now)
            return self._healthy_locked(stats, now)

    def reorder_by_health(self, ordered: list[str]) -> list[str]:
        """Reorder failover candidates: healthy channels first, rest unchanged.

        The input preserves the configured priority (active model first, then
        fallbacks in order). Healthy candidates keep their relative order and
        are put ahead of unhealthy ones. When every candidate is unhealthy the
        list is returned unchanged (configured order, active first) so the
        worker never freezes on a fully degraded set - the breaker and
        backoff still throttle.
        """
        if len(ordered) <= 1:
            return list(ordered)
        now = self._time_fn()
        flags: list[tuple[str, bool]] = []
        with self._lock:
            for model in ordered:
                stats = self._models.get(model)
                if stats is None:
                    flags.append((model, True))
                else:
                    self._prune(stats, now)
                    flags.append((model, self._healthy_locked(stats, now)))
        healthy = [m for m, h in flags if h]
        if not healthy:
            return list(ordered)
        return healthy + [m for m, h in flags if not h]

    def per_model_stats(self) -> dict[str, dict[str, Any]]:
        """JSON-serializable per-model stats for /health (read-only).

        Returns {model: {'429': n, 'empty': n, 'failures': n, 'successes': n,
        'healthy': bool, 'last_429_ts': float|None}} for every known model.
        """
        now = self._time_fn()
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for model in sorted(self._models):
                stats = self._models[model]
                self._prune(stats, now)
                out[model] = {
                    '429': len(stats._429),
                    'empty': len(stats._empty),
                    'failures': len(stats._transient),
                    'successes': len(stats._success),
                    'healthy': self._healthy_locked(stats, now),
                    'last_429_ts': stats.last_429_ts,
                }
        return out
