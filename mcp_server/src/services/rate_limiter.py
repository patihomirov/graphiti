"""Always-active per-channel LLM rate limiter for the journal worker pool.

Unlike the reactive :class:`services.storm_throttle.StormThrottle` (which only
paces a model AFTER it starts throwing 429s), this module implements the
honest, always-on pacing the operator asked for (2026-10-21): "лимитируем
скорость обработки очереди со своей стороны и не будем вызывать обработку
чаще". The gateway (``llm-gateway-proton``) cuts PACKET calls (a 12/12 batch
comes back 429 even when a single call is 200 OK) and does NOT return
Retry-After, so the only robust fix is to keep the request stream per
``(provider, model)`` channel below a hard calls-per-minute ceiling and keep it
uniform - never bursty.

Implementation: a strict sliding-window counter per channel. A call records its
monotonic timestamp; a timestamp leaves the window after ``window_seconds``.
``max_calls_per_minute`` bounds how many timestamps may live in the window at
once, so the number of calls in ANY ``window_seconds``-long interval never
exceeds it. When the window is full the caller waits (asynchronously) until the
oldest timestamp falls out - a built-in smooth slotting that never bursts, and
the wait happens on the caller's own coroutine, so one worker waiting never
blocks the rest of the pool.

Fairness under 429: the limiter has NO branch that loosens the rate after a
429 - a rate-limited call still occupies its window slot, so a storm cannot
feed itself (it only keeps the stream at the configured ceiling). The
StormThrottle / reputation layers above still react to the 429 signal
(candidate ordering, hard pauses); this module is purely prophylactic.

No LLM, no network. All state is in-memory and guarded by an asyncio lock held
only briefly (the critical sections contain no awaits).

``max_calls_per_minute = 0`` keeps the legacy behaviour (OFF): ``acquire`` is a
no-op and nothing is paced.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from typing import Any


class ChannelRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by ``(provider, model)`` channel.

    ``acquire(channel)`` parks the caller until a slot in the channel's window
    is free, then records the call and returns the number of seconds waited.
    ``estimated_wait_seconds(channel)`` answers, without consuming a slot, how
    long the next ``acquire`` would wait - used by the failover loop to decide
    "wait here" vs "move to the next candidate". ``enabled`` is False when the
    limit is 0 (legacy OFF), in which case ``acquire``/``estimated_wait_seconds``
    are cheap no-ops.
    """

    def __init__(
        self,
        max_calls_per_minute: int = 0,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[..., Any] | None = None,
    ):
        self._max_calls = max(0, int(max_calls_per_minute or 0))
        self._window = float(window_seconds or 60.0)
        self._time_fn = time_fn or time.monotonic
        self._sleep = sleep_fn if sleep_fn is not None else asyncio.sleep
        self._lock = asyncio.Lock()
        # channel -> deque of call timestamps (newest at the tail).
        self._channels: dict[str, deque[float]] = {}
        # Pacing statistics surfaced under /health -> pacing -> per_channel.
        self._total_waits = 0
        self._total_wait_seconds = 0.0
        self._per_channel_waits: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._max_calls > 0

    @property
    def max_calls_per_minute(self) -> int:
        return self._max_calls

    @property
    def window_seconds(self) -> float:
        return self._window

    def current_calls(self, channel: str) -> int:
        """Number of recorded calls of ``channel`` still inside the window.

        Read-only helper for tests and /health; prunes expired timestamps.
        """
        if not self.enabled:
            return 0
        now = self._time_fn()
        dq = self._channels.get(channel)
        if not dq:
            return 0
        cutoff = now - self._window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        return len(dq)

    def _estimated_locked(self, channel: str, now: float) -> float:
        """Seconds until the oldest in-window call of ``channel`` leaves (lock held)."""
        dq = self._channels.get(channel)
        if not dq:
            return 0.0
        cutoff = now - self._window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) < self._max_calls:
            return 0.0
        oldest = dq[0]
        return max(0.0, oldest + self._window - now)

    async def estimated_wait_seconds(self, channel: str) -> float:
        """How long the next ``acquire`` on ``channel`` would wait, 0.0 if free/OFF.

        Does not consume a slot. The journal worker uses this to decide, before
        starting an episode on a channel, whether to proceed (short wait: the
        client will pace the individual calls anyway) or to fail over to the
        next candidate (wait would exceed the abort window).
        """
        if not self.enabled:
            return 0.0
        now = self._time_fn()
        async with self._lock:
            return self._estimated_locked(channel, now)

    async def acquire(self, channel: str) -> float:
        """Wait for a free slot on ``channel``, record the call, return seconds waited.

        Returns 0.0 when no wait was needed (or the limiter is OFF). The wait
        is performed with ``await`` on the caller's coroutine, so a saturated
        channel parks only the caller, never the whole pool.
        """
        if not self.enabled:
            return 0.0
        waited = 0.0
        while True:
            async with self._lock:
                now = self._time_fn()
                dq = self._channels.setdefault(channel, deque())
                cutoff = now - self._window
                while dq and dq[0] <= cutoff:
                    dq.popleft()
                if len(dq) < self._max_calls:
                    dq.append(now)
                    return waited
                wait = self._estimated_locked(channel, now)
                if wait <= 0:
                    continue
                self._total_waits += 1
                self._per_channel_waits[channel] = self._per_channel_waits.get(channel, 0) + 1
                self._total_wait_seconds += wait
            await self._sleep(wait)
            waited += wait

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable pacing state for /health (read-only).

        Reports the configured ceiling and window, cumulative wait statistics
        and a per-channel breakdown (how many calls are currently inside the
        window, how many times a caller had to wait on that channel).
        """
        channels: dict[str, dict[str, Any]] = {}
        for channel in sorted(self._channels):
            channels[channel] = {
                'current_calls_in_window': self.current_calls(channel),
                'times_waited': self._per_channel_waits.get(channel, 0),
            }
        return {
            'limit_per_minute': self._max_calls,
            'window_seconds': round(self._window, 1),
            'total_waits': self._total_waits,
            'total_wait_seconds': round(self._total_wait_seconds, 2),
            'channels': channels,
        }
