"""Circuit breaker and backpressure primitives for the episode queue.

The circuit breaker shields the LLM provider API from sustained bursts of transient
failures (429 rate limits, timeouts, 5xx). Once a consecutive threshold of
failures is crossed it trips open, causing new episode submissions to fail fast
(backpressure) instead of silently piling up in an in-memory queue that is lost
on restart. After a cooldown it enters half-open and lets a single probe through
to decide whether it can close again.
"""

import asyncio
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

__all__ = [
    'CircuitBreaker',
    'CircuitOpenError',
    'QueueCapacityExceeded',
    'is_transient_error',
]

logger = __import__('logging').getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and rejects an episode submission.

    The episode is NOT queued and NOT spooled; the caller receives this error and
    should retry add_memory after ``retry_after_seconds``.
    """


class QueueCapacityExceeded(Exception):
    """Raised when the total queued episode depth exceeds the configured maximum.

    The episode is NOT queued; the caller should retry add_memory after the queue
    drains.
    """


def is_transient_error(exc: BaseException) -> bool:
    """Return True if ``exc`` represents a transient, retryable failure.

    Permanent errors (validation, pydantic, business-logic) are NOT transient and
    do not trip the circuit breaker.
    """
    if isinstance(exc, RateLimitError):
        return True

    # httpx transport-level failures: connection/read/write/close errors and
    # timeouts (httpx.TransportError covers all of them).
    if isinstance(exc, httpx.TransportError):
        return True

    # httpx HTTP status errors: 429 (rate limit) and 5xx (server error) are
    # transient, a genuine spontaneous 4xx client error is not.
    if isinstance(exc, httpx.HTTPStatusError):
        return 429 <= exc.response.status_code < 500 or 500 <= exc.response.status_code < 600

    # Builtin network failures.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    # An empty LLM response is most often a flaky upstream, treat as transient.
    if isinstance(exc, EmptyResponseError):
        return True

    return False


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value into seconds (float).

    Supports both the numeric-seconds form and the HTTP-date form (in which case
    the delay is computed as seconds until the given date). Returns None when the
    value is missing or unparseable.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Numeric seconds.
    try:
        return max(0.0, float(stripped))
    except (TypeError, ValueError):
        pass
    # HTTP-date form.
    try:
        parsed = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_after_from_exc(exc: BaseException) -> float | None:
    """Best-effort extraction of a Retry-After / retry delay hint from the error.

    The Retry-After header may live on an httpx.HTTPStatusError wrapped deeper
    in the exception chain (e.g. ``raise RateLimitError(...) from http_status``),
    so the ``__cause__`` chain is walked looking for the first such header.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            headers = getattr(current.response, 'headers', None)
            if headers is not None:
                retry_after = _parse_retry_after(headers.get('retry-after'))
                if retry_after is not None:
                    return retry_after
        current = current.__cause__
    return None


class CircuitBreaker:
    """Concurrency-safe finite-state circuit breaker.

    Transitions:
        closed --(failure_threshold consecutive transient failures)--> open
        open --(open_timeout_seconds elapsed, or retry_after if larger)--> half_open
        half_open --(single probe success)--> closed (counters reset)
        half_open --(single probe failure)--> open (timers restart)

    Only one probe is allowed through while half-open: ``allow_request()`` lets a
    single episode proceed (setting ``_probe_in_flight``) and rejects any further
    submissions until that probe resolves via ``record_success``/``record_failure``.

    All state mutations are guarded by an ``asyncio.Lock`` and, by default, by a
    monotonic clock (``time_fn``).
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        open_timeout_seconds: float = 30.0,
        *,
        time_fn=None,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        self.open_timeout_seconds = float(open_timeout_seconds)
        self._time_fn = time_fn or time.monotonic

        self._lock = asyncio.Lock()

        self._state = 'closed'
        self._failure_count = 0
        self._retry_after_seconds = 0.0
        self._last_failure_ts: float | None = None
        self._last_retry_after: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def _due(self) -> tuple[float, float]:
        """Return (elapsed_since_trip, cooldown) for the half-open probe.

        The cooldown is the larger of ``open_timeout_seconds`` and a Retry-After
        hint carried on the tripping failure (if any).
        """
        retry_after = self.open_timeout_seconds
        if self._retry_after_seconds:
            retry_after = max(self.open_timeout_seconds, self._retry_after_seconds)
        if self._last_failure_ts is None:
            return 0.0, retry_after
        elapsed = self._time_fn() - self._last_failure_ts
        return elapsed, retry_after

    async def allow_request(self) -> bool:
        """Return True if an episode may be submitted right now (fast-path).

        While the breaker is open it rejects every request. When the cooldown has
        elapsed it transitions to half-open and lets exactly ONE probe through
        (``_probe_in_flight`` is set); any further requests while that probe is in
        flight are rejected, so no more than a single episode is ever submitted as
        the half-open probe.
        """
        async with self._lock:
            if self._state == 'open':
                elapsed, retry_after = await self._due()
                if elapsed >= retry_after:
                    self._state = 'half_open'
                    self._probe_in_flight = True
                    return True
                return False
            if self._state == 'half_open' and self._probe_in_flight:
                # A probe is already running; do not allow a second one.
                return False
            return True

    async def record_success(self) -> None:
        """Record a successful probe / request. Only closes from half_open."""
        async with self._lock:
            if self._state == 'half_open':
                self._state = 'closed'
                self._probe_in_flight = False
                self._reset()
                logger.info('Circuit breaker closed after successful probe')
            elif self._state == 'closed':
                # A clean success in the closed state resets any partial runs so a
                # single old failure does not linger, but we keep the counter at 0
                # which is already the case.
                self._reset()

    async def record_failure(self, exc: BaseException) -> None:
        """Record a transient failure. May trip the breaker open."""
        now = self._time_fn()
        async with self._lock:
            self._last_failure_ts = now
            self._last_retry_after = _retry_after_from_exc(exc)

            if self._state == 'half_open':
                # Probe failed: clear the in-flight flag, back to open with a
                # fresh cooldown.
                self._probe_in_flight = False
                self._failure_count = 1
                self._state = 'open'
                self._retry_after_seconds = self._last_retry_after or 0.0
                logger.warning('Circuit breaker probe failed, re-opening')
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = 'open'
                self._retry_after_seconds = self._last_retry_after or 0.0
                logger.warning(
                    'Circuit breaker OPEN after %d consecutive failures', self._failure_count
                )

    def _reset(self) -> None:
        self._failure_count = 0
        self._retry_after_seconds = 0.0
        self._last_failure_ts = None
        self._last_retry_after = None
        self._probe_in_flight = False

    async def get_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of breaker state."""
        async with self._lock:
            elapsed, retry_after = await self._due()
            retry_after_left = max(0.0, retry_after - elapsed) if self._state in ('open', 'half_open') else 0.0
            return {
                'state': self._state,
                'failure_count': self._failure_count,
                'retry_after_seconds': round(retry_after_left, 1),
                'last_failure_ts': self._last_failure_ts,
            }
