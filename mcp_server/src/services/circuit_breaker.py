"""Circuit breaker and backpressure primitives for the episode queue.

The circuit breaker shields the LLM provider API from sustained bursts of transient
failures (429 rate limits, timeouts, 5xx). Once a consecutive threshold of
failures is crossed it trips open, causing new episode submissions to fail fast
(backpressure) instead of silently piling up in an in-memory queue that is lost
on restart. After a cooldown it enters half-open and lets a single probe through
to decide whether it can close again.
"""

import asyncio
import json
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
    'is_permanent_error',
]

logger = __import__('logging').getLogger(__name__)


def _openai_timeout_classes() -> tuple[type[BaseException], ...]:
    """Cached tuple of the openai SDK's timeout / connection error classes.

    The import is guarded: a runtime without the ``openai`` package, or an SDK
    version that renamed the exception classes, must never break the transient
    classification - the classes are only ever consulted via ``isinstance``.
    """
    classes = getattr(_openai_timeout_classes, '_cache', None)
    if classes is None:
        collected: list[type[BaseException]] = []
        try:
            from openai import APIConnectionError, APITimeoutError
        except Exception:  # package missing or the classes were renamed
            pass
        else:
            for cls in (APITimeoutError, APIConnectionError):
                if isinstance(cls, type) and issubclass(cls, BaseException):
                    collected.append(cls)
        classes = tuple(collected)
        _openai_timeout_classes._cache = classes  # type: ignore[attr-defined]
    return classes


def is_openai_timeout_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an openai SDK APITimeoutError / APIConnectionError.

    The openai SDK wraps httpx timeouts / connection failures into
    ``openai.APITimeoutError`` ("Request timed out.") and
    ``openai.APIConnectionError``. These are transient provider failures - the
    call never completed, unlike a hard 4xx. They must trip the circuit breaker
    and feed the model-failover ring so a timing-out primary falls over to the
    zai / fallback channels instead of blocking the queue.
    """
    return any(isinstance(exc, cls) for cls in _openai_timeout_classes())


def _openai_api_status_classes() -> tuple[type[BaseException], ...]:
    """Cached tuple of the openai SDK's HTTP-status error classes.

    ``openai.APIStatusError`` is the base of every SDK error raised for an HTTP
    response (``RateLimitError`` 429, ``BadRequestError`` 400,
    ``InternalServerError`` 500, ...); ``InternalServerError`` is its 500
    subclass. Imports are guarded exactly like the timeout classes: a runtime
    without the ``openai`` package must never break classification.
    """
    classes = getattr(_openai_api_status_classes, '_cache', None)
    if classes is None:
        collected: list[type[BaseException]] = []
        try:
            from openai import APIStatusError, InternalServerError
        except Exception:  # package missing or the classes were renamed
            pass
        else:
            for cls in (APIStatusError, InternalServerError):
                if isinstance(cls, type) and issubclass(cls, BaseException):
                    collected.append(cls)
        classes = tuple(collected)
        _openai_api_status_classes._cache = classes  # type: ignore[attr-defined]
    return classes


def _openai_api_status_code(exc: BaseException) -> int | None:
    """The HTTP status code of an openai SDK status error, or None.

    ``InternalServerError`` is a fixed 500 subclass; other ``APIStatusError``
    subclasses carry ``status_code``. A non-SDK exception returns None.
    """
    for cls in _openai_api_status_classes():
        if isinstance(exc, cls):
            status = getattr(exc, 'status_code', None)
            if isinstance(status, int):
                return status
            if cls.__name__ == 'InternalServerError':
                return 500
            break
    return None


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

    # asyncio.wait_for timeouts raise asyncio.TimeoutError, which on Python 3.10
    # is a distinct class (asyncio.exceptions.TimeoutError) and NOT a subclass of
    # builtin TimeoutError. A timed-out LLM call is a transient failure and must
    # trip the breaker / trigger model failover.
    if isinstance(exc, asyncio.TimeoutError):
        return True

    # The openai SDK wraps httpx timeouts / connection failures into
    # openai.APITimeoutError ("Request timed out.") / APIConnectionError - a
    # provider line that never completes is transient (feeds failover + ring).
    if is_openai_timeout_error(exc):
        return True

    # The openai SDK surfaces provider HTTP responses as APIStatusError
    # subclasses: RateLimitError (429) and InternalServerError / others 5xx are
    # transient (the gateway is up but rate-limiting or failing), a hard 4xx is
    # not. Without this branch an SDK 502/5xx was classified non-transient and
    # never fed the breaker / failover (RCA C7).
    status = _openai_api_status_code(exc)
    if status is not None:
        return status >= 500 or status == 429

    # Neo4j driver network / session failures (ServiceUnavailable,
    # SessionExpired, auth expiry, TransientError, ...) are transient: the
    # database is unreachable or the session was broken, and retrying after the
    # network recovers is meaningful. Only the client-side *data* errors
    # (ClientError / database errors) are treated as permanent.
    if _neo4j_exc_name(exc) is not None and _is_neo4j_transient_exception(exc):
        return True

    # A malformed LLM response body (truncated JSON, "Unterminated string") is
    # most often an upstream glitch - the gateway streams a partial body and the
    # provider cut it mid-token. Retry / failover to a fallback model usually
    # recovers, so treat it as transient (mirrors
    # graphiti_core.llm_client.client.is_server_or_retry_error).
    if isinstance(exc, json.JSONDecodeError):
        return True

    # An empty LLM response is most often a flaky upstream, treat as transient.
    return bool(isinstance(exc, EmptyResponseError))


# --- permanent-error classification (RCA C1) --------------------------------

try:
    from pydantic import ValidationError as _PydanticValidationError  # type: ignore
except Exception:  # pragma: no cover - pydantic is always present in practice
    _PydanticValidationError = None


def _val_error_classes() -> tuple[type[BaseException], ...]:
    """The graphiti-core validation/not-found error classes (permanent)."""
    classes = getattr(_val_error_classes, '_cache', None)
    if classes is None:
        collected: list[type[BaseException]] = []
        try:
            from graphiti_core.errors import (
                EntityTypeValidationError,
                GraphitiError,
                GroupIdValidationError,
            )
        except Exception:  # package missing or the names were renamed
            pass
        else:
            for cls in (GraphitiError, EntityTypeValidationError, GroupIdValidationError):
                if isinstance(cls, type) and issubclass(cls, BaseException):
                    collected.append(cls)
        classes = tuple(collected)
        _val_error_classes._cache = classes  # type: ignore[attr-defined]
    return classes


def _neo4j_exc_classes() -> tuple[type[BaseException], ...]:
    """The neo4j driver error classes relevant to retry classification.

    Client-side data errors (ClientError, CypherSyntaxError, CypherTypeError,
    DatabaseError) are permanent; the network / session classes
    (ServiceUnavailable, SessionExpired, TransientError, auth, ...) are
    transient. Both families are collected so ``_neo4j_exc_name`` can recognize
    any neo4j exception that reaches the classifier.
    """
    classes = getattr(_neo4j_exc_classes, '_cache', None)
    if classes is None:
        collected: list[type[BaseException]] = []
        try:
            from neo4j.exceptions import (
                AuthConfigurationError,
                AuthError,
                ClientError,
                DatabaseError,
                DatabaseUnavailable,
                DriverError,
                ServiceUnavailable,
                SessionExpired,
                TransientError,
            )
        except Exception:  # package missing or the names were renamed
            pass
        else:
            for cls in (
                ClientError,
                DatabaseError,
                ServiceUnavailable,
                SessionExpired,
                AuthError,
                AuthConfigurationError,
                TransientError,
                DatabaseUnavailable,
                DriverError,
            ):
                if isinstance(cls, type) and issubclass(cls, BaseException):
                    collected.append(cls)
        classes = tuple(collected)
        _neo4j_exc_classes._cache = classes  # type: ignore[attr-defined]
    return classes


def _neo4j_exc_name(exc: BaseException) -> str | None:
    """The concrete neo4j exception class name, or None when not a neo4j error."""
    for cls in _neo4j_exc_classes():
        if isinstance(exc, cls):
            return type(exc).__name__
    return None


def _is_neo4j_transient_exception(exc: BaseException) -> bool:
    """Whether a neo4j exception is a transient (network-ish) failure.

    ``ServiceUnavailable`` / ``SessionExpired`` / auth / transient errors mean
    the server is unreachable, the session was broken or the data is fine but
    the database is transiently unable - retrying after the network recovers is
    meaningful. A ``ClientError`` / ``DatabaseError`` (Cypher syntax / data type
    mistake, e.g. the 535 “Neo4j TypeError MAP”) will never fix itself and must
    fast-fail (RCA C1).
    """
    transient_names = {
        'ServiceUnavailable',
        'SessionExpired',
        'AuthError',
        'AuthConfigurationError',
        'AuthenticationError',
        'TransientError',
        'DatabaseUnavailable',
        'RoutingServiceUnavailable',
        'ReadServiceUnavailable',
        'WriteServiceUnavailable',
        'IncompleteCommit',
        'NotALeader',
        'ForbiddenOnReadOnlyDatabase',
    }
    name = _neo4j_exc_name(exc)
    return name is not None and name in transient_names


def is_permanent_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is a permanent (non-recoverable) failure.

    Permanent errors cannot be fixed by retrying the same input against the same
    provider - they are data/contract problems (bad LLM response shape, malformed
    Cypher input, a hard client 4xx). Retrying them burns up to
    ``max_spool_attempts`` full attempt budgets (~10 x up-to-75min each) - the
    "poison zombie" cycle - instead of failing fast (RCA C1).

    Classification precedence (an exception matching several rules takes the
    first hit):
    - pydantic ValidationError (incl. the graphiti ``EdgeDuplicate`` / entity /
      edge JSON-schema failures) -> permanent;
    - neo4j client-side errors (ClientError / DatabaseError) with the exception
      of explicitly transient network classes -> permanent;
    - openai SDK HTTP errors: 4xx (except 429 rate limit) -> permanent;
    - otherwise the inverse of ``is_transient_error`` (a failure that is not
      transient is treated as permanent).
    """
    # pydantic model validation - the class to which the 375/537/540 zombies
    # belong ("Document title Field required" / SummarizedEntities / EdgeDuplicate).
    if _PydanticValidationError is not None and isinstance(exc, _PydanticValidationError):
        return True

    # graphiti-core validation / not-found errors (permanent data problems).
    if any(isinstance(exc, cls) for cls in _val_error_classes()):
        return True

    # neo4j: client-side data errors are permanent; the network-ish classes are
    # handled by is_transient_error (they subclass the driver base errors which
    # the fallback below would otherwise classify via repr/name).
    if _neo4j_exc_name(exc) is not None:
        return not _is_neo4j_transient_exception(exc)

    # openai SDK HTTP errors: a hard 4xx (other than 429) is permanent.
    status = _openai_api_status_code(exc)
    if status is not None:
        return status < 500 and status != 429

    # Default: non-transient -> permanent.
    return not is_transient_error(exc)


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
        # graphiti RateLimitError carries the provider's hint as an attribute.
        attr = getattr(current, 'retry_after', None)
        if isinstance(attr, (int, float)) and attr > 0:
            return float(attr)
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
    A probe that is not resolved within ``probe_timeout_seconds`` is re-granted on
    the next ``allow_request()`` so a consumed-but-forgotten probe cannot strand
    the breaker in half_open forever.

    All state mutations are guarded by an ``asyncio.Lock`` and, by default, by a
    monotonic clock (``time_fn``).
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        open_timeout_seconds: float = 30.0,
        *,
        probe_timeout_seconds: float = 60.0,
        time_fn=None,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.probe_timeout_seconds = float(probe_timeout_seconds)
        self._time_fn = time_fn or time.monotonic

        self._lock = asyncio.Lock()

        self._state = 'closed'
        self._failure_count = 0
        self._retry_after_seconds = 0.0
        self._last_failure_ts: float | None = None
        self._last_retry_after: float | None = None
        self._probe_in_flight = False
        self._probe_granted_ts: float | None = None

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
                    self._probe_granted_ts = self._time_fn()
                    return True
                return False
            if self._state == 'half_open' and self._probe_in_flight:
                # A probe is already running; do not allow a second one — unless
                # it was granted so long ago that its caller must have vanished
                # without calling record_success/record_failure (e.g. the spool
                # retryer probed with nothing to replay). Re-grant the slot so a
                # lost probe cannot strand the breaker in half_open forever.
                probe_age = self._time_fn() - (self._probe_granted_ts or 0.0)
                if probe_age >= self.probe_timeout_seconds:
                    logger.warning(
                        'Half-open probe unresolved for %.1fs (timeout %.1fs), re-granting the probe slot',
                        probe_age,
                        self.probe_timeout_seconds,
                    )
                    self._probe_granted_ts = self._time_fn()
                    return True
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
        self._probe_granted_ts = None

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
