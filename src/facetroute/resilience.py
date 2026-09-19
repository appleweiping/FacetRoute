"""Opt-in, bounded provider resilience without ambiguous duplicate generations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .errors import ConfigurationError
from .providers import ProviderError, ProviderFailure, ProviderRegistry


@dataclass(frozen=True, slots=True)
class ProviderResiliencePolicy:
    """Per-model circuit breaker and optional safe pre-send retries.

    ``max_attempts=1`` never retries. Even with more attempts, only errors
    explicitly certified as occurring before request bytes were sent qualify.
    """

    max_attempts: int = 1
    initial_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 2.0
    failure_threshold: int = 3
    recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 5:
            raise ConfigurationError("max_attempts must be an integer from 1 to 5")
        if type(self.failure_threshold) is not int or not 1 <= self.failure_threshold <= 100:
            raise ConfigurationError("failure_threshold must be an integer from 1 to 100")
        for name, value, lower, upper in (
            ("initial_backoff_seconds", self.initial_backoff_seconds, 0.0, 30.0),
            ("max_backoff_seconds", self.max_backoff_seconds, 0.0, 30.0),
            ("recovery_seconds", self.recovery_seconds, 0.001, 3600.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ConfigurationError(f"{name} must be finite and between {lower} and {upper}")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ConfigurationError("max_backoff_seconds must be at least initial_backoff_seconds")


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0
    probe_in_flight: bool = False
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class _Admission:
    probe: bool
    epoch: int


_COUNTED_FAILURES = frozenset(
    {
        ProviderFailure.TIMEOUT,
        ProviderFailure.RATE_LIMITED,
        ProviderFailure.UNAVAILABLE,
        ProviderFailure.MALFORMED,
        ProviderFailure.FAILED,
    }
)
_RETRY_FAILURES = frozenset(
    {ProviderFailure.TIMEOUT, ProviderFailure.UNAVAILABLE, ProviderFailure.FAILED}
)


class _CircuitBreaker:
    """One bounded state cell per configured model, guarded across threads."""

    def __init__(
        self,
        model_ids: frozenset[str],
        policy: ProviderResiliencePolicy,
        clock: Callable[[], float],
    ) -> None:
        self._states = {model_id: _CircuitState() for model_id in model_ids}
        self._policy = policy
        self._clock = clock
        self._lock = Lock()

    def enter(self, model_id: str) -> _Admission:
        with self._lock:
            state = self._states.get(model_id)
            if state is None:
                raise ProviderError(ProviderFailure.NOT_CONFIGURED)
            if state.open_until:
                if self._clock() < state.open_until or state.probe_in_flight:
                    raise ProviderError(ProviderFailure.UNAVAILABLE)
                state.probe_in_flight = True
                return _Admission(probe=True, epoch=state.epoch)
            return _Admission(probe=False, epoch=state.epoch)

    def finish(
        self,
        model_id: str,
        admission: _Admission,
        failure: ProviderFailure | None,
    ) -> bool:
        """Release the admission; return whether this completion opened the circuit."""

        with self._lock:
            state = self._states[model_id]
            if state.epoch != admission.epoch:
                return bool(state.open_until)
            if admission.probe:
                state.probe_in_flight = False
                if failure is None or failure not in _COUNTED_FAILURES:
                    state.failures = 0
                    state.open_until = 0.0
                    return False
                self._open(state)
                return True
            if state.open_until:
                # A concurrent request entered while the circuit was closed.
                # Its late completion cannot override the newer open state.
                return True
            if failure is None:
                state.failures = 0
            elif failure in _COUNTED_FAILURES:
                state.failures += 1
                if state.failures >= self._policy.failure_threshold:
                    self._open(state)
                    return True
            return False

    def abandon(self, model_id: str, admission: _Admission) -> None:
        """A caller stop is neutral, except an unproven half-open probe."""

        with self._lock:
            state = self._states[model_id]
            if state.epoch == admission.epoch and admission.probe:
                self._open(state)

    def _open(self, state: _CircuitState) -> None:
        state.failures = self._policy.failure_threshold
        state.open_until = self._clock() + self._policy.recovery_seconds
        state.probe_in_flight = False
        state.epoch += 1


class ResilientProviderRegistry:
    """Wrap the existing provider boundary without changing its default path."""

    def __init__(
        self,
        registry: ProviderRegistry,
        policy: ProviderResiliencePolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(registry, ProviderRegistry):
            raise ConfigurationError("registry must be a ProviderRegistry")
        if not isinstance(policy, ProviderResiliencePolicy):
            raise ConfigurationError("policy must be a ProviderResiliencePolicy")
        self._registry = registry
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._breaker = _CircuitBreaker(registry.model_ids, policy, clock)

    @property
    def model_ids(self) -> frozenset[str]:
        return self._registry.model_ids

    def _deadline(self, timeout_seconds: float) -> float:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("provider timeout must be positive")
        return self._clock() + timeout_seconds

    def _next_timeout(
        self, error: ProviderError, attempt: int, deadline: float, opened: bool
    ) -> float | None:
        if (
            opened
            or attempt >= self._policy.max_attempts
            or not error.retry_safe
            or error.failure not in _RETRY_FAILURES
        ):
            return None
        delay = min(
            self._policy.initial_backoff_seconds * 2 ** (attempt - 1),
            self._policy.max_backoff_seconds,
        )
        remaining = deadline - self._clock()
        if remaining <= delay:
            return None
        self._sleep(delay)
        remaining = deadline - self._clock()
        return remaining if remaining > 0 else None

    def complete(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        deadline = self._deadline(timeout_seconds)
        remaining = timeout_seconds
        for attempt in range(1, self._policy.max_attempts + 1):
            admission = self._breaker.enter(model_id)
            try:
                result = self._registry.complete(model_id, payload, timeout_seconds=remaining)
            except ProviderError as error:
                opened = self._breaker.finish(model_id, admission, error.failure)
                next_timeout = self._next_timeout(error, attempt, deadline, opened)
                if next_timeout is None:
                    raise
                remaining = next_timeout
            else:
                self._breaker.finish(model_id, admission, None)
                return result
        raise AssertionError("bounded retry loop did not terminate")  # pragma: no cover

    def stream(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        deadline = self._deadline(timeout_seconds)

        def events() -> Iterator[Mapping[str, Any]]:
            remaining = timeout_seconds
            for attempt in range(1, self._policy.max_attempts + 1):
                admission = self._breaker.enter(model_id)
                source: Iterator[Mapping[str, Any]] | None = None
                completed = False
                emitted = False
                try:
                    source = self._registry.stream(model_id, payload, timeout_seconds=remaining)
                    for chunk in source:
                        emitted = True
                        yield chunk
                    if not emitted:
                        raise ProviderError(ProviderFailure.MALFORMED)
                    self._breaker.finish(model_id, admission, None)
                    completed = True
                    return
                except ProviderError as error:
                    opened = self._breaker.finish(model_id, admission, error.failure)
                    completed = True
                    next_timeout = (
                        None if emitted else self._next_timeout(error, attempt, deadline, opened)
                    )
                    if next_timeout is None:
                        raise
                    remaining = next_timeout
                finally:
                    try:
                        if source is not None:
                            close = getattr(source, "close", None)
                            if callable(close):
                                with suppress(Exception):
                                    close()
                    finally:
                        if not completed:
                            # Caller stopped consuming a half-open probe. It
                            # does not establish provider health; an ordinary
                            # closed-circuit client stop remains neutral.
                            self._breaker.abandon(model_id, admission)
            raise AssertionError("bounded retry loop did not terminate")  # pragma: no cover

        return events()
