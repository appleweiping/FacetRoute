"""Injectable asynchronous routing controller and provider client boundary.

The async transport is supplied by the application. FacetRoute never turns a
blocking HTTP call into a misleading coroutine or reads credentials here.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import ConfigurationError
from .providers import (
    _MAX_PROVIDER_BINDINGS,
    ProviderError,
    ProviderFailure,
    _validate_chunk,
    _validate_completion,
)
from .resilience import (
    _RETRY_FAILURES,
    ProviderResiliencePolicy,
    _CircuitBreaker,
)
from .routers import Router
from .types import RouteDecision, RouteRequest


class AsyncChatCompletionProvider(Protocol):
    """Non-blocking transport contract for one fixed upstream provider."""

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]: ...

    def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class AsyncProviderTarget:
    model_id: str
    upstream_model: str
    provider: AsyncChatCompletionProvider

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id.strip() or len(self.model_id) > 256:
            raise ConfigurationError("provider model_id must contain at most 256 characters")
        if (
            type(self.upstream_model) is not str
            or not self.upstream_model.strip()
            or len(self.upstream_model) > 512
        ):
            raise ConfigurationError("upstream_model must contain at most 512 characters")
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "upstream_model", self.upstream_model.strip())


class AsyncProviderRegistry:
    """Fixed model bindings with optional shared-policy breaker and retries."""

    def __init__(
        self,
        targets: tuple[AsyncProviderTarget, ...],
        *,
        policy: ProviderResiliencePolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not targets:
            raise ConfigurationError("provider registry cannot be empty")
        if len(targets) > _MAX_PROVIDER_BINDINGS:
            raise ConfigurationError(
                f"provider registry cannot exceed {_MAX_PROVIDER_BINDINGS} model bindings"
            )
        if policy is not None and not isinstance(policy, ProviderResiliencePolicy):
            raise ConfigurationError("policy must be a ProviderResiliencePolicy")
        by_model = {target.model_id: target for target in targets}
        if len(by_model) != len(targets):
            raise ConfigurationError("provider registry contains duplicate model_id values")
        self._targets = by_model
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._breaker = (
            _CircuitBreaker(frozenset(by_model), policy, clock) if policy is not None else None
        )

    @property
    def model_ids(self) -> frozenset[str]:
        return frozenset(self._targets)

    def _target(self, model_id: str) -> AsyncProviderTarget:
        try:
            return self._targets[model_id]
        except KeyError as exc:
            raise ProviderError(ProviderFailure.NOT_CONFIGURED) from exc

    def _deadline(self, timeout_seconds: float) -> float:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("provider timeout must be positive")
        return self._clock() + timeout_seconds

    async def _next_timeout(
        self, error: ProviderError, attempt: int, deadline: float, opened: bool
    ) -> float | None:
        policy = self._policy
        if (
            policy is None
            or opened
            or attempt >= policy.max_attempts
            or not error.retry_safe
            or error.failure not in _RETRY_FAILURES
        ):
            return None
        delay = min(policy.initial_backoff_seconds * 2 ** (attempt - 1), policy.max_backoff_seconds)
        if deadline - self._clock() <= delay:
            return None
        await self._sleep(delay)
        remaining = deadline - self._clock()
        return remaining if remaining > 0 else None

    async def complete(
        self, model_id: str, payload: Mapping[str, Any], *, timeout_seconds: float
    ) -> Mapping[str, Any]:
        target = self._target(model_id)
        deadline = self._deadline(timeout_seconds)
        upstream_payload = dict(payload)
        upstream_payload["model"] = target.upstream_model
        upstream_payload["stream"] = False
        remaining = timeout_seconds
        attempts = self._policy.max_attempts if self._policy is not None else 1
        for attempt in range(1, attempts + 1):
            admission = self._breaker.enter(model_id) if self._breaker is not None else None
            try:
                result = await target.provider.complete(
                    upstream_payload, model=target.upstream_model, timeout_seconds=remaining
                )
                validated = _validate_completion(result)
            except asyncio.CancelledError:
                if self._breaker is not None and admission is not None:
                    self._breaker.abandon(model_id, admission)
                raise
            except ProviderError as error:
                opened = (
                    self._breaker.finish(model_id, admission, error.failure)
                    if self._breaker is not None and admission is not None
                    else False
                )
                next_timeout = await self._next_timeout(error, attempt, deadline, opened)
                if next_timeout is None:
                    raise
                remaining = next_timeout
            except TimeoutError as exc:
                timeout_error = ProviderError(ProviderFailure.TIMEOUT)
                if self._breaker is not None and admission is not None:
                    self._breaker.finish(model_id, admission, timeout_error.failure)
                raise timeout_error from exc
            except Exception as exc:
                failed_error = ProviderError(ProviderFailure.FAILED)
                if self._breaker is not None and admission is not None:
                    self._breaker.finish(model_id, admission, failed_error.failure)
                raise failed_error from exc
            else:
                if self._breaker is not None and admission is not None:
                    self._breaker.finish(model_id, admission, None)
                return validated
        raise AssertionError("bounded retry loop did not terminate")  # pragma: no cover

    def stream(
        self, model_id: str, payload: Mapping[str, Any], *, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        target = self._target(model_id)
        deadline = self._deadline(timeout_seconds)
        upstream_payload = dict(payload)
        upstream_payload["model"] = target.upstream_model
        upstream_payload["stream"] = True

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            remaining = timeout_seconds
            attempts = self._policy.max_attempts if self._policy is not None else 1
            for attempt in range(1, attempts + 1):
                admission = self._breaker.enter(model_id) if self._breaker is not None else None
                source: AsyncIterator[Mapping[str, Any]] | None = None
                completed = False
                emitted = False
                try:
                    source = target.provider.stream(
                        upstream_payload, model=target.upstream_model, timeout_seconds=remaining
                    )
                    async for chunk in source:
                        validated = _validate_chunk(chunk)
                        emitted = True
                        yield validated
                    if not emitted:
                        raise ProviderError(ProviderFailure.MALFORMED)
                    if self._breaker is not None and admission is not None:
                        self._breaker.finish(model_id, admission, None)
                    completed = True
                    return
                except ProviderError as error:
                    opened = (
                        self._breaker.finish(model_id, admission, error.failure)
                        if self._breaker is not None and admission is not None
                        else False
                    )
                    completed = True
                    next_timeout = (
                        None
                        if emitted
                        else await self._next_timeout(error, attempt, deadline, opened)
                    )
                    if next_timeout is None:
                        raise
                    remaining = next_timeout
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    if self._breaker is not None and admission is not None:
                        self._breaker.finish(model_id, admission, ProviderFailure.TIMEOUT)
                    completed = True
                    raise ProviderError(ProviderFailure.TIMEOUT) from exc
                except Exception as exc:
                    if self._breaker is not None and admission is not None:
                        self._breaker.finish(model_id, admission, ProviderFailure.FAILED)
                    completed = True
                    raise ProviderError(ProviderFailure.FAILED) from exc
                finally:
                    try:
                        if source is not None:
                            close = getattr(source, "aclose", None)
                            if callable(close):
                                with suppress(Exception):
                                    await close()
                    finally:
                        # Cancellation during transport cleanup must still
                        # release the single half-open probe admission.
                        if not completed and self._breaker is not None and admission is not None:
                            self._breaker.abandon(model_id, admission)
            raise AssertionError("bounded retry loop did not terminate")  # pragma: no cover

        return events()


@dataclass(frozen=True, slots=True)
class RoutedCompletion:
    decision: RouteDecision
    response: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RoutedStream:
    decision: RouteDecision
    chunks: AsyncIterator[Mapping[str, Any]]


class AsyncRoutingController:
    """Route once under all normal constraints, then await the selected client."""

    def __init__(self, router: Router, providers: AsyncProviderRegistry) -> None:
        self._router = router
        self._providers = providers

    async def complete(
        self, request: RouteRequest, payload: Mapping[str, Any], *, timeout_seconds: float
    ) -> RoutedCompletion:
        decision = self._router.route(request)
        response = await self._providers.complete(
            decision.selected_model, payload, timeout_seconds=timeout_seconds
        )
        return RoutedCompletion(decision, response)

    async def stream(
        self, request: RouteRequest, payload: Mapping[str, Any], *, timeout_seconds: float
    ) -> RoutedStream:
        decision = self._router.route(request)
        return RoutedStream(
            decision,
            self._providers.stream(
                decision.selected_model, payload, timeout_seconds=timeout_seconds
            ),
        )
