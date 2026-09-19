from __future__ import annotations

import asyncio
import http.client
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from facetroute import (
    AsyncProviderRegistry,
    AsyncProviderTarget,
    AsyncRoutingController,
    ModelCandidate,
    RouteRequest,
    RuleRouter,
)
from facetroute.cli import main
from facetroute.errors import ConfigurationError
from facetroute.providers import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderFailure,
    ProviderRegistry,
    ProviderTarget,
)
from facetroute.resilience import ProviderResiliencePolicy, ResilientProviderRegistry


def _completion() -> dict[str, Any]:
    return {"id": "c", "object": "chat.completion", "model": "upstream", "choices": [{}]}


def _chunk() -> dict[str, Any]:
    return {"id": "c", "object": "chat.completion.chunk", "model": "upstream", "choices": []}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)
        await asyncio.sleep(0)


class ScriptedProvider:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[float] = []

    def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        assert model == "upstream"
        assert payload == {"messages": []}
        self.calls.append(timeout_seconds)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome

    def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Iterator[Mapping[str, Any]]:
        assert model == "upstream"
        assert payload == {"messages": []}
        self.calls.append(timeout_seconds)
        outcome = self.outcomes.pop(0)
        assert isinstance(outcome, list)

        def chunks() -> Iterator[Mapping[str, Any]]:
            for item in outcome:
                if isinstance(item, Exception):
                    raise item
                assert isinstance(item, dict)
                yield item

        return chunks()


def _registry(provider: ScriptedProvider) -> ProviderRegistry:
    return ProviderRegistry((ProviderTarget("routed", "upstream", provider),))


def test_default_does_not_retry_even_certified_pre_send_failure() -> None:
    provider = ScriptedProvider([ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True)])
    with pytest.raises(ProviderError) as error:
        _registry(provider).complete("routed", {"messages": []}, timeout_seconds=2)
    assert error.value.retry_safe
    assert len(provider.calls) == 1


def test_safe_retry_backoff_uses_remaining_deadline_and_never_retries_ambiguous() -> None:
    clock = FakeClock()
    provider = ScriptedProvider(
        [ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True), _completion()]
    )
    registry = ResilientProviderRegistry(
        _registry(provider),
        ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.5),
        clock=clock,
        sleep=clock.sleep,
    )
    assert registry.complete("routed", {"messages": []}, timeout_seconds=2) == _completion()
    assert provider.calls == [2, 1.5]
    assert clock.sleeps == [0.5]

    ambiguous = ScriptedProvider([ProviderError(ProviderFailure.TIMEOUT), _completion()])
    no_duplicate = ResilientProviderRegistry(
        _registry(ambiguous), ProviderResiliencePolicy(max_attempts=5)
    )
    with pytest.raises(ProviderError) as error:
        no_duplicate.complete("routed", {"messages": []}, timeout_seconds=2)
    assert error.value.failure is ProviderFailure.TIMEOUT
    assert len(ambiguous.calls) == 1


def test_backoff_cannot_consume_deadline_then_send_duplicate() -> None:
    clock = FakeClock()
    provider = ScriptedProvider(
        [ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True), _completion()]
    )
    registry = ResilientProviderRegistry(
        _registry(provider),
        ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.5),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(ProviderError):
        registry.complete("routed", {"messages": []}, timeout_seconds=0.5)
    assert len(provider.calls) == 1
    assert clock.sleeps == []


def test_circuit_opens_per_model_and_half_open_admits_one_concurrent_probe() -> None:
    clock = FakeClock()
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(ScriptedProvider):
        def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            if self.calls:
                started.set()
                assert release.wait(timeout=5)
            return super().complete(payload, model=model, timeout_seconds=timeout_seconds)

    provider = BlockingProvider(
        [ProviderError(ProviderFailure.UNAVAILABLE), _completion(), _completion()]
    )
    registry = ResilientProviderRegistry(
        _registry(provider),
        ProviderResiliencePolicy(failure_threshold=1, recovery_seconds=5),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(ProviderError):
        registry.complete("routed", {"messages": []}, timeout_seconds=2)
    with pytest.raises(ProviderError) as open_error:
        registry.complete("routed", {"messages": []}, timeout_seconds=2)
    assert open_error.value.failure is ProviderFailure.UNAVAILABLE
    assert len(provider.calls) == 1
    clock.now = 5
    with ThreadPoolExecutor(max_workers=1) as executor:
        probe = executor.submit(registry.complete, "routed", {"messages": []}, timeout_seconds=2)
        assert started.wait(timeout=5)
        with pytest.raises(ProviderError):
            registry.complete("routed", {"messages": []}, timeout_seconds=2)
        release.set()
        assert probe.result(timeout=5) == _completion()
    assert registry.complete("routed", {"messages": []}, timeout_seconds=2) == _completion()
    assert len(provider.calls) == 3


def test_open_circuit_does_not_block_another_selected_model() -> None:
    bad = ScriptedProvider([ProviderError(ProviderFailure.UNAVAILABLE)])
    good = ScriptedProvider([_completion()])
    registry = ResilientProviderRegistry(
        ProviderRegistry(
            (
                ProviderTarget("bad", "upstream", bad),
                ProviderTarget("good", "upstream", good),
            )
        ),
        ProviderResiliencePolicy(failure_threshold=1),
    )
    with pytest.raises(ProviderError):
        registry.complete("bad", {"messages": []}, timeout_seconds=2)
    with pytest.raises(ProviderError):
        registry.complete("bad", {"messages": []}, timeout_seconds=2)
    assert registry.complete("good", {"messages": []}, timeout_seconds=2) == _completion()
    assert len(bad.calls) == len(good.calls) == 1


def test_stream_retries_only_before_first_chunk_and_never_after_partial_sse() -> None:
    clock = FakeClock()
    safe = ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True)
    provider = ScriptedProvider([[safe], [_chunk()]])
    registry = ResilientProviderRegistry(
        _registry(provider),
        ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.1),
        clock=clock,
        sleep=clock.sleep,
    )
    assert list(registry.stream("routed", {"messages": []}, timeout_seconds=2)) == [_chunk()]
    assert len(provider.calls) == 2

    partial = ScriptedProvider([[_chunk(), safe], [_chunk()]])
    no_duplicate = ResilientProviderRegistry(
        _registry(partial), ProviderResiliencePolicy(max_attempts=2)
    )
    chunks = no_duplicate.stream("routed", {"messages": []}, timeout_seconds=2)
    assert next(chunks) == _chunk()
    with pytest.raises(ProviderError):
        next(chunks)
    assert len(partial.calls) == 1

    empty = ScriptedProvider([[]])
    malformed = ResilientProviderRegistry(
        _registry(empty), ProviderResiliencePolicy(failure_threshold=1)
    )
    with pytest.raises(ProviderError) as error:
        list(malformed.stream("routed", {"messages": []}, timeout_seconds=2))
    assert error.value.failure is ProviderFailure.MALFORMED
    with pytest.raises(ProviderError) as open_error:
        list(malformed.stream("routed", {"messages": []}, timeout_seconds=2))
    assert open_error.value.failure is ProviderFailure.UNAVAILABLE
    assert len(empty.calls) == 1


def test_caller_abandoned_stream_is_neutral_but_abandoned_probe_reopens() -> None:
    ordinary = ScriptedProvider([[_chunk()], _completion()])
    registry = ResilientProviderRegistry(
        _registry(ordinary), ProviderResiliencePolicy(failure_threshold=1)
    )
    chunks = registry.stream("routed", {"messages": []}, timeout_seconds=2)
    assert next(chunks) == _chunk()
    chunks.close()
    assert registry.complete("routed", {"messages": []}, timeout_seconds=2) == _completion()

    clock = FakeClock()
    probe_provider = ScriptedProvider(
        [ProviderError(ProviderFailure.UNAVAILABLE), [_chunk()], _completion()]
    )
    probe_registry = ResilientProviderRegistry(
        _registry(probe_provider),
        ProviderResiliencePolicy(failure_threshold=1, recovery_seconds=5),
        clock=clock,
    )
    with pytest.raises(ProviderError):
        probe_registry.complete("routed", {"messages": []}, timeout_seconds=2)
    clock.now = 5
    probe = probe_registry.stream("routed", {"messages": []}, timeout_seconds=2)
    assert next(probe) == _chunk()
    probe.close()
    with pytest.raises(ProviderError) as still_open:
        probe_registry.complete("routed", {"messages": []}, timeout_seconds=2)
    assert still_open.value.failure is ProviderFailure.UNAVAILABLE
    assert len(probe_provider.calls) == 2


def test_sync_probe_release_survives_transport_close_base_exception() -> None:
    class CloseAborted(BaseException):
        pass

    class CloseAbortingSource:
        def __iter__(self) -> CloseAbortingSource:
            return self

        def __next__(self) -> Mapping[str, Any]:
            return _chunk()

        def close(self) -> None:
            raise CloseAborted

    class CloseAbortingProvider(ScriptedProvider):
        def stream(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Iterator[Mapping[str, Any]]:
            assert model == "upstream"
            self.calls.append(timeout_seconds)
            return CloseAbortingSource()

    clock = FakeClock()
    provider = CloseAbortingProvider([ProviderError(ProviderFailure.UNAVAILABLE)])
    registry = ResilientProviderRegistry(
        _registry(provider),
        ProviderResiliencePolicy(failure_threshold=1, recovery_seconds=5),
        clock=clock,
    )
    with pytest.raises(ProviderError):
        registry.complete("routed", {"messages": []}, timeout_seconds=2)
    clock.now = 5
    probe = registry.stream("routed", {"messages": []}, timeout_seconds=2)
    assert next(probe) == _chunk()
    with pytest.raises(CloseAborted):
        probe.close()
    with pytest.raises(ProviderError) as still_open:
        registry.complete("routed", {"messages": []}, timeout_seconds=2)
    assert still_open.value.failure is ProviderFailure.UNAVAILABLE
    assert len(provider.calls) == 2


def test_transport_certifies_only_connect_phase_as_retry_safe() -> None:
    provider = OpenAICompatibleProvider("http://127.0.0.1:1/v1")
    with (
        patch.object(http.client.HTTPConnection, "connect", side_effect=TimeoutError),
        pytest.raises(ProviderError) as error,
    ):
        provider.complete({"messages": []}, model="upstream", timeout_seconds=1)
    assert error.value.failure is ProviderFailure.TIMEOUT
    assert error.value.retry_safe
    with (
        patch.object(http.client.HTTPConnection, "connect"),
        patch.object(http.client.HTTPConnection, "request", side_effect=TimeoutError),
        pytest.raises(ProviderError) as post_send,
    ):
        provider.complete({"messages": []}, model="upstream", timeout_seconds=1)
    assert post_send.value.failure is ProviderFailure.TIMEOUT
    assert not post_send.value.retry_safe


class AsyncFakeProvider:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls += 1
        await asyncio.sleep(0)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome

    async def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        assert isinstance(outcome, list)
        for item in outcome:
            await asyncio.sleep(0)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, dict)
            yield item


def test_async_payload_uses_selected_model_and_stream_mode_without_mutating_caller() -> None:
    class ForwardingProvider(AsyncFakeProvider):
        def __init__(self) -> None:
            super().__init__([_completion(), [_chunk()]])
            self.forwarded: list[dict[str, Any]] = []

        async def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            self.forwarded.append(dict(payload))
            assert payload["model"] == model == "upstream"
            assert payload["stream"] is False
            return await super().complete(payload, model=model, timeout_seconds=timeout_seconds)

        async def stream(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> AsyncIterator[Mapping[str, Any]]:
            self.forwarded.append(dict(payload))
            assert payload["model"] == model == "upstream"
            assert payload["stream"] is True
            async for chunk in super().stream(
                payload, model=model, timeout_seconds=timeout_seconds
            ):
                yield chunk

    async def scenario() -> None:
        provider = ForwardingProvider()
        registry = AsyncProviderRegistry((AsyncProviderTarget("routed", "upstream", provider),))
        caller_payload = {"messages": [], "model": "forbidden", "stream": "wrong"}
        assert await registry.complete("routed", caller_payload, timeout_seconds=2) == _completion()
        assert [
            chunk async for chunk in registry.stream("routed", caller_payload, timeout_seconds=2)
        ] == [_chunk()]
        assert provider.forwarded == [
            {"messages": [], "model": "upstream", "stream": False},
            {"messages": [], "model": "upstream", "stream": True},
        ]
        assert caller_payload == {"messages": [], "model": "forbidden", "stream": "wrong"}

    asyncio.run(scenario())


def test_async_controller_respects_hard_constraints_and_safe_retries(
    make_model: Any,
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        eligible: ModelCandidate = make_model("eligible", regions=frozenset({"eu"}))
        other: ModelCandidate = make_model("other", regions=frozenset({"us"}))
        router = RuleRouter((eligible, other), {}, ())
        provider = AsyncFakeProvider(
            [ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True), _completion()]
        )
        other_provider = AsyncFakeProvider([_completion()])
        registry = AsyncProviderRegistry(
            (
                AsyncProviderTarget("eligible", "upstream", provider),
                AsyncProviderTarget("other", "upstream", other_provider),
            ),
            policy=ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.1),
            clock=clock,
            sleep=clock.asleep,
        )
        controller = AsyncRoutingController(router, registry)
        result = await controller.complete(
            RouteRequest(query="hello", region="eu", request_id="r"),
            {"messages": []},
            timeout_seconds=2,
        )
        assert result.decision.selected_model == "eligible"
        assert result.response == _completion()
        assert provider.calls == 2
        assert other_provider.calls == 0

    asyncio.run(scenario())


def test_async_stream_partial_failure_is_not_retried() -> None:
    async def scenario() -> None:
        safe = ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True)
        provider = AsyncFakeProvider([[_chunk(), safe], [_chunk()]])
        registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", provider),),
            policy=ProviderResiliencePolicy(max_attempts=2),
        )
        chunks = registry.stream("routed", {"messages": []}, timeout_seconds=2)
        assert await anext(chunks) == _chunk()
        with pytest.raises(ProviderError):
            await anext(chunks)
        assert provider.calls == 1

    asyncio.run(scenario())


def test_async_client_rejects_unsafe_registry_and_timeout_inputs() -> None:
    provider = AsyncFakeProvider([_completion()])
    target = AsyncProviderTarget("routed", "upstream", provider)
    with pytest.raises(ConfigurationError):
        AsyncProviderTarget("", "upstream", provider)
    with pytest.raises(ConfigurationError):
        AsyncProviderTarget("routed", "", provider)
    with pytest.raises(ConfigurationError):
        AsyncProviderRegistry(())
    with pytest.raises(ConfigurationError):
        AsyncProviderRegistry((target, target))
    with pytest.raises(ConfigurationError):
        AsyncProviderRegistry((target,) * 1_025)
    with pytest.raises(ConfigurationError):
        AsyncProviderRegistry((target,), policy="not a policy")  # type: ignore[arg-type]
    registry = AsyncProviderRegistry((target,))
    assert registry.model_ids == frozenset({"routed"})

    async def scenario() -> None:
        with pytest.raises(ProviderError) as unknown:
            await registry.complete("missing", {}, timeout_seconds=1)
        assert unknown.value.failure is ProviderFailure.NOT_CONFIGURED
        with pytest.raises(ConfigurationError):
            await registry.complete("routed", {}, timeout_seconds=0)
        with pytest.raises(ConfigurationError):
            registry.stream("routed", {}, timeout_seconds=float("nan"))
        assert provider.calls == 0

    asyncio.run(scenario())


def test_async_default_has_no_retry_and_deadline_prevents_duplicate() -> None:
    async def scenario() -> None:
        safe = ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True)
        default_provider = AsyncFakeProvider([safe, _completion()])
        default_registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", default_provider),)
        )
        with pytest.raises(ProviderError):
            await default_registry.complete("routed", {}, timeout_seconds=1)
        assert default_provider.calls == 1

        clock = FakeClock()
        bounded_provider = AsyncFakeProvider([safe, _completion()])
        bounded_registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", bounded_provider),),
            policy=ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.5),
            clock=clock,
            sleep=clock.asleep,
        )
        with pytest.raises(ProviderError):
            await bounded_registry.complete("routed", {}, timeout_seconds=0.5)
        assert bounded_provider.calls == 1
        assert clock.sleeps == []

    asyncio.run(scenario())


def test_async_client_redacts_ambiguous_failures_and_validates_payloads() -> None:
    async def scenario() -> None:
        for outcome, failure in (
            (TimeoutError("secret"), ProviderFailure.TIMEOUT),
            (RuntimeError("secret"), ProviderFailure.FAILED),
            ({"object": "wrong"}, ProviderFailure.MALFORMED),
        ):
            provider = AsyncFakeProvider([outcome, _completion()])
            registry = AsyncProviderRegistry(
                (AsyncProviderTarget("routed", "upstream", provider),),
                policy=ProviderResiliencePolicy(max_attempts=2),
            )
            with pytest.raises(ProviderError) as error:
                await registry.complete("routed", {}, timeout_seconds=2)
            assert error.value.failure is failure
            assert "secret" not in str(error.value)
            assert provider.calls == 1

    asyncio.run(scenario())


def test_async_stream_success_prechunk_retry_and_ambiguous_timeout() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        safe = ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=True)
        provider = AsyncFakeProvider([[safe], [_chunk()]])
        registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", provider),),
            policy=ProviderResiliencePolicy(max_attempts=2, initial_backoff_seconds=0.1),
            clock=clock,
            sleep=clock.asleep,
        )
        assert [chunk async for chunk in registry.stream("routed", {}, timeout_seconds=2)] == [
            _chunk()
        ]
        assert provider.calls == 2
        assert clock.sleeps == [0.1]

        ambiguous = AsyncFakeProvider([[TimeoutError("secret")], [_chunk()]])
        no_duplicate = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", ambiguous),),
            policy=ProviderResiliencePolicy(max_attempts=2),
        )
        with pytest.raises(ProviderError) as error:
            _ = [chunk async for chunk in no_duplicate.stream("routed", {}, timeout_seconds=2)]
        assert error.value.failure is ProviderFailure.TIMEOUT
        assert ambiguous.calls == 1

        empty = AsyncFakeProvider([[]])
        empty_registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", empty),),
            policy=ProviderResiliencePolicy(failure_threshold=1),
        )
        with pytest.raises(ProviderError) as malformed:
            _ = [chunk async for chunk in empty_registry.stream("routed", {}, timeout_seconds=2)]
        assert malformed.value.failure is ProviderFailure.MALFORMED
        with pytest.raises(ProviderError) as open_error:
            _ = [chunk async for chunk in empty_registry.stream("routed", {}, timeout_seconds=2)]
        assert open_error.value.failure is ProviderFailure.UNAVAILABLE
        assert empty.calls == 1

    asyncio.run(scenario())


def test_async_abandoned_stream_is_neutral_but_abandoned_probe_reopens() -> None:
    async def scenario() -> None:
        ordinary = AsyncFakeProvider([[_chunk()], _completion()])
        registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", ordinary),),
            policy=ProviderResiliencePolicy(failure_threshold=1),
        )
        chunks = registry.stream("routed", {}, timeout_seconds=2)
        assert await anext(chunks) == _chunk()
        await chunks.aclose()
        assert await registry.complete("routed", {}, timeout_seconds=2) == _completion()

        clock = FakeClock()
        probe_provider = AsyncFakeProvider(
            [ProviderError(ProviderFailure.UNAVAILABLE), [_chunk()], _completion()]
        )
        probe_registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", probe_provider),),
            policy=ProviderResiliencePolicy(failure_threshold=1, recovery_seconds=5),
            clock=clock,
            sleep=clock.asleep,
        )
        with pytest.raises(ProviderError):
            await probe_registry.complete("routed", {}, timeout_seconds=2)
        clock.now = 5
        probe = probe_registry.stream("routed", {}, timeout_seconds=2)
        assert await anext(probe) == _chunk()
        await probe.aclose()
        with pytest.raises(ProviderError) as still_open:
            await probe_registry.complete("routed", {}, timeout_seconds=2)
        assert still_open.value.failure is ProviderFailure.UNAVAILABLE
        assert probe_provider.calls == 2

    asyncio.run(scenario())


def test_async_probe_release_survives_cancellation_during_source_close() -> None:
    async def scenario() -> None:
        close_started = asyncio.Event()

        class BlockingCloseSource:
            def __aiter__(self) -> BlockingCloseSource:
                return self

            async def __anext__(self) -> Mapping[str, Any]:
                return _chunk()

            async def aclose(self) -> None:
                close_started.set()
                await asyncio.Future()

        class BlockingCloseProvider(AsyncFakeProvider):
            def stream(
                self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
            ) -> AsyncIterator[Mapping[str, Any]]:
                assert model == "upstream"
                self.calls += 1
                return BlockingCloseSource()

        clock = FakeClock()
        provider = BlockingCloseProvider([ProviderError(ProviderFailure.UNAVAILABLE)])
        registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", provider),),
            policy=ProviderResiliencePolicy(failure_threshold=1, recovery_seconds=5),
            clock=clock,
        )
        with pytest.raises(ProviderError):
            await registry.complete("routed", {}, timeout_seconds=2)
        clock.now = 5
        probe = registry.stream("routed", {}, timeout_seconds=2)
        assert await anext(probe) == _chunk()
        closer = asyncio.create_task(probe.aclose())
        await asyncio.wait_for(close_started.wait(), timeout=2)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        with pytest.raises(ProviderError) as still_open:
            await registry.complete("routed", {}, timeout_seconds=2)
        assert still_open.value.failure is ProviderFailure.UNAVAILABLE
        assert provider.calls == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 0},
        {"max_attempts": 6},
        {"initial_backoff_seconds": float("inf")},
        {"max_backoff_seconds": -1},
        {"failure_threshold": 0},
        {"recovery_seconds": 0},
    ],
)
def test_policy_budgets_fail_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ProviderResiliencePolicy(**changes)  # type: ignore[arg-type]


def test_sync_registry_circuit_state_cannot_exceed_binding_budget() -> None:
    provider = ScriptedProvider([])
    target = ProviderTarget("routed", "upstream", provider)
    with pytest.raises(ConfigurationError, match="1024"):
        ProviderRegistry((target,) * 1_025)


def test_cli_resilience_flags_require_explicit_provider_opt_in(capsys: Any) -> None:
    models = Path(__file__).resolve().parents[1] / "examples" / "models.json"
    assert main(["serve", "--models", str(models), "--provider-retry-attempts", "2"]) == 2
    assert "require --provider-resilience" in capsys.readouterr().err
    assert main(["serve", "--models", str(models), "--provider-resilience"]) == 2
    assert "requires --providers" in capsys.readouterr().err
