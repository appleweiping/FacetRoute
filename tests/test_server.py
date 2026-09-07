from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO

import pytest

from facetroute.errors import ConfigurationError
from facetroute.providers import (
    ProviderError,
    ProviderFailure,
    ProviderRegistry,
    ProviderTarget,
)
from facetroute.routers import RuleRouter
from facetroute.server import (
    FacetRouteHandler,
    chat_completion_from_http,
    create_server,
    route_request_from_http,
)
from facetroute.types import ModelCandidate, RouteRequest

# These talk to a local server running in a background thread, so the timeout
# guards against a hang rather than measuring anything. Two seconds was tight
# enough to fail intermittently under coverage instrumentation, which slows
# every call; a generous bound removes the flake without weakening a single
# assertion, and a genuine hang still fails the run.
SOCKET_TIMEOUT_SECONDS = 30


class RecordingProvider:
    def __init__(
        self,
        *,
        complete_failure: ProviderFailure | None = None,
        stream_failure: ProviderFailure | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, object], float]] = []
        self.complete_failure = complete_failure
        self.stream_failure = stream_failure

    def complete(self, payload, *, model, timeout_seconds):
        self.calls.append((model, dict(payload), timeout_seconds))
        if self.complete_failure is not None:
            raise ProviderError(self.complete_failure)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "fixture answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

    def stream(self, payload, *, model, timeout_seconds):
        self.calls.append((model, dict(payload), timeout_seconds))
        yield {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "fixture"}}],
        }
        if self.stream_failure is not None:
            raise ProviderError(self.stream_failure)
        yield {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


def _provider_registry(provider: RecordingProvider) -> ProviderRegistry:
    return ProviderRegistry(
        tuple(
            ProviderTarget(model_id, f"provider/{model_id}-v1", provider)
            for model_id in ("cheap", "balanced", "quality")
        )
    )


@contextmanager
def running_server(models, *, router=None, **options: object) -> Iterator[tuple[str, int]]:
    server = create_server(
        router or RuleRouter(models),
        models,
        host="127.0.0.1",
        port=0,
        **options,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=SOCKET_TIMEOUT_SECONDS)


def request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    body: object | bytes | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object], dict[str, str]]:
    connection = http.client.HTTPConnection(*address, timeout=SOCKET_TIMEOUT_SECONDS)
    data: bytes | None = (
        body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    )
    request_headers = dict(headers or {})
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=data, headers=request_headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, payload, response_headers


def request_sse(address: tuple[str, int], body: object) -> tuple[int, str, dict[str, str]]:
    connection = http.client.HTTPConnection(*address, timeout=SOCKET_TIMEOUT_SECONDS)
    data = json.dumps(body).encode()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=data,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    text = response.read().decode("utf-8")
    headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, text, headers


def test_chat_completion_routes_then_executes_injected_provider(three_models):
    provider = RecordingProvider()
    with running_server(three_models, provider_registry=_provider_registry(provider)) as address:
        status, body, headers = request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {
                "model": "facetroute",
                "messages": [{"role": "user", "content": "Use a tool to calculate 2+2"}],
                "tools": [{"type": "function", "function": {"name": "calculator"}}],
                "temperature": 0,
                "user": "customer-1",
                "facetroute": {"max_cost_usd": 1.0, "metadata": {"tenant": "test"}},
            },
        )

    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "fixture answer"
    assert headers["x-facetroute-model"] == "quality"
    assert headers["x-facetroute-policy"] == "rule"
    upstream_model, forwarded, timeout = provider.calls[0]
    assert upstream_model == "provider/quality-v1"
    assert timeout == 60.0
    assert forwarded["model"] == "facetroute"
    assert forwarded["user"] == "customer-1"
    assert "facetroute" not in forwarded


def test_chat_completion_stream_is_chunked_sse_with_terminal_marker(three_models):
    provider = RecordingProvider()
    with running_server(three_models, provider_registry=_provider_registry(provider)) as address:
        status, body, headers = request_sse(
            address,
            {
                "model": "facetroute",
                "messages": [{"role": "user", "content": "Calculate 2+2 with a tool"}],
                "tools": [{"type": "function"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["transfer-encoding"] == "chunked"
    assert body.count("chat.completion.chunk") == 2
    assert body.endswith("data: [DONE]\n\n")


def test_midstream_provider_failure_is_a_redacted_sse_error(three_models):
    provider = RecordingProvider(stream_failure=ProviderFailure.FAILED)
    with running_server(three_models, provider_registry=_provider_registry(provider)) as address:
        status, body, _ = request_sse(
            address,
            {
                "model": "facetroute",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [{"type": "function"}],
                "stream": True,
            },
        )

    assert status == 200
    assert '"code":"upstream_failed"' in body
    assert "[DONE]" not in body


@pytest.mark.parametrize(
    "failure, expected_status",
    [
        (ProviderFailure.REJECTED, 502),
        (ProviderFailure.MALFORMED, 502),
        (ProviderFailure.RATE_LIMITED, 503),
        (ProviderFailure.UNAVAILABLE, 503),
        (ProviderFailure.TIMEOUT, 504),
    ],
)
def test_provider_failures_are_redacted_and_mapped(failure, expected_status, three_models):
    provider = RecordingProvider(complete_failure=failure)
    with running_server(three_models, provider_registry=_provider_registry(provider)) as address:
        status, body, _ = request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {
                "model": "facetroute",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [{"type": "function"}],
            },
        )
    assert status == expected_status
    assert body["error"]["code"] == failure.value
    assert "Traceback" not in body["error"]["message"]


def test_stream_failure_before_first_event_returns_http_error(three_models):
    class FailingStreamProvider(RecordingProvider):
        def stream(self, payload, *, model, timeout_seconds):
            self.calls.append((model, dict(payload), timeout_seconds))
            yield from ()
            raise ProviderError(ProviderFailure.TIMEOUT)

    provider = FailingStreamProvider()
    with running_server(three_models, provider_registry=_provider_registry(provider)) as address:
        status, body, _ = request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {
                "model": "facetroute",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [{"type": "function"}],
                "stream": True,
            },
        )
    assert status == 504
    assert body["error"]["code"] == "upstream_timeout"


def test_proxy_disabled_returns_structured_503(three_models):
    body = {
        "model": "facetroute",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{"type": "function"}],
    }
    with running_server(three_models) as address:
        status, payload, _ = request_json(address, "POST", "/v1/chat/completions", body)
    assert status == 503
    assert payload["error"]["code"] == "provider_not_configured"


def test_chat_completion_parser_rejects_ambiguous_or_unsafe_shapes():
    valid = {
        "model": "facetroute",
        "messages": [{"role": "user", "content": "hello"}],
    }
    request, forwarded, stream = chat_completion_from_http(valid, request_id="request-1")
    assert request.request_id == "request-1"
    assert forwarded["stream"] is False
    assert not stream

    invalid = (
        {**valid, "model": "quality"},
        {**valid, "stream": "yes"},
        {**valid, "temperature": float("nan")},
        {**valid, "n": True},
        {**valid, "max_tokens": 1, "max_completion_tokens": 1},
        {**valid, "top_logprobs": 2},
        {**valid, "stream_options": {"include_usage": True}},
        {**valid, "stop": ["a", "b", "c", "d", "e"]},
        {**valid, "logit_bias": {"1": True}},
        {**valid, "tools": [{}] * 129},
        {**valid, "tool_choice": True},
        {**valid, "messages": [{"role": "user", "content": "x", "name": 4}]},
        {**valid, "facetroute": {"provider_url": "http://attacker.invalid"}},
    )
    for payload in invalid:
        with pytest.raises(ConfigurationError):
            chat_completion_from_http(payload, request_id="request-1")


def test_health_models_and_native_route(three_models):
    with running_server(three_models) as address:
        health, health_body, _ = request_json(address, "GET", "/health")
        models_status, models_body, _ = request_json(address, "GET", "/v1/models")
        route_status, route_body, headers = request_json(
            address,
            "POST",
            "/v1/route",
            {"query": "Write a parser", "user_id": "u", "request_id": "native-1"},
            headers={"X-Request-ID": "edge-request"},
        )

    assert health == 200
    assert health_body == {"status": "ok"}
    assert models_status == 200
    assert len(models_body["data"]) == 3
    assert route_status == 200
    assert route_body["object"] == "routing.decision"
    assert route_body["input_dialect"] == "native"
    assert route_body["decision"]["request_id"] == "native-1"
    assert headers["cache-control"] == "no-store"


def test_openai_request_shaped_parser_routes_but_does_not_claim_completion(three_models):
    with running_server(three_models) as address:
        status, body, _ = request_json(
            address,
            "POST",
            "/v1/route",
            {
                "messages": [
                    {"role": "system", "content": "Use tools carefully"},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Find the answer"}],
                    },
                ],
                "tools": [{"type": "function"}],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 40,
            },
        )

    assert status == 200
    assert body["input_dialect"] == "openai-chat-request"
    assert body["model"] == "quality"
    assert "choices" not in body


def test_bearer_authentication_and_request_id_sanitization(three_models):
    with running_server(three_models, bearer_token="secret-token") as address:
        unauthorized, body, headers = request_json(address, "GET", "/v1/models")
        authorized, _, safe_headers = request_json(
            address,
            "GET",
            "/v1/models",
            headers={
                "Authorization": "Bearer secret-token",
                "X-Request-ID": "bad\tidentifier",
            },
        )

    assert unauthorized == 401
    assert body["error"]["code"] == "unauthorized"
    assert headers["www-authenticate"] == "Bearer"
    assert authorized == 200
    assert safe_headers["x-request-id"] != "bad\tidentifier"


def test_response_boundary_encodes_all_unsafe_header_characters():
    handler = object.__new__(FacetRouteHandler)
    emitted_headers: list[tuple[str, str]] = []
    handler.close_connection = False
    handler.wfile = BytesIO()
    handler.send_response = lambda _status: None  # type: ignore[method-assign]
    handler.send_header = (  # type: ignore[method-assign]
        lambda name, value: emitted_headers.append((name, value))
    )
    handler.end_headers = lambda: None  # type: ignore[method-assign]

    handler._json_response(
        HTTPStatus.OK,
        {"status": "ok"},
        "trusted\r\nX-Injected: yes\n",
        extra_headers={"X-Test": "model\r\nInjected: 是"},
    )

    assert ("X-Request-ID", "trusted%0D%0AX-Injected%3A%20yes%0A") in emitted_headers
    assert ("X-Test", "model%0D%0AInjected%3A%20%E6%98%AF") in emitted_headers
    assert all("\r" not in value and "\n" not in value for _, value in emitted_headers)


@pytest.mark.parametrize(
    "body, headers, expected, code",
    [
        (b'{"query":"a","query":"b"}', {}, 400, "invalid_json"),
        (b'{"query":NaN}', {}, 400, "invalid_json"),
        (b"[]", {}, 400, "invalid_json"),
        (b'{"query":"x"}', {"Content-Type": "text/plain"}, 415, "unsupported_media_type"),
        (b'{"query":"x","unknown":1}', {}, 422, "invalid_request"),
        (b'{"query":"x","required_capabilities":["vision"]}', {}, 422, "invalid_request"),
    ],
)
def test_http_errors_are_structured_and_do_not_expose_tracebacks(
    three_models, body, headers, expected, code
):
    with running_server(three_models) as address:
        status, payload, _ = request_json(address, "POST", "/v1/route", body, headers=headers)

    assert status == expected
    assert payload["error"]["code"] == code
    assert "Traceback" not in payload["error"]["message"]


def test_body_limit_and_unknown_endpoint(three_models):
    with running_server(three_models, max_body_bytes=12) as address:
        too_large, body, too_large_headers = request_json(
            address, "POST", "/v1/route", {"query": "this is too long"}
        )
        missing, missing_body, _ = request_json(address, "GET", "/absent")
        missing_post, _, missing_post_headers = request_json(
            address, "POST", "/absent", {"query": "x"}
        )

    assert too_large == 413
    assert body["error"]["code"] == "payload_too_large"
    assert missing == missing_post == 404
    assert missing_body["error"]["code"] == "not_found"
    assert too_large_headers["connection"] == "close"
    assert missing_post_headers["connection"] == "close"


def test_transfer_encoding_missing_and_invalid_lengths_are_rejected(three_models):
    with running_server(three_models) as address:
        transfer, transfer_body, _ = request_json(
            address,
            "POST",
            "/v1/route",
            b'{"query":"x"}',
            headers={"Transfer-Encoding": "chunked"},
        )
        connection = http.client.HTTPConnection(*address, timeout=SOCKET_TIMEOUT_SECONDS)
        connection.putrequest("POST", "/v1/route")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        missing_response = connection.getresponse()
        missing_body = json.loads(missing_response.read())
        connection.close()

        raw = socket.create_connection(address, timeout=SOCKET_TIMEOUT_SECONDS)
        raw.sendall(
            b"POST /v1/route HTTP/1.1\r\nHost: local\r\n"
            b"Content-Type: application/json\r\nContent-Length: nope\r\n"
            b"Connection: close\r\n\r\n"
        )
        invalid_chunks: list[bytes] = []
        while chunk := raw.recv(4096):
            invalid_chunks.append(chunk)
        invalid_response = b"".join(invalid_chunks)
        raw.close()

        duplicate = socket.create_connection(address, timeout=SOCKET_TIMEOUT_SECONDS)
        duplicate.sendall(
            b"POST /v1/route HTTP/1.1\r\nHost: local\r\n"
            b"Content-Type: application/json\r\nContent-Length: 13\r\n"
            b"Content-Length: 13\r\nConnection: close\r\n\r\n"
            b'{"query":"x"}'
        )
        duplicate_chunks: list[bytes] = []
        while chunk := duplicate.recv(4096):
            duplicate_chunks.append(chunk)
        duplicate_response = b"".join(duplicate_chunks)
        duplicate.close()

    assert transfer == 400
    assert transfer_body["error"]["code"] == "unsupported_transfer_encoding"
    assert missing_response.status == 411
    assert missing_body["error"]["code"] == "length_required"
    assert b"400 Bad Request" in invalid_response
    assert b"invalid_content_length" in invalid_response
    assert b"400 Bad Request" in duplicate_response
    assert b"ambiguous_content_length" in duplicate_response


def test_capacity_limit_returns_structured_503(three_models):
    server = create_server(
        RuleRouter(three_models), three_models, host="127.0.0.1", port=0, max_concurrency=1
    )
    assert server._capacity.acquire(blocking=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = request_json(server.server_address, "GET", "/health")
    finally:
        server._capacity.release()
        server.shutdown()
        server.server_close()
        thread.join(timeout=SOCKET_TIMEOUT_SECONDS)

    assert status == 503
    assert body["error"]["code"] == "server_busy"


def test_unexpected_router_error_is_redacted(three_models):
    class BrokenRouter:
        def route(self, _request: RouteRequest):
            raise RuntimeError("database secret detail")

    with running_server(three_models, router=BrokenRouter()) as address:
        status, body, _ = request_json(address, "POST", "/v1/route", {"query": "hello"})

    assert status == 500
    assert body["error"]["message"] == "routing failed unexpectedly"
    assert "secret" not in json.dumps(body)


def test_request_shaped_parser_validation():
    native, dialect = route_request_from_http({"query": "hello"}, request_id="r")
    compat, compat_dialect = route_request_from_http(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 12,
            "response_format": {"type": "text"},
        },
        request_id="c",
    )

    assert dialect == "native"
    assert native.request_id == "r"
    assert compat_dialect == "openai-chat-request"
    assert compat.expected_output_tokens == 12
    assert not compat.needs_json
    with pytest.raises(ConfigurationError, match="at most 128"):
        route_request_from_http(
            {"messages": [{"role": "user", "content": "x"}] * 129},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="unsupported"):
        route_request_from_http(
            {"messages": [{"role": "developer", "content": "x"}]},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="text part"):
        route_request_from_http(
            {"messages": [{"role": "user", "content": [{"type": "image", "url": "x"}]}]},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="unknown messages"):
        route_request_from_http(
            {"messages": [{"role": "user", "content": "x", "extra": True}]},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="tools must be an array"):
        route_request_from_http(
            {"messages": [{"role": "user", "content": "x"}], "tools": {}},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="response_format"):
        route_request_from_http(
            {
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "binary"},
            },
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="no text"):
        route_request_from_http(
            {"messages": [{"role": "assistant", "content": "   "}]},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="expected_output_tokens must be an integer"):
        route_request_from_http(
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": True},
            request_id="x",
        )
    with pytest.raises(ConfigurationError, match="max_cost_usd must be a number"):
        route_request_from_http({"query": "x", "max_cost_usd": "cheap"}, request_id="x")


def test_server_configuration_validation(three_models: tuple[ModelCandidate, ...]):
    router = RuleRouter(three_models)
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_server(router, three_models, max_body_bytes=0)
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_server(router, three_models, max_body_bytes=True)
    with pytest.raises(ValueError, match="max_concurrency"):
        create_server(router, three_models, max_concurrency=0)
    with pytest.raises(ValueError, match="request_timeout"):
        create_server(router, three_models, request_timeout_seconds=0)
    with pytest.raises(ValueError, match="request_timeout"):
        create_server(router, three_models, request_timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="bearer_token"):
        create_server(router, three_models, bearer_token="")
    with pytest.raises(ValueError, match="port"):
        create_server(router, three_models, port=70_000)
    with pytest.raises(ValueError, match="port"):
        create_server(router, three_models, port=True)
    provider = RecordingProvider()
    incomplete = ProviderRegistry((ProviderTarget("quality", "upstream", provider),))
    with pytest.raises(ValueError, match="missing enabled catalog models"):
        create_server(router, three_models, provider_registry=incomplete)
    unknown = ProviderRegistry((ProviderTarget("unknown", "upstream", provider),))
    with pytest.raises(ValueError, match="unknown catalog models"):
        create_server(router, three_models, provider_registry=unknown)
