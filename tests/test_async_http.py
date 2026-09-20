from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
import pytest

from facetroute import RouteRequest, RuleRouter
from facetroute.async_client import (
    AsyncProviderRegistry,
    AsyncProviderTarget,
    AsyncRoutingController,
)
from facetroute.async_http import AsyncOpenAICompatibleProvider
from facetroute.errors import ConfigurationError
from facetroute.providers import ProviderError, ProviderFailure


def completion(model: str = "upstream") -> dict[str, Any]:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "four"}}],
    }


def chunk(model: str = "upstream") -> dict[str, Any]:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "four"}}],
    }


class LocalServer(ThreadingHTTPServer):
    calls: list[dict[str, Any]]
    status: int
    content_type: str | None
    content_encoding: str | None
    body_override: bytes | None
    fragments: tuple[bytes, ...] | None
    delay_headers: float
    hold_headers: bool
    headers_release: threading.Event
    received: threading.Event
    hold_stream: bool
    closed: threading.Event


class LocalHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        server = cast(LocalServer, self.server)
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        server.calls.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "accept_encoding": self.headers.get("Accept-Encoding"),
                "payload": request,
            }
        )
        server.received.set()
        if server.hold_headers:
            server.headers_release.wait(timeout=20)
        if server.delay_headers:
            time.sleep(server.delay_headers)
        if server.status != 200:
            self.send_response(server.status)
            if server.status in {301, 302, 307, 308}:
                self.send_header("Location", "https://untrusted.invalid/v1/chat/completions")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if server.hold_stream:
            first = b"data: " + json.dumps(chunk()).encode() + b"\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            self.connection.settimeout(3)
            try:
                if self.connection.recv(1) == b"":
                    server.closed.set()
            except OSError:
                pass
            return
        body = server.body_override
        if body is None:
            if request["stream"]:
                body = (
                    b"data: "
                    + json.dumps(chunk(request["model"])).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
            else:
                body = json.dumps(completion(request["model"])).encode()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            server.content_type
            or ("text/event-stream" if request["stream"] else "application/json"),
        )
        if server.content_encoding is not None:
            self.send_header("Content-Encoding", server.content_encoding)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if server.fragments is None:
                self.wfile.write(body)
            else:
                for fragment in server.fragments:
                    self.wfile.write(fragment)
                    self.wfile.flush()
                    time.sleep(0.002)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


@contextmanager
def local_server(
    *,
    status: int = 200,
    content_type: str | None = None,
    content_encoding: str | None = None,
    body_override: bytes | None = None,
    fragments: tuple[bytes, ...] | None = None,
    delay_headers: float = 0,
    hold_headers: bool = False,
    hold_stream: bool = False,
) -> Iterator[LocalServer]:
    server = LocalServer(("127.0.0.1", 0), LocalHandler)
    server.calls = []
    server.status = status
    server.content_type = content_type
    server.content_encoding = content_encoding
    server.body_override = body_override
    server.fragments = fragments
    server.delay_headers = delay_headers
    server.hold_headers = hold_headers
    server.headers_release = threading.Event()
    server.received = threading.Event()
    server.hold_stream = hold_stream
    server.closed = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.headers_release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def provider_for(server: LocalServer, **options: object) -> AsyncOpenAICompatibleProvider:
    return AsyncOpenAICompatibleProvider(
        f"http://127.0.0.1:{server.server_port}/v1", api_key="local-secret", **options
    )


def test_native_async_json_sse_and_registry_round_trip() -> None:
    async def scenario(server: LocalServer) -> None:
        provider = provider_for(server)
        registry = AsyncProviderRegistry((AsyncProviderTarget("routed", "upstream", provider),))
        payload = {
            "messages": [{"role": "user", "content": "2+2"}],
            "model": "wrong",
            "stream": True,
        }
        result = await registry.complete("routed", payload, timeout_seconds=10)
        chunks = [item async for item in registry.stream("routed", payload, timeout_seconds=10)]
        assert result == completion()
        assert chunks == [chunk()]
        assert payload["model"] == "wrong" and payload["stream"] is True

    with local_server() as server:
        asyncio.run(scenario(server))
        assert [call["path"] for call in server.calls] == [
            "/v1/chat/completions",
            "/v1/chat/completions",
        ]
        assert [call["payload"]["stream"] for call in server.calls] == [False, True]
        assert all(call["payload"]["model"] == "upstream" for call in server.calls)
        assert all(call["authorization"] == "Bearer local-secret" for call in server.calls)
        assert all(call["accept_encoding"] == "identity" for call in server.calls)


def test_native_async_provider_integrates_with_routing_controller(
    make_model: Any,
) -> None:
    async def scenario(server: LocalServer) -> None:
        router = RuleRouter((make_model("routed"),), {}, ())
        registry = AsyncProviderRegistry(
            (AsyncProviderTarget("routed", "upstream", provider_for(server)),)
        )
        controller = AsyncRoutingController(router, registry)
        result = await controller.complete(
            RouteRequest(query="2+2", request_id="local"),
            {"messages": [{"role": "user", "content": "2+2"}]},
            timeout_seconds=10,
        )
        assert result.decision.selected_model == "routed"
        assert result.response == completion()

    with local_server() as server:
        asyncio.run(scenario(server))
        assert len(server.calls) == 1
        assert server.calls[0]["payload"]["model"] == "upstream"


def test_async_sse_fragmented_comments_crlf_and_multiline_data() -> None:
    encoded = json.dumps(chunk(), separators=(",", ":")).encode()
    # A JSON object split at a legal whitespace boundary across data fields.
    split = encoded.index(b',"choices"')
    body = (
        b": keepalive\r\n\r\ndata: "
        + encoded[:split]
        + b"\r\ndata: "
        + encoded[split:]
        + b"\r\n\r\ndata: [DONE]\r\n\r\n"
    )
    fragments = tuple(body[index : index + 3] for index in range(0, len(body), 3))

    async def scenario(server: LocalServer) -> list[dict[str, Any]]:
        return [
            dict(value)
            async for value in provider_for(server).stream({}, model="upstream", timeout_seconds=10)
        ]

    with local_server(body_override=body, fragments=fragments) as server:
        assert asyncio.run(scenario(server)) == [chunk()]


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (302, ProviderFailure.REJECTED),
        (400, ProviderFailure.REJECTED),
        (408, ProviderFailure.TIMEOUT),
        (429, ProviderFailure.RATE_LIMITED),
        (500, ProviderFailure.UNAVAILABLE),
        (504, ProviderFailure.TIMEOUT),
    ],
)
def test_async_status_mapping_reads_no_error_body(status: int, failure: ProviderFailure) -> None:
    async def scenario(server: LocalServer) -> None:
        with pytest.raises(ProviderError) as raised:
            await provider_for(server).complete({}, model="upstream", timeout_seconds=10)
        assert raised.value.failure is failure
        assert not raised.value.retry_safe
        assert "untrusted.invalid" not in str(raised.value)

    with local_server(status=status) as server:
        asyncio.run(scenario(server))
        assert len(server.calls) == 1


@pytest.mark.parametrize(
    ("body", "content_type", "content_encoding", "limit"),
    [
        (b"not json", "application/json", None, 100),
        (b"[]", "application/json", None, 100),
        (b'{"a":1,"a":2}', "application/json", None, 100),
        (b'{"object":"chat.completion","choices":[]}', "application/json", None, 100),
        (json.dumps(completion()).encode(), "text/plain", None, 1000),
        (json.dumps(completion()).encode(), "application/json", "gzip", 1000),
        (json.dumps(completion()).encode(), "application/json", None, 10),
    ],
)
def test_async_json_rejects_malformed_oversize_or_encoded(
    body: bytes, content_type: str, content_encoding: str | None, limit: int
) -> None:
    async def scenario(server: LocalServer) -> None:
        with pytest.raises(ProviderError) as raised:
            await provider_for(server, max_response_bytes=limit).complete(
                {}, model="upstream", timeout_seconds=10
            )
        assert raised.value.failure is ProviderFailure.MALFORMED
        assert not raised.value.retry_safe

    with local_server(
        body_override=body, content_type=content_type, content_encoding=content_encoding
    ) as server:
        asyncio.run(scenario(server))


@pytest.mark.parametrize(
    ("body", "options", "content_type"),
    [
        (b"data: not-json\n\n", {}, None),
        (b"data: []\n\n", {}, None),
        (b"id: forbidden\n\n", {}, None),
        (b"data: \xff\n\n", {}, None),
        (b"data: [DONE]", {}, None),
        (b"data: " + b"x" * 80 + b"\n\n", {"max_event_bytes": 20}, None),
        (b":" + b"x" * 80 + b"\n\n", {"max_event_bytes": 20}, None),
        (b"data: {}\n\ndata: [DONE]\n\n", {"max_stream_bytes": 15}, None),
        (b"data: [DONE]\n\n", {}, "application/json"),
    ],
)
def test_async_sse_rejects_malformed_or_bounded_input(
    body: bytes, options: dict[str, int], content_type: str | None
) -> None:
    async def scenario(server: LocalServer) -> None:
        with pytest.raises(ProviderError) as raised:
            _ = [
                item
                async for item in provider_for(server, **options).stream(
                    {}, model="upstream", timeout_seconds=10
                )
            ]
        assert raised.value.failure is ProviderFailure.MALFORMED
        assert not raised.value.retry_safe

    with local_server(body_override=body, content_type=content_type) as server:
        asyncio.run(scenario(server))


def test_async_network_wait_does_not_block_event_loop() -> None:
    async def scenario(server: LocalServer) -> None:
        provider = provider_for(server)
        blocked = asyncio.create_task(provider.complete({}, model="upstream", timeout_seconds=15))
        for _ in range(2000):
            if server.received.is_set():
                break
            await asyncio.sleep(0.005)
        try:
            assert server.received.is_set() and not blocked.done()
            # The server cannot send headers until this task releases it.
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.5)
            assert not blocked.done()
        finally:
            server.headers_release.set()
        assert await blocked == completion()

    with local_server(hold_headers=True) as server:
        asyncio.run(scenario(server))


def test_async_transport_ignores_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")

    async def scenario(server: LocalServer) -> None:
        assert await provider_for(server).complete({}, model="upstream", timeout_seconds=10) == (
            completion()
        )

    with local_server() as server:
        asyncio.run(scenario(server))
        assert len(server.calls) == 1


def test_async_total_deadline_is_enforced_after_request_send() -> None:
    async def scenario(server: LocalServer) -> None:
        provider = provider_for(server)
        with pytest.raises(ProviderError) as raised:
            await provider.complete({}, model="upstream", timeout_seconds=2)
        assert raised.value.failure is ProviderFailure.TIMEOUT
        assert not raised.value.retry_safe
        assert server.received.is_set()

    with local_server(hold_headers=True) as server:
        asyncio.run(scenario(server))


def test_async_stream_cancellation_closes_connection() -> None:
    async def scenario(server: LocalServer) -> None:
        source = provider_for(server).stream({}, model="upstream", timeout_seconds=10)
        assert await anext(source) == chunk()
        waiter = asyncio.create_task(anext(source))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        for _ in range(100):
            if server.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert server.closed.is_set()

    with local_server(hold_stream=True) as server:
        asyncio.run(scenario(server))


def test_async_stream_deadline_does_not_cancel_paused_consumer() -> None:
    async def scenario(server: LocalServer) -> None:
        source = provider_for(server).stream({}, model="upstream", timeout_seconds=1)
        assert await anext(source) == chunk()
        # The absolute call budget expires during consumer work, not a network
        # await. The caller must stay alive; the next read reports timeout.
        await asyncio.sleep(1.1)
        with pytest.raises(ProviderError) as raised:
            await anext(source)
        assert raised.value.failure is ProviderFailure.TIMEOUT
        assert not raised.value.retry_safe
        for _ in range(100):
            if server.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert server.closed.is_set()

    with local_server(hold_stream=True) as server:
        asyncio.run(scenario(server))


def test_async_stream_network_stall_hits_total_deadline() -> None:
    async def scenario(server: LocalServer) -> None:
        source = provider_for(server).stream({}, model="upstream", timeout_seconds=1)
        assert await anext(source) == chunk()
        with pytest.raises(ProviderError) as raised:
            await anext(source)
        assert raised.value.failure is ProviderFailure.TIMEOUT
        assert not raised.value.retry_safe

    with local_server(hold_stream=True) as server:
        asyncio.run(scenario(server))


def test_async_stream_explicit_aclose_releases_socket() -> None:
    async def scenario(server: LocalServer) -> None:
        source = provider_for(server).stream({}, model="upstream", timeout_seconds=10)
        assert await anext(source) == chunk()
        await source.aclose()
        for _ in range(100):
            if server.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert server.closed.is_set()

    with local_server(hold_stream=True) as server:
        asyncio.run(scenario(server))


def test_async_stream_cross_task_aclose_releases_socket() -> None:
    async def scenario(server: LocalServer) -> None:
        source = provider_for(server).stream({}, model="upstream", timeout_seconds=10)
        assert await anext(source) == chunk()
        await asyncio.create_task(source.aclose())
        for _ in range(100):
            if server.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert server.closed.is_set()

    with local_server(hold_stream=True) as server:
        asyncio.run(scenario(server))


@pytest.mark.parametrize(
    ("error", "failure", "retry_safe"),
    [
        (httpx.ConnectTimeout("secret-local-url"), ProviderFailure.TIMEOUT, True),
        (httpx.ConnectError("secret-local-url"), ProviderFailure.UNAVAILABLE, True),
        (httpx.WriteTimeout("secret-local-url"), ProviderFailure.TIMEOUT, False),
        (httpx.ReadError("secret-local-url"), ProviderFailure.UNAVAILABLE, False),
        (httpx.RemoteProtocolError("secret-local-url"), ProviderFailure.FAILED, False),
    ],
)
def test_async_transport_only_certifies_presend_connect_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    failure: ProviderFailure,
    retry_safe: bool,
) -> None:
    async def failing(*_args: object, **_kwargs: object) -> httpx.Response:
        raise error

    monkeypatch.setattr(httpx.AsyncClient, "send", failing)

    async def scenario() -> None:
        with pytest.raises(ProviderError) as raised:
            await AsyncOpenAICompatibleProvider("http://127.0.0.1:1/v1").complete(
                {}, model="upstream", timeout_seconds=2
            )
        assert raised.value.failure is failure
        assert raised.value.retry_safe is retry_safe
        assert "secret-local-url" not in str(raised.value)

    asyncio.run(scenario())


def test_async_invalid_endpoint_error_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid(*_args: object, **_kwargs: object) -> httpx.Request:
        raise httpx.InvalidURL("secret-local-url")

    monkeypatch.setattr(httpx.AsyncClient, "build_request", invalid)

    async def scenario() -> None:
        with pytest.raises(ConfigurationError, match="endpoint") as raised:
            await AsyncOpenAICompatibleProvider("http://127.0.0.1:1/v1").complete(
                {}, model="upstream", timeout_seconds=2
            )
        assert "secret-local-url" not in str(raised.value)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid",
    [
        "http://provider.example/v1",
        "https://user:pass@example.com/v1",
        "https://provider.example/v2",
        "https://provider.example/v1?q=x",
    ],
)
def test_async_url_validation_matches_sync_policy(invalid: str) -> None:
    with pytest.raises(ConfigurationError):
        AsyncOpenAICompatibleProvider(invalid)


def test_async_input_validation_and_redacted_configuration() -> None:
    with pytest.raises(ConfigurationError, match="api key"):
        AsyncOpenAICompatibleProvider("https://provider.example/v1", api_key="secret\nvalue")
    with pytest.raises(ConfigurationError, match="api key") as raised:
        AsyncOpenAICompatibleProvider("https://provider.example/v1", api_key="secret\ud800")
    assert "secret" not in str(raised.value)
    with pytest.raises(ConfigurationError, match="max_response_bytes"):
        AsyncOpenAICompatibleProvider("https://provider.example/v1", max_response_bytes=0)
    with pytest.raises(ConfigurationError, match="max_request_bytes"):
        AsyncOpenAICompatibleProvider("https://provider.example/v1", max_request_bytes=0)

    async def scenario() -> None:
        provider = AsyncOpenAICompatibleProvider("http://127.0.0.1:1/v1")
        with pytest.raises(ConfigurationError, match="timeout"):
            await provider.complete({}, model="upstream", timeout_seconds=0)
        with pytest.raises(ConfigurationError, match="model"):
            await provider.complete({}, model="", timeout_seconds=1)
        with pytest.raises(ConfigurationError, match="strict UTF-8 JSON") as raised:
            await provider.complete({"messages": ["\ud800"]}, model="upstream", timeout_seconds=1)
        assert "\ud800" not in str(raised.value)
        cyclic: list[Any] = ["private prompt"]
        cyclic.append(cyclic)
        with pytest.raises(ConfigurationError, match="strict UTF-8 JSON") as raised:
            await provider.complete({"messages": cyclic}, model="upstream", timeout_seconds=1)
        assert "private prompt" not in str(raised.value)
        deeply_nested: list[Any] = ["private prompt"]
        for _ in range(1200):
            deeply_nested = [deeply_nested]
        with pytest.raises(ConfigurationError, match="strict UTF-8 JSON") as raised:
            await provider.complete(
                {"messages": deeply_nested}, model="upstream", timeout_seconds=1
            )
        assert "private prompt" not in str(raised.value)
        with pytest.raises(ConfigurationError, match="request exceeds byte limit"):
            await AsyncOpenAICompatibleProvider(
                "http://127.0.0.1:1/v1", max_request_bytes=30
            ).complete({"messages": ["private prompt"]}, model="upstream", timeout_seconds=1)

    asyncio.run(scenario())
