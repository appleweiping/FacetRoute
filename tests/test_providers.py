from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from facetroute.errors import ConfigurationError
from facetroute.providers import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderFailure,
    ProviderRegistry,
    ProviderTarget,
    load_provider_registry,
)


class UpstreamServer(ThreadingHTTPServer):
    calls: list[dict[str, Any]]
    status: int
    malformed: bool
    body_override: bytes | None
    content_type_override: str | None


class UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        app = cast(UpstreamServer, self.server)
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        app.calls.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "payload": payload,
            }
        )
        if app.status != 200:
            self.send_response(app.status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if app.body_override is not None:
            body = app.body_override
            self.send_response(200)
            self.send_header(
                "Content-Type",
                app.content_type_override
                or ("text/event-stream" if payload["stream"] else "application/json"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if payload["stream"]:
            if app.malformed:
                body = b"event: forbidden\n\ndata: [DONE]\n\n"
            else:
                chunks = (
                    {
                        "id": "chatcmpl-local",
                        "object": "chat.completion.chunk",
                        "model": payload["model"],
                        "choices": [{"index": 0, "delta": {"content": "four"}}],
                    },
                    {
                        "id": "chatcmpl-local",
                        "object": "chat.completion.chunk",
                        "model": payload["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                )
                body = (
                    b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
                    + b"data: [DONE]\n\n"
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if app.malformed:
            body = b'{"object":"chat.completion","choices":[]}'
        else:
            body = json.dumps(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "four"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def upstream_server(
    *,
    status: int = 200,
    malformed: bool = False,
    body_override: bytes | None = None,
    content_type_override: str | None = None,
) -> Iterator[UpstreamServer]:
    server = UpstreamServer(("127.0.0.1", 0), UpstreamHandler)
    server.calls = []
    server.status = status
    server.malformed = malformed
    server.body_override = body_override
    server.content_type_override = content_type_override
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def provider_for(server: UpstreamServer, **options: object) -> OpenAICompatibleProvider:
    _, port = server.server_address[:2]
    return OpenAICompatibleProvider(
        f"http://127.0.0.1:{port}/v1",
        api_key="test-secret",
        **options,
    )


def test_openai_compatible_provider_executes_complete_and_stream_locally():
    with upstream_server() as upstream:
        provider = provider_for(upstream)
        result = provider.complete(
            {"messages": [{"role": "user", "content": "2+2"}]},
            model="upstream/model-v2",
            timeout_seconds=5,
        )
        chunks = list(
            provider.stream(
                {"messages": [{"role": "user", "content": "2+2"}]},
                model="upstream/model-v2",
                timeout_seconds=5,
            )
        )

    assert result["choices"][0]["message"]["content"] == "four"
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "four"
    assert [call["path"] for call in upstream.calls] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all(call["authorization"] == "Bearer test-secret" for call in upstream.calls)
    assert [call["payload"]["stream"] for call in upstream.calls] == [False, True]
    assert all(call["payload"]["model"] == "upstream/model-v2" for call in upstream.calls)


@pytest.mark.parametrize(
    "status, failure",
    [
        (400, ProviderFailure.REJECTED),
        (408, ProviderFailure.TIMEOUT),
        (429, ProviderFailure.RATE_LIMITED),
        (500, ProviderFailure.UNAVAILABLE),
    ],
)
def test_provider_maps_upstream_status_without_exposing_body(status, failure):
    with upstream_server(status=status) as upstream, pytest.raises(ProviderError) as raised:
        provider_for(upstream).complete({}, model="upstream", timeout_seconds=5)
    assert raised.value.failure is failure


def test_provider_rejects_malformed_complete_and_stream_responses():
    with upstream_server(malformed=True) as upstream:
        provider = provider_for(upstream)
        with pytest.raises(ProviderError) as complete_error:
            provider.complete({}, model="upstream", timeout_seconds=5)
        with pytest.raises(ProviderError) as stream_error:
            list(provider.stream({}, model="upstream", timeout_seconds=5))
    assert complete_error.value.failure is ProviderFailure.MALFORMED
    assert stream_error.value.failure is ProviderFailure.MALFORMED


@pytest.mark.parametrize(
    "body, content_type, limit",
    [
        (b"not json", "application/json", 100),
        (b"[]", "application/json", 100),
        (b'{"id":"x","object":"chat.completion","model":"m","choices":[]}', "text/plain", 100),
        (b'{"id":"x","object":"chat.completion","model":"m","choices":[]}', "application/json", 10),
    ],
)
def test_provider_bounds_and_validates_nonstream_payload(body, content_type, limit):
    with (
        upstream_server(body_override=body, content_type_override=content_type) as upstream,
        pytest.raises(ProviderError) as raised,
    ):
        provider_for(upstream, max_response_bytes=limit).complete(
            {}, model="upstream", timeout_seconds=5
        )
    assert raised.value.failure is ProviderFailure.MALFORMED


@pytest.mark.parametrize(
    "body, options",
    [
        (b"data: not-json\n\n", {}),
        (b"data: []\n\n", {}),
        (b"data: {}\n\n", {}),
        (b"id: unsupported\n\n", {}),
        (b"data: " + b"x" * 40 + b"\n\n", {"max_event_bytes": 16}),
        (
            b'data: {"id":"chunk",\n'
            b'data: "object":"chat.completion.chunk",\n'
            b'data: "model":"upstream","choices":[]}\n\n',
            {"max_event_bytes": 48},
        ),
        (b": keepalive padding\n\ndata: [DONE]\n\n", {"max_stream_bytes": 8}),
    ],
)
def test_provider_bounds_and_validates_stream_events(body, options):
    with upstream_server(body_override=body) as upstream, pytest.raises(ProviderError) as raised:
        list(provider_for(upstream, **options).stream({}, model="upstream", timeout_seconds=5))
    assert raised.value.failure is ProviderFailure.MALFORMED


def test_provider_accepts_sse_comments_and_requires_event_stream_content_type():
    chunk = {
        "id": "chunk",
        "object": "chat.completion.chunk",
        "model": "upstream",
        "choices": [],
    }
    body = b": heartbeat\n\n\ndata: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"
    with upstream_server(body_override=body) as upstream:
        assert list(provider_for(upstream).stream({}, model="upstream", timeout_seconds=5)) == [
            chunk
        ]
    with (
        upstream_server(body_override=body, content_type_override="application/json") as upstream,
        pytest.raises(ProviderError) as raised,
    ):
        provider_for(upstream).stream({}, model="upstream", timeout_seconds=5)
    assert raised.value.failure is ProviderFailure.MALFORMED


@pytest.mark.parametrize(
    "url",
    [
        "ftp://provider.example/v1",
        "https://user:password@provider.example/v1",
        "https://provider.example/v1?tenant=1",
        "https://provider.example/api",
        "https://provider.example:invalid/v1",
        "https://provider.example:0/v1",
        "http://provider.example/v1",
    ],
)
def test_provider_url_boundary_rejects_unsafe_or_ambiguous_urls(url):
    with pytest.raises(ConfigurationError):
        OpenAICompatibleProvider(url)
    OpenAICompatibleProvider("http://localhost/v1")
    OpenAICompatibleProvider("http://provider.example/v1", allow_insecure_http=True)


def test_provider_registry_validates_responses_from_injected_executor():
    class InvalidProvider:
        def complete(self, _payload, *, model, timeout_seconds):
            return {"object": "not-a-completion"}

        def stream(self, _payload, *, model, timeout_seconds):
            yield {"object": "not-a-chunk"}

    registry = ProviderRegistry((ProviderTarget("catalog", "upstream", InvalidProvider()),))
    with pytest.raises(ProviderError, match="invalid response"):
        registry.complete("catalog", {}, timeout_seconds=1)
    with pytest.raises(ProviderError, match="invalid response"):
        next(registry.stream("catalog", {}, timeout_seconds=1))
    with pytest.raises(ProviderError, match="no configured provider"):
        registry.complete("missing", {}, timeout_seconds=1)


def test_provider_registry_redacts_unexpected_executor_errors():
    class BrokenProvider:
        def complete(self, _payload, *, model, timeout_seconds):
            raise RuntimeError("secret provider detail")

        def stream(self, _payload, *, model, timeout_seconds):
            yield {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
            }
            raise RuntimeError("secret stream detail")

    registry = ProviderRegistry((ProviderTarget("catalog", "upstream", BrokenProvider()),))
    with pytest.raises(ProviderError) as complete_error:
        registry.complete("catalog", {}, timeout_seconds=1)
    stream = registry.stream("catalog", {}, timeout_seconds=1)
    assert next(stream)["object"] == "chat.completion.chunk"
    with pytest.raises(ProviderError) as stream_error:
        next(stream)
    assert complete_error.value.failure is ProviderFailure.FAILED
    assert stream_error.value.failure is ProviderFailure.FAILED
    assert "secret" not in str(complete_error.value)
    assert "secret" not in str(stream_error.value)


def test_load_provider_registry_uses_environment_only_for_secrets(tmp_path: Path):
    config = tmp_path / "providers.json"
    config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "quality",
                        "upstream_model": "vendor/model-v1",
                        "base_url": "https://provider.example/v1",
                        "api_key_env": "VENDOR_TOKEN",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = load_provider_registry(config, environment={"VENDOR_TOKEN": "secret"})
    assert registry.model_ids == {"quality"}
    with pytest.raises(ConfigurationError, match="VENDOR_TOKEN"):
        load_provider_registry(config, environment={})
    config.write_text('{"models":[{"model_id":"x","unexpected":true}]}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown provider"):
        load_provider_registry(config, environment={})


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "non-empty models"),
        ({"models": []}, "non-empty models"),
        ({"models": [1]}, "must be an object"),
        ({"models": [{"model_id": "x"}]}, "missing upstream_model"),
        (
            {
                "models": [
                    {"model_id": 1, "upstream_model": "u", "base_url": "https://x.example/v1"}
                ]
            },
            "must be strings",
        ),
        (
            {
                "models": [
                    {
                        "model_id": "x",
                        "upstream_model": "u",
                        "base_url": "https://x.example/v1",
                        "api_key_env": 1,
                    }
                ]
            },
            "api_key_env",
        ),
    ],
)
def test_load_provider_registry_rejects_malformed_files(tmp_path, payload, message):
    config = tmp_path / "providers.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_provider_registry(config, environment={})
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_provider_registry(tmp_path / "missing.json", environment={})


def test_provider_configuration_validates_limits_and_duplicate_bindings():
    with pytest.raises(ConfigurationError, match="max_response_bytes"):
        OpenAICompatibleProvider("https://provider.example/v1", max_response_bytes=0)
    target = ProviderTarget(
        "catalog",
        "upstream",
        OpenAICompatibleProvider("https://provider.example/v1"),
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        ProviderRegistry((target, target))
    with pytest.raises(ConfigurationError, match="cannot be empty"):
        ProviderRegistry(())
    with pytest.raises(ConfigurationError, match="model_id"):
        ProviderTarget(" ", "upstream", target.provider)
    with pytest.raises(ConfigurationError, match="upstream_model"):
        ProviderTarget("catalog", " ", target.provider)
    with pytest.raises(ConfigurationError, match="model_id"):
        ProviderTarget(1, "upstream", target.provider)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="upstream_model"):
        ProviderTarget("catalog", 1, target.provider)  # type: ignore[arg-type]
    normalized = ProviderTarget(" catalog ", " upstream ", target.provider)
    assert (normalized.model_id, normalized.upstream_model) == ("catalog", "upstream")
    with pytest.raises(ConfigurationError, match="api key"):
        OpenAICompatibleProvider("https://provider.example/v1", api_key="")


@pytest.mark.parametrize("timeout", [0, True, float("nan")])
def test_provider_rejects_invalid_timeout_before_network(timeout):
    provider = OpenAICompatibleProvider("https://provider.example/v1")
    with pytest.raises(ConfigurationError, match="timeout"):
        provider.complete({}, model="upstream", timeout_seconds=timeout)


def test_provider_rejects_non_json_forward_payload_before_network():
    provider = OpenAICompatibleProvider("https://provider.example/v1")
    with pytest.raises(ConfigurationError, match="strict JSON"):
        provider.complete({"temperature": float("nan")}, model="upstream", timeout_seconds=1)
