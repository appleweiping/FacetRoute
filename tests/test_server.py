from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from facetroute.errors import ConfigurationError
from facetroute.routers import RuleRouter
from facetroute.server import create_server, route_request_from_http
from facetroute.types import ModelCandidate, RouteRequest


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
        thread.join(timeout=2)


def request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    body: object | bytes | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object], dict[str, str]]:
    connection = http.client.HTTPConnection(*address, timeout=2)
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
        connection = http.client.HTTPConnection(*address, timeout=2)
        connection.putrequest("POST", "/v1/route")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        missing_response = connection.getresponse()
        missing_body = json.loads(missing_response.read())
        connection.close()

        raw = socket.create_connection(address, timeout=2)
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

        duplicate = socket.create_connection(address, timeout=2)
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
        thread.join(timeout=2)

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
