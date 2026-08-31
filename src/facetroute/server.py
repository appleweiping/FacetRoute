"""Bounded standard-library HTTP service for provider-independent routing."""

from __future__ import annotations

import hmac
import json
import math
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from threading import BoundedSemaphore
from typing import Any, cast
from uuid import uuid4

from ._json import loads_strict
from .config import request_from_dict
from .errors import ConfigurationError, FacetRouteError
from .routers import Router
from .types import ModelCandidate, RouteRequest


def _server_version() -> str:
    try:
        return f"FacetRoute/{version('facetroute')}"
    except PackageNotFoundError:  # pragma: no cover - unpacked source tree
        return "FacetRoute"


class FacetRouteHTTPServer(ThreadingHTTPServer):
    """Threaded server with a hard bound on concurrently processed requests."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        router: Router,
        models: tuple[ModelCandidate, ...],
        *,
        max_body_bytes: int,
        max_concurrency: int,
        request_timeout_seconds: float,
        bearer_token: str | None,
    ) -> None:
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes <= 0
        ):
            raise ValueError("max_body_bytes must be positive")
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be positive")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        if bearer_token is not None and (not isinstance(bearer_token, str) or not bearer_token):
            raise ValueError("bearer_token must be a non-empty string")
        self.router = router
        self.models = models
        self.max_body_bytes = max_body_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.bearer_token = bearer_token
        self._capacity = BoundedSemaphore(max_concurrency)
        super().__init__(server_address, FacetRouteHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._capacity.acquire(blocking=False):
            try:
                request.settimeout(self.request_timeout_seconds)
                _BusyHandler(request, client_address, self)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._capacity.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._capacity.release()


class FacetRouteHandler(BaseHTTPRequestHandler):
    """HTTP surface that returns routing decisions, never model completions."""

    protocol_version = "HTTP/1.1"
    server_version = _server_version()
    sys_version = ""

    @property
    def app(self) -> FacetRouteHTTPServer:
        return cast(FacetRouteHTTPServer, self.server)

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.app.request_timeout_seconds)

    def log_message(self, format: str, *args: Any) -> None:
        # Applications can wrap the server with their own metadata-only access
        # logging. The built-in service never prints headers, prompts, or bodies.
        del format, args

    def do_GET(self) -> None:
        request_id = self._request_id()
        if self.path == "/health":
            self._json_response(HTTPStatus.OK, {"status": "ok"}, request_id)
            return
        if self.path == "/v1/models":
            if not self._authorized(request_id):
                return
            models = [
                {
                    "id": model.model_id,
                    "object": "routing.model",
                    "display_name": model.display_name,
                    "enabled": model.enabled,
                    "capabilities": sorted(model.capabilities),
                    "context_window": model.context_window,
                    "regions": sorted(model.regions),
                    "supports_tools": model.supports_tools,
                    "supports_json": model.supports_json,
                }
                for model in self.app.models
            ]
            self._json_response(
                HTTPStatus.OK,
                {"object": "list", "data": models},
                request_id,
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found", request_id)

    def do_POST(self) -> None:
        request_id = self._request_id()
        if self.path != "/v1/route":
            self.close_connection = True
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found", request_id)
            return
        if not self._authorized(request_id):
            return
        try:
            payload = self._read_payload()
            request, dialect = route_request_from_http(payload, request_id=request_id)
            decision = self.app.router.route(request)
        except _HTTPInputError as exc:
            if exc.close_connection:
                self.close_connection = True
            self._error(exc.status, exc.code, str(exc), request_id)
            return
        except (ConfigurationError, FacetRouteError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request", str(exc), request_id)
            return
        except TimeoutError:
            self.close_connection = True
            self._error(
                HTTPStatus.REQUEST_TIMEOUT, "request_timeout", "request timed out", request_id
            )
            return
        except Exception:
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "routing failed unexpectedly",
                request_id,
            )
            return
        self._json_response(
            HTTPStatus.OK,
            {
                "id": request_id,
                "object": "routing.decision",
                "model": decision.selected_model,
                "input_dialect": dialect,
                "decision": decision.to_dict(),
            },
            request_id,
        )

    def _authorized(self, request_id: str) -> bool:
        expected = self.app.bearer_token
        if expected is None:
            return True
        authorization = self.headers.get_all("Authorization", [])
        supplied = authorization[0] if len(authorization) == 1 else ""
        prefix = "Bearer "
        valid = supplied.startswith(prefix) and hmac.compare_digest(
            supplied[len(prefix) :], expected
        )
        if not valid:
            if self.command == "POST":
                # Authentication happens before the body is consumed. Reusing
                # this connection would interpret those bytes as a new request.
                self.close_connection = True
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "a valid bearer token is required",
                request_id,
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
        return valid

    def _read_payload(self) -> Mapping[str, Any]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "unsupported_transfer_encoding",
                "Transfer-Encoding is not supported",
                close_connection=True,
            )
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "ambiguous_content_type",
                "exactly one Content-Type header is required",
                close_connection=True,
            )
        media_type = content_types[0].split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise _HTTPInputError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
                close_connection=True,
            )
        length_headers = self.headers.get_all("Content-Length", [])
        if not length_headers:
            raise _HTTPInputError(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "Content-Length is required",
                close_connection=True,
            )
        if len(length_headers) != 1:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "ambiguous_content_length",
                "exactly one Content-Length header is required",
                close_connection=True,
            )
        length_header = length_headers[0]
        if not length_header.isascii() or not length_header.isdecimal():
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must contain only decimal digits",
                close_connection=True,
            )
        length = int(length_header)
        if length > self.app.max_body_bytes:
            raise _HTTPInputError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "payload_too_large",
                f"request body exceeds {self.app.max_body_bytes} bytes",
                close_connection=True,
            )
        try:
            raw = self.rfile.read(length)
        except TimeoutError as exc:
            raise TimeoutError from exc
        if len(raw) != length:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "incomplete_body",
                "request body ended before Content-Length bytes were received",
                close_connection=True,
            )
        try:
            payload = loads_strict(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                f"request body is not strict JSON: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body must be a JSON object",
            )
        return payload

    def _request_id(self) -> str:
        supplied = self.headers.get("X-Request-ID")
        if supplied and len(supplied) <= 128 and supplied.isascii() and supplied.isprintable():
            return supplied
        return uuid4().hex

    def _error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        request_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._json_response(
            status,
            {"error": {"code": code, "message": message, "request_id": request_id}},
            request_id,
            extra_headers=extra_headers,
        )

    def _json_response(
        self,
        status: HTTPStatus,
        payload: Mapping[str, Any],
        request_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        body = (
            json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", request_id)
        if self.close_connection:
            self.send_header("Connection", "close")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


class _BusyHandler(BaseHTTPRequestHandler):
    """Consume the request line before closing an overloaded connection.

    Sending on a socket while request bytes remain unread can result in a TCP
    reset on Windows, causing clients to lose the 503 response.  Parsing only
    the bounded HTTP header avoids that platform-specific failure without
    spawning another worker thread.
    """

    protocol_version = "HTTP/1.1"
    server_version = _server_version()
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_GET(self) -> None:
        self._reject()

    def do_POST(self) -> None:
        self._reject()

    def _reject(self) -> None:
        body = b'{"error":{"code":"server_busy","message":"routing capacity is full"}}\n'
        self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


class _HTTPInputError(ValueError):
    def __init__(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        close_connection: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.close_connection = close_connection


_NATIVE_FIELDS = {
    "query",
    "user_id",
    "expected_output_tokens",
    "required_capabilities",
    "max_cost_usd",
    "max_latency_ms",
    "region",
    "needs_tools",
    "needs_json",
    "sensitivity",
    "task_hint",
    "context_tokens",
    "request_id",
    "metadata",
}
_COMPAT_FIELDS = (
    _NATIVE_FIELDS - {"query", "expected_output_tokens", "needs_tools", "needs_json"}
) | {
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "response_format",
}


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list) or not messages or len(messages) > 128:
        raise ConfigurationError("messages must be a non-empty array of at most 128 messages")
    parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ConfigurationError(f"messages[{index}] must be an object")
        unknown = set(message) - {"role", "content", "name"}
        if unknown:
            raise ConfigurationError(f"unknown messages[{index}] fields: {sorted(unknown)}")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ConfigurationError(f"messages[{index}].role is unsupported")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or set(part) != {"type", "text"}:
                    raise ConfigurationError(
                        f"messages[{index}].content[{part_index}] must be a text part"
                    )
                if part["type"] != "text" or not isinstance(part["text"], str):
                    raise ConfigurationError(
                        f"messages[{index}].content[{part_index}] must be a text part"
                    )
                text_parts.append(part["text"])
            text = "\n".join(text_parts)
        else:
            raise ConfigurationError(f"messages[{index}].content must be text or text parts")
        if text.strip():
            parts.append(f"[{role}]\n{text.strip()}")
    if not parts:
        raise ConfigurationError("messages contain no text")
    return "\n\n".join(parts)


def route_request_from_http(
    payload: Mapping[str, Any], *, request_id: str
) -> tuple[RouteRequest, str]:
    """Parse native or request-shaped OpenAI chat input without proxying it."""

    data = dict(payload)
    if "messages" not in data:
        unknown = set(data) - _NATIVE_FIELDS
        if unknown:
            raise ConfigurationError(f"unknown route fields: {sorted(unknown)}")
        data.setdefault("request_id", request_id)
        return request_from_dict(data), "native"
    unknown = set(data) - _COMPAT_FIELDS
    if unknown:
        raise ConfigurationError(f"unknown compatible request fields: {sorted(unknown)}")
    data["query"] = _message_text(data.pop("messages"))
    max_tokens = data.pop("max_completion_tokens", data.pop("max_tokens", 256))
    data["expected_output_tokens"] = max_tokens
    tools = data.pop("tools", [])
    if not isinstance(tools, list):
        raise ConfigurationError("tools must be an array")
    data["needs_tools"] = bool(tools)
    response_format = data.pop("response_format", None)
    if response_format is not None:
        if not isinstance(response_format, dict) or response_format.get("type") not in {
            "text",
            "json_object",
            "json_schema",
        }:
            raise ConfigurationError("response_format.type is unsupported")
        data["needs_json"] = response_format.get("type") != "text"
    else:
        data["needs_json"] = False
    data.setdefault("request_id", request_id)
    return request_from_dict(data), "openai-chat-request"


def create_server(
    router: Router,
    models: tuple[ModelCandidate, ...],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_body_bytes: int = 262_144,
    max_concurrency: int = 32,
    request_timeout_seconds: float = 10.0,
    bearer_token: str | None = None,
) -> FacetRouteHTTPServer:
    """Create, but do not start, a bounded routing HTTP server."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    return FacetRouteHTTPServer(
        (host, port),
        router,
        models,
        max_body_bytes=max_body_bytes,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        bearer_token=bearer_token,
    )
