"""Bounded HTTP service for routing and explicitly configured provider execution."""

from __future__ import annotations

import hmac
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from threading import BoundedSemaphore
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

from ._json import loads_strict
from .config import request_from_dict
from .errors import ConfigurationError, FacetRouteError
from .providers import ProviderError, ProviderFailure, ProviderRegistry
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
        provider_registry: ProviderRegistry | None,
        provider_timeout_seconds: float,
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
        if (
            isinstance(provider_timeout_seconds, bool)
            or not isinstance(provider_timeout_seconds, (int, float))
            or not math.isfinite(provider_timeout_seconds)
            or provider_timeout_seconds <= 0
        ):
            raise ValueError("provider_timeout_seconds must be positive")
        self.router = router
        self.models = models
        self.max_body_bytes = max_body_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.bearer_token = bearer_token
        self.provider_registry = provider_registry
        self.provider_timeout_seconds = float(provider_timeout_seconds)
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
    """HTTP surface for decisions and optional routed model completions."""

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
        if self.path not in {"/v1/route", "/v1/chat/completions"}:
            self.close_connection = True
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found", request_id)
            return
        if not self._authorized(request_id):
            return
        try:
            payload = self._read_payload()
            if self.path == "/v1/chat/completions":
                self._chat_completion(payload, request_id)
                return
            request, dialect = route_request_from_http(payload, request_id=request_id)
            decision = self.app.router.route(request)
        except _HTTPInputError as exc:
            if exc.close_connection:
                self.close_connection = True
            self._error(exc.status, exc.code, str(exc), request_id)
            return
        except ProviderError as exc:
            status = _provider_status(exc.failure)
            self._error(status, exc.failure.value, str(exc), request_id)
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

    def _chat_completion(self, payload: Mapping[str, Any], request_id: str) -> None:
        request, provider_payload, stream = chat_completion_from_http(
            payload, request_id=request_id
        )
        decision = self.app.router.route(request)
        registry = self.app.provider_registry
        if registry is None:
            raise ProviderError(ProviderFailure.NOT_CONFIGURED)
        headers = {
            "X-FacetRoute-Model": decision.selected_model,
            "X-FacetRoute-Policy": decision.policy,
        }
        if not stream:
            result = registry.complete(
                decision.selected_model,
                provider_payload,
                timeout_seconds=self.app.provider_timeout_seconds,
            )
            self._json_response(HTTPStatus.OK, result, request_id, extra_headers=headers)
            return
        chunks = registry.stream(
            decision.selected_model,
            provider_payload,
            timeout_seconds=self.app.provider_timeout_seconds,
        )
        try:
            first = next(chunks)
        except StopIteration as exc:
            raise ProviderError(ProviderFailure.MALFORMED) from exc
        self._sse_response(first, chunks, request_id, extra_headers=headers)

    def _sse_response(
        self,
        first: Mapping[str, Any],
        chunks: Iterator[Mapping[str, Any]],
        request_id: str,
        *,
        extra_headers: Mapping[str, str],
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-ID", _safe_header(request_id))
        for key, value in extra_headers.items():
            self.send_header(key, _safe_header(value))
        self.end_headers()
        failed = False
        try:
            self._write_sse_data(first)
            for chunk in chunks:
                self._write_sse_data(chunk)
            self._write_chunk(b"data: [DONE]\n\n")
        except ProviderError as exc:
            failed = True
            error = {
                "error": {
                    "code": exc.failure.value,
                    "message": str(exc),
                    "request_id": request_id,
                }
            }
            with suppress(BrokenPipeError, ConnectionResetError):
                self._write_sse_data(error)
        except (BrokenPipeError, ConnectionResetError):
            failed = True
        except Exception:
            failed = True
            with suppress(BrokenPipeError, ConnectionResetError):
                self._write_sse_data(
                    {
                        "error": {
                            "code": ProviderFailure.FAILED.value,
                            "message": "the upstream provider request failed",
                            "request_id": request_id,
                        }
                    }
                )
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            if failed:
                self.close_connection = True

    def _write_sse_data(self, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        self._write_chunk(b"data: " + encoded + b"\n\n")

    def _write_chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

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
        # Keep this final sink-side guard even though ``_request_id`` rejects
        # control characters. It protects future internal callers and makes
        # the response-splitting invariant explicit at the HTTP boundary.
        safe_request_id = _safe_header(request_id)
        self.send_header("X-Request-ID", safe_request_id)
        if self.close_connection:
            self.send_header("Connection", "close")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, _safe_header(value))
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
        name = message.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 64):
            raise ConfigurationError(
                f"messages[{index}].name must be a non-empty string of at most 64 characters"
            )
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


_CHAT_COMPLETION_FIELDS = {
    "model",
    "messages",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "user",
    "facetroute",
}
_ROUTING_EXTENSION_FIELDS = {
    "required_capabilities",
    "max_cost_usd",
    "max_latency_ms",
    "region",
    "sensitivity",
    "task_hint",
    "context_tokens",
    "metadata",
}


def chat_completion_from_http(
    payload: Mapping[str, Any], *, request_id: str
) -> tuple[RouteRequest, dict[str, Any], bool]:
    """Validate a text chat request and derive the provider-independent request."""

    data = dict(payload)
    unknown = set(data) - _CHAT_COMPLETION_FIELDS
    if unknown:
        raise ConfigurationError(f"unknown chat completion fields: {sorted(unknown)}")
    model = data.get("model")
    if model != "facetroute":
        raise ConfigurationError("model must be 'facetroute' for routed chat completions")
    if "messages" not in data:
        raise ConfigurationError("messages is required")
    if "max_tokens" in data and "max_completion_tokens" in data:
        raise ConfigurationError("max_tokens and max_completion_tokens cannot both be supplied")
    stream = data.get("stream", False)
    if not isinstance(stream, bool):
        raise ConfigurationError("stream must be a boolean")
    data["stream"] = stream
    _validate_optional_number(data, "frequency_penalty", minimum=-2.0, maximum=2.0)
    _validate_optional_number(data, "presence_penalty", minimum=-2.0, maximum=2.0)
    _validate_optional_number(data, "temperature", minimum=0.0, maximum=2.0)
    _validate_optional_number(data, "top_p", minimum=0.0, maximum=1.0)
    _validate_optional_integer(data, "n", minimum=1, maximum=8)
    _validate_optional_integer(data, "seed", minimum=-(2**63), maximum=2**63 - 1)
    _validate_optional_integer(data, "top_logprobs", minimum=0, maximum=20)
    if "logprobs" in data and not isinstance(data["logprobs"], bool):
        raise ConfigurationError("logprobs must be a boolean")
    if data.get("top_logprobs") is not None and data.get("logprobs") is not True:
        raise ConfigurationError("top_logprobs requires logprobs=true")
    if "parallel_tool_calls" in data and not isinstance(data["parallel_tool_calls"], bool):
        raise ConfigurationError("parallel_tool_calls must be a boolean")
    logit_bias = data.get("logit_bias")
    if logit_bias is not None:
        if not isinstance(logit_bias, dict) or len(logit_bias) > 1024:
            raise ConfigurationError("logit_bias must be an object with at most 1024 entries")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -100 <= value <= 100
            for value in logit_bias.values()
        ):
            raise ConfigurationError("logit_bias values must be numbers between -100 and 100")
    _validate_stop(data.get("stop"))
    tools = data.get("tools")
    if tools is not None and (not isinstance(tools, list) or len(tools) > 128):
        raise ConfigurationError("tools must be an array of at most 128 entries")
    tool_choice = data.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        raise ConfigurationError("tool_choice must be a string or an object")
    stream_options = data.get("stream_options")
    if stream_options is not None:
        if not stream or not isinstance(stream_options, dict):
            raise ConfigurationError("stream_options requires stream=true and must be an object")
        if set(stream_options) - {"include_usage"} or not isinstance(
            stream_options.get("include_usage", False), bool
        ):
            raise ConfigurationError("stream_options only supports boolean include_usage")
    user = data.get("user", "default")
    if not isinstance(user, str) or not user.strip() or len(user) > 256:
        raise ConfigurationError("user must be a non-empty string of at most 256 characters")
    extension = data.pop("facetroute", {})
    if not isinstance(extension, dict):
        raise ConfigurationError("facetroute must be an object")
    unknown_extension = set(extension) - _ROUTING_EXTENSION_FIELDS
    if unknown_extension:
        raise ConfigurationError(f"unknown facetroute fields: {sorted(unknown_extension)}")
    route_payload: dict[str, Any] = {
        "messages": data["messages"],
        "user_id": user,
        "request_id": request_id,
    }
    for name in ("max_tokens", "max_completion_tokens", "tools", "response_format"):
        if name in data:
            route_payload[name] = data[name]
    route_payload.update(extension)
    route_request, _ = route_request_from_http(route_payload, request_id=request_id)
    return route_request, data, stream


def _validate_optional_number(
    data: Mapping[str, Any], name: str, *, minimum: float, maximum: float
) -> None:
    value = data.get(name)
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ConfigurationError(f"{name} must be a number between {minimum} and {maximum}")


def _validate_optional_integer(
    data: Mapping[str, Any], name: str, *, minimum: int, maximum: int
) -> None:
    value = data.get(name)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be an integer between {minimum} and {maximum}")


def _validate_stop(value: Any) -> None:
    if value is None or isinstance(value, str):
        return
    if (
        not isinstance(value, list)
        or len(value) > 4
        or any(not isinstance(item, str) for item in value)
    ):
        raise ConfigurationError("stop must be a string or an array of at most four strings")


def _safe_header(value: str) -> str:
    # Keep advisory header values ASCII-only and response-splitting safe while
    # preserving the original identifier in the JSON response body.
    return quote(value, safe="!#$%&'*+-.^_`|~")[:2048]


def _provider_status(failure: ProviderFailure) -> HTTPStatus:
    if failure is ProviderFailure.TIMEOUT:
        return HTTPStatus.GATEWAY_TIMEOUT
    if failure in {
        ProviderFailure.NOT_CONFIGURED,
        ProviderFailure.RATE_LIMITED,
        ProviderFailure.UNAVAILABLE,
    }:
        return HTTPStatus.SERVICE_UNAVAILABLE
    return HTTPStatus.BAD_GATEWAY


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
    provider_registry: ProviderRegistry | None = None,
    provider_timeout_seconds: float = 60.0,
) -> FacetRouteHTTPServer:
    """Create, but do not start, a bounded routing HTTP server."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if provider_registry is not None:
        catalog_ids = {model.model_id for model in models}
        unknown = provider_registry.model_ids - catalog_ids
        if unknown:
            raise ValueError(
                f"provider registry references unknown catalog models: {sorted(unknown)}"
            )
        missing = {
            model.model_id for model in models if model.enabled
        } - provider_registry.model_ids
        if missing:
            raise ValueError(
                f"provider registry is missing enabled catalog models: {sorted(missing)}"
            )
    return FacetRouteHTTPServer(
        (host, port),
        router,
        models,
        max_body_bytes=max_body_bytes,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        bearer_token=bearer_token,
        provider_registry=provider_registry,
        provider_timeout_seconds=provider_timeout_seconds,
    )
