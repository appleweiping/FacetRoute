"""Optional native-async OpenAI-compatible HTTP transport.

Install ``facetroute[async]`` to use this module. The default package and
synchronous provider do not import or depend on HTTPX.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from ._json import loads_strict
from .errors import ConfigurationError
from .providers import (
    ProviderError,
    ProviderFailure,
    _validate_base_url,
    _validate_chunk,
    _validate_completion,
)


class _SSEDecoder:
    """Incremental raw-byte SSE parser with total and per-event ceilings."""

    def __init__(self, *, max_stream_bytes: int, max_event_bytes: int) -> None:
        self._max_stream_bytes = max_stream_bytes
        self._max_event_bytes = max_event_bytes
        self._buffer = bytearray()
        self._total = 0
        self._event_bytes = 0
        self._data_lines: list[str] = []
        self.done = False

    def feed(self, raw: bytes) -> Iterator[Mapping[str, Any]]:
        if len(raw) > self._max_stream_bytes - self._total:
            raise ProviderError(ProviderFailure.MALFORMED)
        self._total += len(raw)
        self._buffer.extend(raw)
        while not self.done:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if (
                    len(self._buffer) > self._max_event_bytes
                    or self._event_bytes + len(self._buffer) > self._max_event_bytes
                ):
                    raise ProviderError(ProviderFailure.MALFORMED)
                return
            line_bytes = bytes(self._buffer[: newline + 1])
            del self._buffer[: newline + 1]
            self._event_bytes += len(line_bytes)
            if len(line_bytes) > self._max_event_bytes or self._event_bytes > self._max_event_bytes:
                raise ProviderError(ProviderFailure.MALFORMED)
            try:
                line = line_bytes.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                raise ProviderError(ProviderFailure.MALFORMED) from None
            if not line:
                self._event_bytes = 0
                if not self._data_lines:
                    continue
                data = "\n".join(self._data_lines)
                self._data_lines.clear()
                if data == "[DONE]":
                    self.done = True
                    return
                try:
                    chunk = loads_strict(data)
                except ValueError:
                    raise ProviderError(ProviderFailure.MALFORMED) from None
                if not isinstance(chunk, dict):
                    raise ProviderError(ProviderFailure.MALFORMED)
                yield _validate_chunk(chunk)
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                self._data_lines.append(line[5:].lstrip(" "))
            else:
                raise ProviderError(ProviderFailure.MALFORMED)


class AsyncOpenAICompatibleProvider:
    """Task-local HTTPX transport for one fixed upstream `/v1` endpoint.

    Each call owns and closes its client. This deliberately avoids hidden
    cross-loop connection state; applications needing pooling can layer it
    outside this conservative transport without changing the async registry.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        allow_insecure_http: bool = False,
        max_request_bytes: int = 16 * 1024 * 1024,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_stream_bytes: int = 64 * 1024 * 1024,
        max_event_bytes: int = 1024 * 1024,
    ) -> None:
        parsed = _validate_base_url(base_url, allow_insecure_http=allow_insecure_http)
        if api_key is not None and (
            not isinstance(api_key, str)
            or not api_key
            or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
        ):
            raise ConfigurationError("provider api key must be a non-empty printable ASCII token")
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
            ("max_stream_bytes", max_stream_bytes),
            ("max_event_bytes", max_event_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        self._endpoint = (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/chat/completions"
        )
        self._api_key = api_key
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_stream_bytes = max_stream_bytes
        self._max_event_bytes = max_event_bytes

    @staticmethod
    def _timeout(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ConfigurationError("provider timeout must be positive")
        return float(value)

    @staticmethod
    def _require_deadline(deadline: float) -> None:
        if asyncio.get_running_loop().time() >= deadline:
            raise ProviderError(ProviderFailure.TIMEOUT)

    def _body(self, payload: Mapping[str, Any], *, model: str, stream: bool) -> bytes:
        if not isinstance(payload, Mapping):
            raise ConfigurationError("provider payload must be a mapping")
        if type(model) is not str or not model.strip() or len(model) > 512:
            raise ConfigurationError("upstream model must be a bounded non-empty string")
        body_payload = dict(payload)
        body_payload["model"] = model
        body_payload["stream"] = stream
        try:
            encoded = bytearray()
            encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            for piece in encoder.iterencode(body_payload):
                part = piece.encode("utf-8")
                if len(part) > self._max_request_bytes - len(encoded):
                    raise ConfigurationError("provider request exceeds byte limit")
                encoded.extend(part)
            return bytes(encoded)
        except ConfigurationError:
            raise
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ConfigurationError("provider payload is not strict UTF-8 JSON") from None

    @staticmethod
    def _check_status(status: int) -> None:
        if status == 200:
            return
        if status == 429:
            raise ProviderError(ProviderFailure.RATE_LIMITED)
        if status in {408, 504}:
            raise ProviderError(ProviderFailure.TIMEOUT)
        if status >= 500:
            raise ProviderError(ProviderFailure.UNAVAILABLE)
        raise ProviderError(ProviderFailure.REJECTED)

    @staticmethod
    def _content_type(response: httpx.Response, expected: str) -> None:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if content_type != expected or encoding != "identity":
            raise ProviderError(ProviderFailure.MALFORMED)

    @asynccontextmanager
    async def _response(
        self, payload: Mapping[str, Any], *, model: str, stream: bool, timeout_seconds: float
    ) -> AsyncIterator[tuple[httpx.Response, float]]:
        timeout = self._timeout(timeout_seconds)
        deadline = asyncio.get_running_loop().time() + timeout
        body = self._body(payload, model=model, stream=stream)
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
            "User-Agent": "FacetRoute-provider/1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        before_response = True
        try:
            # Never leave a timeout armed across an SSE yield: it could cancel
            # the consumer while it is doing unrelated work between events.
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                verify=True,
                timeout=httpx.Timeout(timeout),
            ) as client:
                request = client.build_request(
                    "POST", self._endpoint, content=body, headers=headers
                )
                self._require_deadline(deadline)
                async with asyncio.timeout_at(deadline):
                    response = await client.send(request, stream=True)
                before_response = False
                try:
                    self._require_deadline(deadline)
                    self._check_status(response.status_code)
                    yield response, deadline
                finally:
                    await response.aclose()
        except ProviderError:
            raise
        except httpx.InvalidURL:
            raise ConfigurationError("provider endpoint is invalid") from None
        except httpx.ConnectTimeout:
            raise ProviderError(ProviderFailure.TIMEOUT, retry_safe=before_response) from None
        except httpx.ConnectError:
            raise ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=before_response) from None
        except httpx.TimeoutException:
            raise ProviderError(ProviderFailure.TIMEOUT) from None
        except TimeoutError:
            raise ProviderError(ProviderFailure.TIMEOUT) from None
        except httpx.NetworkError:
            raise ProviderError(ProviderFailure.UNAVAILABLE) from None
        except httpx.TransportError:
            raise ProviderError(ProviderFailure.FAILED) from None

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        async with self._response(
            payload, model=model, stream=False, timeout_seconds=timeout_seconds
        ) as (response, deadline):
            self._content_type(response, "application/json")
            body = bytearray()
            self._require_deadline(deadline)
            async with asyncio.timeout_at(deadline):
                async for chunk in response.aiter_raw():
                    if len(chunk) > self._max_response_bytes - len(body):
                        raise ProviderError(ProviderFailure.MALFORMED)
                    body.extend(chunk)
            try:
                result = loads_strict(bytes(body))
            except (UnicodeDecodeError, ValueError):
                raise ProviderError(ProviderFailure.MALFORMED) from None
            if not isinstance(result, dict):
                raise ProviderError(ProviderFailure.MALFORMED)
            validated = _validate_completion(result)
            self._require_deadline(deadline)
            return validated

    async def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        async with self._response(
            payload, model=model, stream=True, timeout_seconds=timeout_seconds
        ) as (response, deadline):
            self._content_type(response, "text/event-stream")
            decoder = _SSEDecoder(
                max_stream_bytes=self._max_stream_bytes, max_event_bytes=self._max_event_bytes
            )
            raw_chunks = response.aiter_raw()
            while True:
                self._require_deadline(deadline)
                try:
                    async with asyncio.timeout_at(deadline):
                        raw = await anext(raw_chunks)
                except StopAsyncIteration:
                    raise ProviderError(ProviderFailure.MALFORMED) from None
                for chunk in decoder.feed(raw):
                    self._require_deadline(deadline)
                    yield chunk
                if decoder.done:
                    self._require_deadline(deadline)
                    return
