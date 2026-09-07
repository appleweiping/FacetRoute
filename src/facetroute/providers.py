"""Provider execution boundary for OpenAI-compatible chat completions.

The routing core stays provider independent.  This module owns the explicitly
configured network boundary and exposes a small protocol so applications and
tests can inject an executor without credentials or outbound traffic.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import SplitResult, urlsplit

from ._json import loads_strict
from .errors import ConfigurationError, FacetRouteError

_MAX_PROVIDER_CONFIG_BYTES = 1024 * 1024
_MAX_PROVIDER_BINDINGS = 1024
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ProviderFailure(StrEnum):
    """Stable, non-secret failure classes exposed by the HTTP boundary."""

    NOT_CONFIGURED = "provider_not_configured"
    TIMEOUT = "upstream_timeout"
    RATE_LIMITED = "upstream_rate_limited"
    UNAVAILABLE = "upstream_unavailable"
    REJECTED = "upstream_rejected"
    MALFORMED = "upstream_malformed"
    FAILED = "upstream_failed"


_FAILURE_MESSAGES = {
    ProviderFailure.NOT_CONFIGURED: "the selected model has no configured provider",
    ProviderFailure.TIMEOUT: "the upstream provider timed out",
    ProviderFailure.RATE_LIMITED: "the upstream provider is rate limited",
    ProviderFailure.UNAVAILABLE: "the upstream provider is unavailable",
    ProviderFailure.REJECTED: "the upstream provider rejected the routed request",
    ProviderFailure.MALFORMED: "the upstream provider returned an invalid response",
    ProviderFailure.FAILED: "the upstream provider request failed",
}


class ProviderError(FacetRouteError):
    """A redacted provider error safe to return to a caller."""

    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(_FAILURE_MESSAGES[failure])
        self.failure = failure


class ChatCompletionProvider(Protocol):
    """Injectable provider contract used after a model has been selected."""

    def complete(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...

    def stream(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    """Bind one catalog model to one executor and upstream model name."""

    model_id: str
    upstream_model: str
    provider: ChatCompletionProvider

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id.strip()
            or len(self.model_id) > 256
        ):
            raise ConfigurationError("provider model_id must contain at most 256 characters")
        if (
            not isinstance(self.upstream_model, str)
            or not self.upstream_model.strip()
            or len(self.upstream_model) > 512
        ):
            raise ConfigurationError("upstream_model must contain at most 512 characters")
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "upstream_model", self.upstream_model.strip())


class ProviderRegistry:
    """Immutable catalog-model to provider mapping."""

    def __init__(self, targets: tuple[ProviderTarget, ...]) -> None:
        if not targets:
            raise ConfigurationError("provider registry cannot be empty")
        by_model = {target.model_id: target for target in targets}
        if len(by_model) != len(targets):
            raise ConfigurationError("provider registry contains duplicate model_id values")
        self._targets = by_model

    @property
    def model_ids(self) -> frozenset[str]:
        return frozenset(self._targets)

    def complete(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        target = self._target(model_id)
        try:
            result = target.provider.complete(
                payload,
                model=target.upstream_model,
                timeout_seconds=timeout_seconds,
            )
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise ProviderError(ProviderFailure.TIMEOUT) from exc
        except Exception as exc:
            raise ProviderError(ProviderFailure.FAILED) from exc
        return _validate_completion(result)

    def stream(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        target = self._target(model_id)
        try:
            source = target.provider.stream(
                payload,
                model=target.upstream_model,
                timeout_seconds=timeout_seconds,
            )
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise ProviderError(ProviderFailure.TIMEOUT) from exc
        except Exception as exc:
            raise ProviderError(ProviderFailure.FAILED) from exc

        def validated() -> Iterator[Mapping[str, Any]]:
            try:
                try:
                    for chunk in source:
                        yield _validate_chunk(chunk)
                except ProviderError:
                    raise
                except TimeoutError as exc:
                    raise ProviderError(ProviderFailure.TIMEOUT) from exc
                except Exception as exc:
                    raise ProviderError(ProviderFailure.FAILED) from exc
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        raise ProviderError(ProviderFailure.FAILED) from exc

        return validated()

    def _target(self, model_id: str) -> ProviderTarget:
        try:
            return self._targets[model_id]
        except KeyError as exc:
            raise ProviderError(ProviderFailure.NOT_CONFIGURED) from exc


def _validate_completion(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProviderError(ProviderFailure.MALFORMED)
    result = dict(payload)
    _require_strict_json(result)
    if payload.get("object") != "chat.completion":
        raise ProviderError(ProviderFailure.MALFORMED)
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ProviderError(ProviderFailure.MALFORMED)
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ProviderError(ProviderFailure.MALFORMED)
    if (
        not isinstance(payload.get("choices"), list)
        or not payload["choices"]
        or any(not isinstance(choice, Mapping) for choice in payload["choices"])
    ):
        raise ProviderError(ProviderFailure.MALFORMED)
    return result


def _validate_chunk(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProviderError(ProviderFailure.MALFORMED)
    result = dict(payload)
    _require_strict_json(result)
    if payload.get("object") != "chat.completion.chunk":
        raise ProviderError(ProviderFailure.MALFORMED)
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ProviderError(ProviderFailure.MALFORMED)
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ProviderError(ProviderFailure.MALFORMED)
    if not isinstance(payload.get("choices"), list) or any(
        not isinstance(choice, Mapping) for choice in payload["choices"]
    ):
        raise ProviderError(ProviderFailure.MALFORMED)
    return result


def _require_strict_json(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProviderError(ProviderFailure.MALFORMED) from exc


class OpenAICompatibleProvider:
    """Execute chat completions against one fixed OpenAI-compatible base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        allow_insecure_http: bool = False,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_stream_bytes: int = 64 * 1024 * 1024,
        max_event_bytes: int = 1024 * 1024,
    ) -> None:
        self._url = _validate_base_url(base_url, allow_insecure_http=allow_insecure_http)
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key or "\r" in api_key or "\n" in api_key
        ):
            raise ConfigurationError("provider api key must be a non-empty string")
        for name, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_stream_bytes", max_stream_bytes),
            ("max_event_bytes", max_event_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        self._api_key = api_key
        self._max_response_bytes = max_response_bytes
        self._max_stream_bytes = max_stream_bytes
        self._max_event_bytes = max_event_bytes

    def complete(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        connection, response = self._request(
            payload, model=model, stream=False, timeout_seconds=timeout_seconds
        )
        try:
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].lower()
            if content_type != "application/json":
                raise ProviderError(ProviderFailure.MALFORMED)
            raw = response.read(self._max_response_bytes + 1)
            if len(raw) > self._max_response_bytes:
                raise ProviderError(ProviderFailure.MALFORMED)
            try:
                result = loads_strict(raw)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ProviderError(ProviderFailure.MALFORMED) from exc
            if not isinstance(result, dict):
                raise ProviderError(ProviderFailure.MALFORMED)
            return _validate_completion(result)
        except TimeoutError as exc:
            raise ProviderError(ProviderFailure.TIMEOUT) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise ProviderError(ProviderFailure.FAILED) from exc
        finally:
            connection.close()

    def stream(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        timeout_seconds: float,
    ) -> Iterator[Mapping[str, Any]]:
        connection, response = self._request(
            payload, model=model, stream=True, timeout_seconds=timeout_seconds
        )
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].lower()
        if content_type != "text/event-stream":
            connection.close()
            raise ProviderError(ProviderFailure.MALFORMED)

        def events() -> Iterator[Mapping[str, Any]]:
            total = 0
            event_bytes = 0
            data_lines: list[str] = []
            done = False
            try:
                while not done:
                    raw = response.readline(self._max_event_bytes + 1)
                    if not raw:
                        raise ProviderError(ProviderFailure.MALFORMED)
                    total += len(raw)
                    event_bytes += len(raw)
                    if (
                        len(raw) > self._max_event_bytes
                        or event_bytes > self._max_event_bytes
                        or total > self._max_stream_bytes
                    ):
                        raise ProviderError(ProviderFailure.MALFORMED)
                    try:
                        line = raw.decode("utf-8").rstrip("\r\n")
                    except UnicodeDecodeError as exc:
                        raise ProviderError(ProviderFailure.MALFORMED) from exc
                    if not line:
                        event_bytes = 0
                        if not data_lines:
                            continue
                        data = "\n".join(data_lines)
                        data_lines.clear()
                        if data == "[DONE]":
                            done = True
                            continue
                        try:
                            chunk = loads_strict(data)
                        except ValueError as exc:
                            raise ProviderError(ProviderFailure.MALFORMED) from exc
                        if not isinstance(chunk, dict):
                            raise ProviderError(ProviderFailure.MALFORMED)
                        yield _validate_chunk(chunk)
                    elif line.startswith(":"):
                        continue
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                    else:
                        raise ProviderError(ProviderFailure.MALFORMED)
            except TimeoutError as exc:
                raise ProviderError(ProviderFailure.TIMEOUT) from exc
            except (OSError, http.client.HTTPException) as exc:
                raise ProviderError(ProviderFailure.FAILED) from exc
            finally:
                connection.close()

        return events()

    def _request(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
        timeout_seconds: float,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("provider timeout must be positive")
        body_payload = dict(payload)
        body_payload["model"] = model
        body_payload["stream"] = stream
        try:
            body = json.dumps(
                body_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"provider payload is not strict JSON: {exc}") from exc
        connection_class: type[http.client.HTTPConnection]
        if self._url.scheme == "https":
            connection_class = http.client.HTTPSConnection
        else:
            connection_class = http.client.HTTPConnection
        connection = connection_class(
            cast(str, self._url.hostname),
            self._url.port,
            timeout=float(timeout_seconds),
        )
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "User-Agent": "FacetRoute-provider/1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        endpoint = f"{self._url.path.rstrip('/')}/chat/completions"
        try:
            connection.request("POST", endpoint, body=body, headers=headers)
            response = connection.getresponse()
        except TimeoutError as exc:
            connection.close()
            raise ProviderError(ProviderFailure.TIMEOUT) from exc
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            raise ProviderError(ProviderFailure.UNAVAILABLE) from exc
        if response.status == 200:
            return connection, response
        connection.close()
        if response.status == 429:
            raise ProviderError(ProviderFailure.RATE_LIMITED)
        if response.status in {408, 504}:
            raise ProviderError(ProviderFailure.TIMEOUT)
        if response.status >= 500:
            raise ProviderError(ProviderFailure.UNAVAILABLE)
        raise ProviderError(ProviderFailure.REJECTED)


def _validate_base_url(base_url: str, *, allow_insecure_http: bool) -> SplitResult:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigurationError("provider base_url must be a non-empty URL")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("provider base_url must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError("provider base_url cannot contain credentials, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("provider base_url contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ConfigurationError("provider base_url port must be between 1 and 65535")
    if not parsed.path.rstrip("/").endswith("/v1"):
        raise ConfigurationError("provider base_url path must end with /v1")
    if parsed.scheme == "http" and not allow_insecure_http and not _is_loopback(parsed.hostname):
        raise ConfigurationError("plain HTTP provider URLs are restricted to loopback hosts")
    return parsed


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_provider_registry(
    path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    allow_insecure_http: bool = False,
    max_response_bytes: int = 16 * 1024 * 1024,
    max_stream_bytes: int = 64 * 1024 * 1024,
    max_event_bytes: int = 1024 * 1024,
) -> ProviderRegistry:
    """Load provider bindings without allowing literal secrets in the file."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read provider configuration {source}: {exc}") from exc
    if len(raw) > _MAX_PROVIDER_CONFIG_BYTES:
        raise ConfigurationError(
            f"provider configuration exceeds {_MAX_PROVIDER_CONFIG_BYTES} bytes"
        )
    try:
        payload = loads_strict(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read provider configuration {source}: {exc}") from exc
    if isinstance(payload, dict) and set(payload) - {"models"}:
        raise ConfigurationError(
            f"unknown provider configuration fields: {sorted(set(payload) - {'models'})}"
        )
    records = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ConfigurationError("provider configuration must contain a non-empty models list")
    if len(records) > _MAX_PROVIDER_BINDINGS:
        raise ConfigurationError(
            f"provider configuration cannot exceed {_MAX_PROVIDER_BINDINGS} model bindings"
        )
    env = os.environ if environment is None else environment
    targets: list[ProviderTarget] = []
    allowed = {"model_id", "upstream_model", "base_url", "api_key_env"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ConfigurationError(f"provider models[{index}] must be an object")
        unknown = set(record) - allowed
        if unknown:
            raise ConfigurationError(f"unknown provider models[{index}] fields: {sorted(unknown)}")
        try:
            model_id = record["model_id"]
            upstream_model = record["upstream_model"]
            base_url = record["base_url"]
        except KeyError as exc:
            raise ConfigurationError(f"provider models[{index}] is missing {exc.args[0]}") from exc
        if not all(isinstance(item, str) for item in (model_id, upstream_model, base_url)):
            raise ConfigurationError(
                f"provider models[{index}] identifiers and URL must be strings"
            )
        api_key_env = record.get("api_key_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or _ENVIRONMENT_NAME.fullmatch(api_key_env) is None
        ):
            raise ConfigurationError(f"provider models[{index}].api_key_env must be a string")
        api_key = None
        if api_key_env is not None:
            api_key = env.get(api_key_env)
            if not api_key:
                raise ConfigurationError(
                    f"provider models[{index}] requires environment variable {api_key_env}"
                )
        targets.append(
            ProviderTarget(
                model_id=model_id,
                upstream_model=upstream_model,
                provider=OpenAICompatibleProvider(
                    base_url,
                    api_key=api_key,
                    allow_insecure_http=allow_insecure_http,
                    max_response_bytes=max_response_bytes,
                    max_stream_bytes=max_stream_bytes,
                    max_event_bytes=max_event_bytes,
                ),
            )
        )
    return ProviderRegistry(tuple(targets))
