"""JSON configuration loading for catalogs, profiles, rules, and requests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import ConfigurationError
from .rules import RoutingRule
from .types import ModelCandidate, RouteRequest, UserPreferences


def _boolean_field(data: Mapping[str, Any], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _string_list(data: Mapping[str, Any], name: str) -> frozenset[str]:
    value = data.get(name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be a JSON array of strings")
    return frozenset(value)


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read JSON configuration {source}: {exc}") from exc


def load_models(path: str | Path) -> tuple[ModelCandidate, ...]:
    payload = _read_json(path)
    records = payload.get("models", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ConfigurationError("model catalog must be a list or an object with a models list")
    models = tuple(ModelCandidate.from_dict(record) for record in records)
    if not models:
        raise ConfigurationError("model catalog is empty")
    identifiers = [model.model_id for model in models]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("model catalog contains duplicate model_id values")
    return models


def load_preferences(path: str | Path | None) -> dict[str, UserPreferences]:
    if path is None:
        return {}
    payload = _read_json(path)
    records = payload.get("profiles", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ConfigurationError("preferences must be a list or an object with a profiles list")
    profiles = [UserPreferences.from_dict(record) for record in records]
    result = {profile.user_id: profile for profile in profiles}
    if len(result) != len(profiles):
        raise ConfigurationError("preferences contain duplicate user_id values")
    return result


def load_rules(path: str | Path | None) -> tuple[RoutingRule, ...]:
    if path is None:
        return ()
    payload = _read_json(path)
    records = payload.get("rules", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ConfigurationError("rules must be a list or an object with a rules list")
    rules = tuple(RoutingRule.from_dict(record) for record in records)
    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise ConfigurationError("rules contain duplicate names")
    return rules


def request_from_dict(data: Mapping[str, Any]) -> RouteRequest:
    try:
        return RouteRequest(
            query=str(data["query"]),
            user_id=str(data.get("user_id", "default")),
            expected_output_tokens=int(data.get("expected_output_tokens", 256)),
            required_capabilities=_string_list(data, "required_capabilities"),
            max_cost_usd=(float(data["max_cost_usd"]) if data.get("max_cost_usd") is not None else None),
            max_latency_ms=(
                float(data["max_latency_ms"])
                if data.get("max_latency_ms") is not None
                else None
            ),
            region=str(data["region"]) if data.get("region") else None,
            needs_tools=_boolean_field(data, "needs_tools", False),
            needs_json=_boolean_field(data, "needs_json", False),
            sensitivity=str(data.get("sensitivity", "normal")),
            task_hint=str(data["task_hint"]) if data.get("task_hint") else None,
            context_tokens=(
                int(data["context_tokens"]) if data.get("context_tokens") is not None else None
            ),
            request_id=str(data["request_id"]) if data.get("request_id") else uuid4().hex,
            metadata=dict(data.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid route request: {exc}") from exc


def load_requests(path: str | Path) -> tuple[RouteRequest, ...]:
    source = Path(path)
    requests: list[RouteRequest] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ConfigurationError("request line must be a JSON object")
                    requests.append(request_from_dict(payload))
                except (json.JSONDecodeError, ConfigurationError) as exc:
                    raise ConfigurationError(
                        f"Invalid request at {source}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise ConfigurationError(f"Cannot read request file {source}: {exc}") from exc
    if not requests:
        raise ConfigurationError("request file contains no requests")
    return tuple(requests)
