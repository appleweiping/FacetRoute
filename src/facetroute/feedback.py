"""Append-only offline feedback events and aggregate reporting."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from ._json import loads_strict
from .errors import ConfigurationError, PersistenceError


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """A provider-independent observation that can be replayed into a policy."""

    request_id: str
    user_id: str
    model_id: str
    reward: float
    policy: str
    context_vector: tuple[float, ...] = ()
    success: bool = True
    latency_ms: float | None = None
    cost_usd: float | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("user_id", self.user_id),
            ("model_id", self.model_id),
            ("policy", self.policy),
            ("event_id", self.event_id),
        ):
            if not value.strip():
                raise ConfigurationError(f"{name} cannot be empty")
        if not math.isfinite(self.reward) or not 0 <= self.reward <= 1:
            raise ConfigurationError("reward must be finite and between 0 and 1")
        for name, optional_value in (
            ("latency_ms", self.latency_ms),
            ("cost_usd", self.cost_usd),
        ):
            if optional_value is not None and (
                not math.isfinite(optional_value) or optional_value < 0
            ):
                raise ConfigurationError(f"{name} must be finite and non-negative")
        if any(not math.isfinite(value) for value in self.context_vector):
            raise ConfigurationError("context_vector must contain only finite values")
        if not isinstance(self.success, bool):
            raise ConfigurationError("success must be a boolean")
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigurationError("timestamp must be ISO 8601") from exc
        object.__setattr__(self, "tags", dict(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "model_id": self.model_id,
            "reward": self.reward,
            "policy": self.policy,
            "context_vector": list(self.context_vector),
            "success": self.success,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeedbackEvent:
        try:
            success = data.get("success", True)
            if not isinstance(success, bool):
                raise ConfigurationError("success must be a boolean")
            return cls(
                request_id=str(data["request_id"]),
                user_id=str(data["user_id"]),
                model_id=str(data["model_id"]),
                reward=float(data["reward"]),
                policy=str(data["policy"]),
                context_vector=tuple(float(item) for item in data.get("context_vector", [])),
                success=success,
                latency_ms=(float(data["latency_ms"]) if data.get("latency_ms") is not None else None),
                cost_usd=(float(data["cost_usd"]) if data.get("cost_usd") is not None else None),
                tags={str(key): str(value) for key, value in dict(data.get("tags", {})).items()},
                event_id=str(data.get("event_id", uuid4().hex)),
                timestamp=str(data.get("timestamp", _utc_now())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid feedback event: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ModelFeedbackSummary:
    count: int
    average_reward: float
    success_rate: float
    average_latency_ms: float | None
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "average_reward": self.average_reward,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "total_cost_usd": self.total_cost_usd,
        }


class FeedbackLog:
    """Newline-delimited JSON feedback with duplicate-event protection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def append(self, event: FeedbackEvent) -> None:
        with self._lock:
            if any(existing.event_id == event.event_id for existing in self.iter_events()):
                raise PersistenceError(f"duplicate feedback event_id: {event.event_id}")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False))
                    handle.write("\n")
                    handle.flush()
            except OSError as exc:
                raise PersistenceError(f"Cannot append feedback to {self.path}: {exc}") from exc

    def iter_events(self) -> Iterator[FeedbackEvent]:
        if not self.path.exists():
            return
        seen_event_ids: set[str] = set()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = loads_strict(line)
                        event = FeedbackEvent.from_dict(payload)
                        if event.event_id in seen_event_ids:
                            raise PersistenceError(
                                f"Duplicate feedback event_id at {self.path}:{line_number}: "
                                f"{event.event_id}"
                            )
                        seen_event_ids.add(event.event_id)
                        yield event
                    except (ValueError, ConfigurationError) as exc:
                        raise PersistenceError(
                            f"Invalid feedback at {self.path}:{line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise PersistenceError(f"Cannot read feedback log {self.path}: {exc}") from exc

    def summarize(self) -> dict[str, ModelFeedbackSummary]:
        groups: dict[str, list[FeedbackEvent]] = {}
        for event in self.iter_events():
            groups.setdefault(event.model_id, []).append(event)
        result: dict[str, ModelFeedbackSummary] = {}
        for model_id, events in sorted(groups.items()):
            latencies = [event.latency_ms for event in events if event.latency_ms is not None]
            result[model_id] = ModelFeedbackSummary(
                count=len(events),
                average_reward=sum(event.reward for event in events) / len(events),
                success_rate=sum(1 for event in events if event.success) / len(events),
                average_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
                total_cost_usd=sum(event.cost_usd or 0.0 for event in events),
            )
        return result
