from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from facetroute import (
    ConfigurationError,
    FeedbackEvent,
    FeedbackLog,
    ModelCandidate,
    PreferenceStore,
    UserPreferences,
)
from facetroute.config import (
    load_models,
    load_preferences,
    load_requests,
    load_rules,
    request_from_dict,
)
from facetroute.errors import PersistenceError
from facetroute.persistence import AtomicJsonStore


def _event(**overrides: object) -> FeedbackEvent:
    values: dict[str, object] = {
        "request_id": "r",
        "user_id": "u",
        "model_id": "m",
        "reward": 0.8,
        "policy": "rule",
        "event_id": "event-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    values.update(overrides)
    return FeedbackEvent(**values)  # type: ignore[arg-type]


def test_feedback_validates_reward_and_timestamp() -> None:
    with pytest.raises(ConfigurationError, match="reward"):
        _event(reward=-0.1)
    with pytest.raises(ConfigurationError, match="timestamp"):
        _event(timestamp="yesterday")
    with pytest.raises(ConfigurationError, match="success"):
        _event(success="false")


def test_feedback_round_trip_preserves_fields() -> None:
    event = _event(context_vector=(1.0, 0.2), tags={"task": "code"})
    assert FeedbackEvent.from_dict(event.to_dict()) == event


def test_feedback_json_rejects_string_boolean() -> None:
    payload = _event().to_dict()
    payload["success"] = "false"
    with pytest.raises(ConfigurationError, match="boolean"):
        FeedbackEvent.from_dict(payload)


def test_feedback_log_append_and_iterate(tmp_path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.append(_event())
    log.append(_event(event_id="event-2", model_id="other", reward=0.4))
    assert [event.event_id for event in log.iter_events()] == ["event-1", "event-2"]


def test_feedback_log_rejects_duplicate_event(tmp_path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.append(_event())
    with pytest.raises(PersistenceError, match="duplicate"):
        log.append(_event())


def test_feedback_log_reports_malformed_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(PersistenceError, match=":1"):
        list(FeedbackLog(path).iter_events())


def test_feedback_log_rejects_duplicate_ids_already_on_disk(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    serialized = json.dumps(_event().to_dict())
    path.write_text(f"{serialized}\n{serialized}\n", encoding="utf-8")
    with pytest.raises(PersistenceError, match=r"Duplicate.*:2"):
        list(FeedbackLog(path).iter_events())


def test_feedback_summary_aggregates_by_model(tmp_path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.append(_event(latency_ms=100, cost_usd=0.1))
    log.append(
        _event(event_id="event-2", reward=0.4, success=False, latency_ms=300, cost_usd=0.2)
    )
    summary = log.summarize()["m"]
    assert summary.count == 2
    assert summary.average_reward == pytest.approx(0.6)
    assert summary.success_rate == 0.5
    assert summary.average_latency_ms == 200
    assert summary.total_cost_usd == pytest.approx(0.3)


def test_atomic_json_store_round_trip_and_missing_default(tmp_path) -> None:
    store = AtomicJsonStore(tmp_path / "nested" / "state.json")
    assert store.load({"new": True}) == {"new": True}
    store.save({"value": [1, 2]})
    assert store.load() == {"value": [1, 2]}


def test_atomic_json_store_reports_invalid_json(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(PersistenceError, match="Cannot read"):
        AtomicJsonStore(path).load()


def test_atomic_json_store_failed_write_preserves_previous_state(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = AtomicJsonStore(path)
    store.save({"version": 1})

    with pytest.raises(PersistenceError, match="Cannot write"):
        store.save({"unsupported": object()})

    assert store.load() == {"version": 1}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_preference_store_upsert_and_round_trip(tmp_path) -> None:
    store = PreferenceStore(tmp_path / "profiles.json")
    store.upsert(UserPreferences("u", quality_weight=1, cost_weight=0, latency_weight=0))
    store.upsert(UserPreferences("v", preferred_models=frozenset({"m"})))
    assert set(store.load_all()) == {"u", "v"}
    assert store.get("v").preferred_models == frozenset({"m"})  # type: ignore[union-attr]


def test_preference_store_wraps_invalid_profile_data(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps({"schema_version": 1, "profiles": "not-a-list"}), encoding="utf-8"
    )
    with pytest.raises(PersistenceError, match="profiles must be a list"):
        PreferenceStore(path).load_all()


def test_load_models_rejects_duplicates(tmp_path, make_model: Callable[..., ModelCandidate]) -> None:
    path = tmp_path / "models.json"
    model = make_model("same").to_dict()
    path.write_text(json.dumps([model, model]), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate"):
        load_models(path)


def test_load_preferences_and_rules_from_wrapped_objects(tmp_path) -> None:
    preferences = tmp_path / "preferences.json"
    preferences.write_text(json.dumps({"profiles": [{"user_id": "u"}]}), encoding="utf-8")
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"rules": [{"name": "r", "prefer_models": ["m"]}]}), encoding="utf-8"
    )
    assert list(load_preferences(preferences)) == ["u"]
    assert load_rules(rules)[0].name == "r"


def test_load_requests_reports_line_number(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text('{"query":"ok"}\n[]\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match=":2"):
        load_requests(path)


def test_load_requests_preserves_explicit_request_id(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text('{"query":"ok","request_id":"fixed"}\n', encoding="utf-8")
    assert load_requests(path)[0].request_id == "fixed"


def test_request_json_rejects_string_booleans_and_capability_strings() -> None:
    with pytest.raises(ConfigurationError, match="boolean"):
        request_from_dict({"query": "hello", "needs_tools": "false"})
    with pytest.raises(ConfigurationError, match="array of strings"):
        request_from_dict({"query": "hello", "required_capabilities": "text"})
