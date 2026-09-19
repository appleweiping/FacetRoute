from __future__ import annotations

import json
from collections.abc import Callable

import pytest

import facetroute.persistence as persistence_module
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
from facetroute.persistence import AtomicJsonStore, _atomic_write_bytes_bundle


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
    log.append(_event(event_id="event-2", reward=0.4, success=False, latency_ms=300, cost_usd=0.2))
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


def test_atomic_json_store_wraps_invalid_utf8(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"\xff")

    with pytest.raises(PersistenceError, match="Cannot read"):
        AtomicJsonStore(path).load()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"query": "x", "needs_tools": "false"}, "needs_tools"),
        ({"query": "x", "required_capabilities": ["text", 7]}, "required_capabilities"),
        ({"query": "x", "region": 9}, "region"),
        ({"query": "x", "max_cost_usd": True}, "max_cost_usd"),
        ({"query": "x", "context_tokens": True}, "context_tokens"),
        ({"query": "x", "request_id": 9}, "request_id"),
        ({"query": "x", "metadata": []}, "metadata"),
        ({"query": 9}, "query"),
    ],
)
def test_request_loader_rejects_malformed_typed_fields(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        request_from_dict(payload)


def test_request_loader_accepts_all_optional_typed_fields() -> None:
    request = request_from_dict(
        {
            "query": "x",
            "user_id": "u",
            "required_capabilities": ["text"],
            "max_latency_ms": 123,
            "region": "US",
            "needs_json": True,
            "sensitivity": "sensitive",
            "task_hint": "math",
            "context_tokens": 12,
            "request_id": "r",
        }
    )
    assert request.request_id == "r"
    assert request.context_tokens == 12
    assert request.max_latency_ms == 123


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{}", "empty"),
        ("[]", "empty"),
        ('{"models": 1}', "models list"),
        ("not json", "Cannot read"),
    ],
)
def test_model_loader_rejects_bad_catalog_shape(tmp_path, content: str, message: str) -> None:
    path = tmp_path / "models.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_models(path)


@pytest.mark.parametrize(
    ("loader", "content", "message"),
    [
        (load_preferences, '{"profiles": 1}', "profiles list"),
        (load_rules, '{"rules": 1}', "rules list"),
    ],
)
def test_optional_configuration_loader_rejects_bad_container(
    tmp_path, loader, content: str, message: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        loader(path)


def test_atomic_json_store_failed_write_preserves_previous_state(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = AtomicJsonStore(path)
    store.save({"version": 1})

    with pytest.raises(PersistenceError, match="Cannot write"):
        store.save({"unsupported": object()})

    assert store.load() == {"version": 1}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


@pytest.mark.parametrize("stage", ["fdopen", "write", "flush", "fsync", "replace"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_atomic_json_store_failures_clean_temporary_and_preserve_target(
    tmp_path, monkeypatch, stage: str, error_type: type[BaseException]
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old-state")
    store = AtomicJsonStore(path)
    original_fdopen = persistence_module.os.fdopen

    class InterruptingWriter:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.handle.close()

        def write(self, value: bytes) -> int:
            if stage == "write":
                raise error_type("injected write failure")
            return self.handle.write(value)

        def flush(self) -> None:
            if stage == "flush":
                raise error_type("injected flush failure")
            self.handle.flush()

        def fileno(self) -> int:
            return self.handle.fileno()

    if stage == "fdopen":

        def interrupt_fdopen(*args, **kwargs):
            raise error_type("injected fdopen failure")

        monkeypatch.setattr(persistence_module.os, "fdopen", interrupt_fdopen)
    elif stage in {"write", "flush"}:

        def interrupting_fdopen(*args, **kwargs):
            return InterruptingWriter(original_fdopen(*args, **kwargs))

        monkeypatch.setattr(persistence_module.os, "fdopen", interrupting_fdopen)
    elif stage == "fsync":

        def interrupt_fsync(descriptor: int) -> None:
            raise error_type("injected fsync failure")

        monkeypatch.setattr(persistence_module.os, "fsync", interrupt_fsync)
    else:

        def interrupt_replace(source, destination) -> None:
            raise error_type("injected replace failure")

        monkeypatch.setattr(persistence_module.os, "replace", interrupt_replace)

    expected_error = PersistenceError if error_type is OSError else KeyboardInterrupt
    with pytest.raises(expected_error):
        store.save({"version": 2})

    assert path.read_bytes() == b"old-state"
    assert list(tmp_path.glob(".state.json.*")) == []


def test_atomic_json_store_does_not_close_a_reused_descriptor_on_replace_failure(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old-state")
    other_path = tmp_path / "other.bin"
    other_descriptor: int | None = None

    def fail_after_reuse(source, destination) -> None:
        nonlocal other_descriptor
        del source, destination
        other_descriptor = persistence_module.os.open(
            other_path,
            persistence_module.os.O_CREAT
            | persistence_module.os.O_WRONLY
            | persistence_module.os.O_TRUNC,
        )
        raise OSError("injected replacement failure")

    monkeypatch.setattr(persistence_module.os, "replace", fail_after_reuse)
    with pytest.raises(PersistenceError):
        AtomicJsonStore(path).save({"version": 2})

    assert other_descriptor is not None
    try:
        assert persistence_module.os.write(other_descriptor, b"still-open") == 10
    finally:
        persistence_module.os.close(other_descriptor)
    assert path.read_bytes() == b"old-state"
    assert list(tmp_path.glob(".state.json.*")) == []


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
@pytest.mark.parametrize("after_replace", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_atomic_byte_bundle_rolls_back_every_replacement_boundary(
    tmp_path,
    monkeypatch,
    fail_at: int,
    after_replace: bool,
    error_type: type[BaseException],
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    real_replace = persistence_module.os.replace
    calls = 0

    def fail_once(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            if after_replace:
                real_replace(source, destination)
            raise error_type("injected bundle replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(persistence_module.os, "replace", fail_once)
    with pytest.raises(error_type):
        _atomic_write_bytes_bundle({first: b"new-first", second: b"new-second"})

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.json", "second.json"]


def test_atomic_byte_bundle_rejects_parent_child_destinations_without_side_effects(
    tmp_path,
) -> None:
    parent = tmp_path / "artifact"
    child = parent / "report.json"

    with pytest.raises(ValueError, match="destinations collide"):
        _atomic_write_bytes_bundle({parent: b"model", child: b"report"})

    assert not parent.exists()


@pytest.mark.parametrize("after_replace", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_atomic_byte_bundle_removes_new_outputs_after_partial_install(
    tmp_path,
    monkeypatch,
    after_replace: bool,
    error_type: type[BaseException],
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    real_replace = persistence_module.os.replace
    calls = 0

    def fail_second_install(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            if after_replace:
                real_replace(source, destination)
            raise error_type("injected second-install failure")
        real_replace(source, destination)

    monkeypatch.setattr(persistence_module.os, "replace", fail_second_install)
    with pytest.raises(error_type):
        _atomic_write_bytes_bundle({first: b"first", second: b"second"})

    assert not first.exists()
    assert not second.exists()
    assert list(tmp_path.iterdir()) == []


def test_preference_store_upsert_and_round_trip(tmp_path) -> None:
    store = PreferenceStore(tmp_path / "profiles.json")
    store.upsert(UserPreferences("u", quality_weight=1, cost_weight=0, latency_weight=0))
    store.upsert(UserPreferences("v", preferred_models=frozenset({"m"})))
    assert set(store.load_all()) == {"u", "v"}
    assert store.get("v").preferred_models == frozenset({"m"})  # type: ignore[union-attr]


def test_preference_store_wraps_invalid_profile_data(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": "not-a-list"}), encoding="utf-8")
    with pytest.raises(PersistenceError, match="profiles must be a list"):
        PreferenceStore(path).load_all()


def test_load_models_rejects_duplicates(
    tmp_path, make_model: Callable[..., ModelCandidate]
) -> None:
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
