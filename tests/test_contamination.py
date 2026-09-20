from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from facetroute import ContaminationReport, contamination
from facetroute import audit_contamination as public_audit_contamination
from facetroute.contamination import audit_contamination, contamination_json
from facetroute.contamination_cli import main
from facetroute.errors import ConfigurationError


def _set(path: Path, rows: list[tuple[str, list[float]]], **overrides: object) -> bytes:
    data: dict[str, object] = {
        "schema": "facet-embedding-set-v1",
        "model_id": "fixture-encoder",
        "model_revision": "r1",
        "dimension": 2,
        "records": [{"id": record_id, "embedding": vector} for record_id, vector in rows],
    }
    data.update(overrides)
    raw = (json.dumps(data, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_hand_oracle_tie_hash_and_privacy(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    train_raw = _set(train, [("z", [3, 4]), ("a", [3, 4]), ("opposite", [-1, 0])])
    eval_raw = _set(evaluation, [("near", [1, 0]), ("opposed", [-1, 0])])
    report = audit_contamination(train, evaluation, threshold=0.6)
    assert isinstance(
        public_audit_contamination(train, evaluation, threshold=0.6), ContaminationReport
    )
    assert [(hit.evaluation_id, hit.training_id, hit.cosine_similarity) for hit in report.hits] == [
        ("near", "a", 0.6),
        ("opposed", "opposite", 1.0),
    ]
    assert report.coordinate_work == 12
    assert report.training_sha256 == hashlib.sha256(train_raw).hexdigest()
    assert report.evaluation_sha256 == hashlib.sha256(eval_raw).hexdigest()
    output = contamination_json(report)
    assert output.endswith(b"\n")
    assert json.loads(output)["schema"] == "facet-contamination-audit-v1"
    assert b"embedding" not in output


def test_threshold_and_negative_cosine(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 0])])
    _set(evaluation, [("b", [-1, 0])])
    assert audit_contamination(train, evaluation).hits == ()
    assert audit_contamination(train, evaluation, threshold=-1).hits[0].cosine_similarity == -1


def test_identical_non_axis_vectors_meet_inclusive_one(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 1])])
    _set(evaluation, [("b", [1, 1])])
    assert audit_contamination(train, evaluation, threshold=1).hits == (
        contamination.ContaminationHit("b", "a", 1.0),
    )


def test_large_and_subnormal_finite_vectors(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("large", [1e308, 1e308]), ("tiny", [5e-324, 0])])
    _set(evaluation, [("q-large", [1e308, 1e308]), ("q-tiny", [5e-324, 0])])
    assert [
        (hit.evaluation_id, hit.training_id)
        for hit in audit_contamination(train, evaluation, threshold=1).hits
    ] == [("q-large", "large"), ("q-tiny", "tiny")]


def test_cli_and_error_status(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 0])])
    _set(evaluation, [("b", [1, 0])])
    assert main(["--training", str(train), "--evaluation", str(evaluation)]) == 0
    assert json.loads(capfd.readouterr().out)["hits"][0]["training_id"] == "a"
    assert main(["--training", str(train), "--evaluation", str(train)]) == 2
    assert "must differ" in capfd.readouterr().err


@pytest.mark.parametrize(
    "threshold", [True, "0.95", float("nan"), float("inf"), 1.01, -1.01, 10**400]
)
def test_invalid_threshold(tmp_path: Path, threshold: object) -> None:
    with pytest.raises(ConfigurationError, match="threshold"):
        audit_contamination(tmp_path / "a", tmp_path / "b", threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"schema": "wrong"}, "schema"),
        ({"dimension": True}, "metadata"),
        ({"dimension": 0}, "metadata"),
        ({"dimension": 2049}, "metadata"),
        ({"model_id": "bad id"}, "metadata"),
        ({"model_revision": ""}, "metadata"),
        ({"records": []}, "metadata"),
        ({"records": [{}]}, "record"),
        ({"records": [{"id": "a", "embedding": [1]}]}, "record"),
        ({"records": [{"id": "a", "embedding": [True, 0]}]}, "record"),
        ({"records": [{"id": "a", "embedding": [0, 0]}]}, "norm"),
        (
            {"records": [{"id": "a", "embedding": [1, 0]}, {"id": "a", "embedding": [0, 1]}]},
            "record",
        ),
        ({"records": [{"id": "bad id", "embedding": [1, 0]}]}, "record"),
    ],
)
def test_invalid_schema_and_vectors(
    tmp_path: Path, overrides: dict[str, object], fragment: str
) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 0])], **overrides)
    _set(evaluation, [("b", [1, 0])])
    with pytest.raises(ConfigurationError, match=fragment):
        audit_contamination(train, evaluation)


def test_nonfinite_overflow_and_strict_json(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(evaluation, [("b", [1, 0])])
    for malformed in (
        b'{"schema":"facet-embedding-set-v1","schema":"x"}',
        b"\xff",
    ):
        train.write_bytes(malformed)
        with pytest.raises(ConfigurationError, match="strict JSON"):
            audit_contamination(train, evaluation)
    _set(train, [("a", [float("nan"), 0])])
    with pytest.raises(ConfigurationError, match="strict JSON"):
        audit_contamination(train, evaluation)
    _set(train, [("a", [10**400, 0])])
    with pytest.raises(ConfigurationError, match="coordinate"):
        audit_contamination(train, evaluation)


def test_mismatched_model_revision_and_dimension(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 0])])
    _set(evaluation, [("b", [1, 0])], model_revision="r2")
    with pytest.raises(ConfigurationError, match="identity"):
        audit_contamination(train, evaluation)
    _set(evaluation, [("b", [1, 0, 0])], dimension=3)
    with pytest.raises(ConfigurationError, match="dimension"):
        audit_contamination(train, evaluation)


def test_preflight_and_read_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    _set(train, [("a", [1, 0])])
    _set(evaluation, [("b", [1, 0])])
    monkeypatch.setattr(contamination, "_MAX_WORK", 1)
    with pytest.raises(ConfigurationError, match="work limit"):
        audit_contamination(train, evaluation)
    monkeypatch.setattr(contamination, "_MAX_COORDINATES", 1)
    with pytest.raises(ConfigurationError, match="metadata"):
        audit_contamination(train, evaluation)
    monkeypatch.setattr(contamination, "_MAX_FILE_BYTES", 1)
    with pytest.raises(ConfigurationError, match="byte limit"):
        audit_contamination(train, evaluation)
    with pytest.raises(ConfigurationError, match="cannot read"):
        audit_contamination(tmp_path / "missing", evaluation)


def test_unresolvable_path_is_domain_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_resolve = Path.resolve

    def resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path.name == "loop":
            raise RuntimeError("synthetic symlink loop")
        return original_resolve(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ConfigurationError, match="cannot resolve"):
        audit_contamination(tmp_path / "loop", tmp_path / "other")


def test_training_input_is_not_mutated(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    original = _set(train, [("a", [1, 0]), ("b", [0, 1])])
    _set(evaluation, [("q", [0, 1])])
    audit_contamination(train, evaluation)
    assert train.read_bytes() == original
