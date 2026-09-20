from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from facetroute import contamination_exclusions as linking
from facetroute import plan_contamination_exclusions
from facetroute.contamination_exclusions_cli import main
from facetroute.errors import ConfigurationError


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "".join(
            json.dumps({"id": key, "question": prompt, "answer": answer}) + "\n"
            for key, prompt, answer in (
                ("a", "Shared secret prompt", "4"),
                ("b", "Shared secret prompt", "9"),
                ("c", "Unrelated secret prompt", "2"),
            )
        ),
        encoding="utf-8",
    )
    training = tmp_path / "training.json"
    evaluation = tmp_path / "evaluation.json"
    metadata: dict[str, object] = {
        "schema": "facet-embedding-set-v1",
        "model_id": "declared-encoder",
        "model_revision": "r1",
        "dimension": 2,
    }
    training.write_text(
        json.dumps({**metadata, "records": [{"id": "train", "embedding": [1, 0]}]}) + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        json.dumps(
            {
                **metadata,
                "records": [
                    {"id": "a", "embedding": [1, 0]},
                    {"id": "b", "embedding": [0, 1]},
                    {"id": "c", "embedding": [-1, 0]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source, training, evaluation


def test_hand_join_duplicate_prompt_and_private_evidence(tmp_path: Path) -> None:
    source, training, evaluation = _inputs(tmp_path)
    plan = plan_contamination_exclusions(source, training, evaluation, threshold=0.9)
    digest = hashlib.sha256(b"Shared secret prompt").hexdigest()
    assert plan.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan.audit.evaluation_sha256 == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert plan.matched_embedding_records == 1
    # The public-score audit excludes by normalized prompt, not by record ID.
    assert plan.excluded_source_records == 2
    assert plan.prompt_sha256 == (digest,)
    exclusions = plan.exclusions_jsonl()
    assert exclusions == (json.dumps({"prompt_sha256": digest}, sort_keys=True) + "\n").encode()
    evidence = json.loads(plan.evidence_json())
    assert evidence["schema"] == "facet-contamination-exclusions-v1"
    assert evidence["exclusions_sha256"] == hashlib.sha256(exclusions).hexdigest()
    assert evidence["training_sha256"] == hashlib.sha256(training.read_bytes()).hexdigest()
    for sensitive in (b"secret", b'"answer"', b'"embedding"'):
        assert sensitive not in exclusions + plan.evidence_json()


def test_empty_exclusions_are_explicit_and_hashed(tmp_path: Path) -> None:
    source, training, evaluation = _inputs(tmp_path)
    plan = plan_contamination_exclusions(source, training, evaluation, threshold=1)
    assert plan.matched_embedding_records == 1
    evaluation.write_text(
        evaluation.read_text(encoding="utf-8").replace("[1, 0]", "[0, 1]"),
        encoding="utf-8",
    )
    no_hits = plan_contamination_exclusions(source, training, evaluation, threshold=1)
    assert no_hits.exclusions_jsonl() == b""
    assert (
        json.loads(no_hits.evidence_json())["exclusions_sha256"] == hashlib.sha256(b"").hexdigest()
    )


def test_exact_id_join_and_snapshot_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, training, evaluation = _inputs(tmp_path)
    original = evaluation.read_bytes()
    evaluation.write_bytes(original.replace(b'"c"', b'"d"'))
    with pytest.raises(ConfigurationError, match="IDs must match"):
        plan_contamination_exclusions(source, training, evaluation)
    evaluation.write_bytes(original)
    real_audit = linking.audit_contamination

    def changing_audit(*args: object, **kwargs: object) -> object:
        result = real_audit(*args, **kwargs)  # type: ignore[arg-type]
        evaluation.write_bytes(original + b" ")
        return result

    monkeypatch.setattr(linking, "audit_contamination", changing_audit)
    with pytest.raises(ConfigurationError, match="changed"):
        plan_contamination_exclusions(source, training, evaluation)


def test_cli_create_only_and_no_mutation(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    source, training, evaluation = _inputs(tmp_path)
    inputs = tuple(path.read_bytes() for path in (source, training, evaluation))
    output = tmp_path / "exclusions.jsonl"
    argv = [
        "--source",
        str(source),
        "--training",
        str(training),
        "--evaluation",
        str(evaluation),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    evidence = json.loads(capfd.readouterr().out)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == evidence["exclusions_sha256"]
    assert main(argv) == 2
    assert "already exists" in capfd.readouterr().err
    assert inputs == tuple(path.read_bytes() for path in (source, training, evaluation))


@pytest.mark.parametrize("problem", ["same-path", "mt-bench", "too-large"])
def test_fail_closed_source(tmp_path: Path, problem: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source, training, evaluation = _inputs(tmp_path)
    if problem == "same-path":
        with pytest.raises(ConfigurationError, match="inputs must differ"):
            plan_contamination_exclusions(training, training, evaluation)
    elif problem == "mt-bench":
        source.write_text('{"question_id":"a","turns":["hello"]}\n', encoding="utf-8")
        with pytest.raises(ConfigurationError, match="only MMLU and GSM8K"):
            plan_contamination_exclusions(source, training, evaluation)
    else:
        monkeypatch.setattr(linking, "_MAX_SOURCE_BYTES", 1)
        with pytest.raises(ConfigurationError, match="byte limit"):
            plan_contamination_exclusions(source, training, evaluation)
