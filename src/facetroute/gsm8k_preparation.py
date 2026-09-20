"""Offline few-shot preparation of caller-pinned GSM8K-shaped JSONL.

This local protocol does not reproduce any external tokenizer, provider,
published model response, or official benchmark result.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

from ._json import loads_strict
from .benchmark_formats import BenchmarkExample, BenchmarkFormat
from .errors import ConfigurationError
from .public_scores import PublicScoreProvenance, _prompt

PROTOCOL = "facet-gsm8k-jsonl-fewshot-v1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TRAIN_ROWS = 64
MAX_TEST_ROWS = 500
MAX_LINE_BYTES = 16 * 1024
MAX_FIELD_BYTES = 4096
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_FINAL = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\Z")


def _json(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ConfigurationError("GSM8K preparation artifact is not strict JSON") from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class GSM8KJSONLPreparationPlan:
    """Exact declared source identities and a deterministic prompt budget."""

    train: PublicScoreProvenance
    test: PublicScoreProvenance
    requested_shots: int = 8
    max_prompt_bytes: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.train, PublicScoreProvenance) or not isinstance(
            self.test, PublicScoreProvenance
        ):
            raise ConfigurationError("train/test require declared source URI, license and SHA-256")
        if type(self.requested_shots) is not int or not 0 <= self.requested_shots <= 8:
            raise ConfigurationError("requested_shots must be an integer in [0, 8]")
        if type(self.max_prompt_bytes) is not int or not 32 <= self.max_prompt_bytes <= 32 * 1024:
            raise ConfigurationError("max_prompt_bytes must be an integer in [32, 32768]")

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_shots": self.requested_shots,
            "max_prompt_bytes": self.max_prompt_bytes,
            "train": {
                "uri": self.train.source_uri,
                "license": self.train.license_id,
                "sha256": self.train.source_sha256,
            },
            "test": {
                "uri": self.test.source_uri,
                "license": self.test.license_id,
                "sha256": self.test.source_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class _Row:
    question: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return {"question": self.question, "answer": self.answer}


def _field(value: object, label: str, *, answer: bool) -> str:
    if type(value) is not str or not value.strip():
        raise ConfigurationError(f"{label} must be bounded non-empty text")
    text = value.strip()
    try:
        if len(text.encode("utf-8", "strict")) > MAX_FIELD_BYTES:
            raise ConfigurationError(f"{label} exceeds field byte limit")
    except UnicodeError as error:
        raise ConfigurationError(f"{label} must be strict UTF-8") from error
    if any(
        (ord(character) < 32 and not (answer and character == "\n")) or ord(character) == 127
        for character in text
    ):
        raise ConfigurationError(f"{label} contains unsafe controls")
    return text


def _final_answer(answer: str, label: str) -> None:
    if answer.count("####") != 1:
        raise ConfigurationError(f"{label} requires exactly one #### final-number marker")
    number = answer.rsplit("####", 1)[1].strip()
    if _FINAL.fullmatch(number) is None:
        raise ConfigurationError(f"{label} has invalid final number")


def _rows(raw: bytes, *, label: str, maximum: int) -> tuple[_Row, ...]:
    if type(raw) is not bytes or len(raw) > MAX_SOURCE_BYTES:
        raise ConfigurationError(f"{label} must be a byte snapshot within {MAX_SOURCE_BYTES} bytes")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise ConfigurationError(f"{label} must be strict UTF-8 JSONL") from error
    if text.startswith("\ufeff"):
        raise ConfigurationError(f"{label} must not contain a UTF-8 BOM")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ConfigurationError(f"{label} contains no rows or a blank line")
    if len(lines) > maximum:
        raise ConfigurationError(f"{label} exceeds {maximum} rows")
    result: list[_Row] = []
    for index, line in enumerate(lines, 1):
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise ConfigurationError(f"{label} line {index} exceeds byte limit")
        try:
            record = loads_strict(line)
        except (ValueError, TypeError) as error:
            raise ConfigurationError(f"{label} line {index} is not strict JSON") from error
        if not isinstance(record, dict) or set(record) != {"question", "answer"}:
            raise ConfigurationError(f"{label} line {index} schema requires question and answer")
        question = _field(record["question"], f"{label} question {index}", answer=False)
        answer = _field(record["answer"], f"{label} answer {index}", answer=True)
        _final_answer(answer, f"{label} answer {index}")
        result.append(_Row(question, answer))
    return tuple(result)


def _stem(row: _Row) -> str:
    return " ".join(unicodedata.normalize("NFKC", row.question).casefold().split())


def _unique(rows: tuple[_Row, ...], label: str) -> set[str]:
    stems = {_stem(row) for row in rows}
    if len(stems) != len(rows):
        raise ConfigurationError(f"{label} contains duplicate normalized questions")
    return stems


def _context(rows: tuple[_Row, ...]) -> str | None:
    if not rows:
        return None
    return "\n\n".join(
        ["Worked examples (answers shown only for examples):"]
        + [f"Question: {row.question}\nAnswer: {row.answer}" for row in rows]
    )


def _examples(rows: tuple[_Row, ...], context: str | None) -> tuple[BenchmarkExample, ...]:
    return tuple(
        BenchmarkExample(
            example_id=f"gsm8k-{index:04d}",
            format=BenchmarkFormat.GSM8K,
            prompt=row.question,
            answer=row.answer,
            few_shot_context=context,
        )
        for index, row in enumerate(rows, 1)
    )


@dataclass(frozen=True, slots=True)
class PreparedGSM8KJSONL:
    """Private JSONL and replay evidence, including per-prompt digests."""

    plan: GSM8KJSONLPreparationPlan
    train_rows: int
    test_rows: int
    selected_shots: int
    train_canonical_sha256: str
    test_canonical_sha256: str
    examples: tuple[BenchmarkExample, ...]

    def benchmark_jsonl(self) -> bytes:
        """Return public-score-compatible JSONL containing private test gold labels."""
        raw = b"".join(_json(example.to_dict()) + b"\n" for example in self.examples)
        if len(raw) > MAX_OUTPUT_BYTES:
            raise ConfigurationError("prepared GSM8K artifact exceeds output byte limit")
        return raw

    def evidence_json(self) -> bytes:
        """Return replay digests without raw text; hashes do not guarantee privacy."""
        report = {
            "protocol": PROTOCOL,
            "plan": self.plan.to_dict(),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "selected_shots": self.selected_shots,
            "train_canonical_sha256": self.train_canonical_sha256,
            "test_canonical_sha256": self.test_canonical_sha256,
            "artifact_sha256": _sha(self.benchmark_jsonl()),
            "provider_prompt_sha256": [
                {"id": example.example_id, "sha256": _sha(_prompt(example)[0].encode("utf-8"))}
                for example in self.examples
            ],
        }
        return _json(report) + b"\n"


def prepare_gsm8k_jsonl(
    train_source: bytes,
    test_source: bytes,
    plan: GSM8KJSONLPreparationPlan,
) -> PreparedGSM8KJSONL:
    """Prepare answer-free provider prompts without calling a model or provider."""
    if not isinstance(plan, GSM8KJSONLPreparationPlan):
        raise ConfigurationError("GSM8K preparation requires an explicit plan")
    if type(train_source) is not bytes or type(test_source) is not bytes:
        raise ConfigurationError("GSM8K sources must be immutable byte snapshots")
    if len(train_source) > MAX_SOURCE_BYTES or len(test_source) > MAX_SOURCE_BYTES:
        raise ConfigurationError(
            f"GSM8K source must be a byte snapshot within {MAX_SOURCE_BYTES} bytes"
        )
    if (
        _sha(train_source) != plan.train.source_sha256
        or _sha(test_source) != plan.test.source_sha256
    ):
        raise ConfigurationError("GSM8K source SHA-256 differs from caller pin")
    train = _rows(train_source, label="train", maximum=MAX_TRAIN_ROWS)
    test = _rows(test_source, label="test", maximum=MAX_TEST_ROWS)
    if plan.requested_shots > len(train):
        raise ConfigurationError("requested_shots exceeds available train rows")
    if _unique(train, "train") & _unique(test, "test"):
        raise ConfigurationError("train/test normalized question stems overlap")
    for selected in range(plan.requested_shots, -1, -1):
        context = _context(train[:selected])
        if context is not None and len(context.encode("utf-8")) > 32 * 1024:
            continue
        examples = _examples(test, context)
        if all(
            len(_prompt(example)[0].encode("utf-8", "strict")) <= plan.max_prompt_bytes
            for example in examples
        ):
            result = PreparedGSM8KJSONL(
                plan=plan,
                train_rows=len(train),
                test_rows=len(test),
                selected_shots=selected,
                train_canonical_sha256=_sha(_json([row.to_dict() for row in train])),
                test_canonical_sha256=_sha(_json([row.to_dict() for row in test])),
                examples=examples,
            )
            result.benchmark_jsonl()
            return result
    raise ConfigurationError("test question exceeds prompt byte budget even with zero shots")


def verify_gsm8k_jsonl_preparation(
    artifact: bytes,
    evidence: bytes,
    train_source: bytes,
    test_source: bytes,
    plan: GSM8KJSONLPreparationPlan,
) -> bool:
    """Replay pinned sources and compare both output artifacts byte-for-byte."""
    if type(artifact) is not bytes or type(evidence) is not bytes:
        raise ConfigurationError("prepared artifact and evidence must be byte snapshots")
    expected = prepare_gsm8k_jsonl(train_source, test_source, plan)
    return artifact == expected.benchmark_jsonl() and evidence == expected.evidence_json()
