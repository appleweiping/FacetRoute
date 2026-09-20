"""Bounded, offline preparation of caller-pinned MMLU-shaped dev/test CSVs.

The artifact is an input to FacetRoute's own public-score adapter, not a
replication of any external tokenizer, response generator, or benchmark score.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass

from .benchmark_formats import BenchmarkExample, BenchmarkFormat
from .errors import ConfigurationError
from .public_scores import PublicScoreProvenance, _prompt

PROTOCOL = "facet-mmlu-csv-fewshot-v1"
MAX_CSV_BYTES = 1024 * 1024
MAX_DEV_ROWS = 20
MAX_TEST_ROWS = 500
MAX_CELL_BYTES = 4096
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_SUBJECT = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_LETTERS = "ABCD"


def _json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8", "strict")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class MMLUCSVPreparationPlan:
    """Caller-declared subject, exact source digests, licenses, and byte budget."""

    subject: str
    dev: PublicScoreProvenance
    test: PublicScoreProvenance
    requested_shots: int = 5
    max_prompt_bytes: int = 8192

    def __post_init__(self) -> None:
        if type(self.subject) is not str or _SUBJECT.fullmatch(self.subject) is None:
            raise ConfigurationError("subject must be a bounded lowercase ASCII slug")
        if not isinstance(self.dev, PublicScoreProvenance) or not isinstance(
            self.test, PublicScoreProvenance
        ):
            raise ConfigurationError("dev/test require declared source URI, license and SHA-256")
        if type(self.requested_shots) is not int or not 0 <= self.requested_shots <= 5:
            raise ConfigurationError("requested_shots must be an integer in [0, 5]")
        if type(self.max_prompt_bytes) is not int or not 64 <= self.max_prompt_bytes <= 32 * 1024:
            raise ConfigurationError("max_prompt_bytes must be an integer in [64, 32768]")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "requested_shots": self.requested_shots,
            "max_prompt_bytes": self.max_prompt_bytes,
            "dev": {
                "uri": self.dev.source_uri,
                "license": self.dev.license_id,
                "sha256": self.dev.source_sha256,
            },
            "test": {
                "uri": self.test.source_uri,
                "license": self.test.license_id,
                "sha256": self.test.source_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class _CSVRow:
    question: str
    choices: tuple[str, str, str, str]
    answer: int

    def to_dict(self) -> dict[str, object]:
        return {"question": self.question, "choices": list(self.choices), "answer": self.answer}


def _cell(raw: str, *, row: int, column: int) -> str:
    text = raw.strip()
    if (
        not text
        or len(text.encode("utf-8", "strict")) > MAX_CELL_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ConfigurationError(f"CSV row {row} column {column} is empty, unsafe or too long")
    return text


def _csv_rows(raw: bytes, *, label: str, maximum: int) -> tuple[_CSVRow, ...]:
    if type(raw) is not bytes or len(raw) > MAX_CSV_BYTES:
        raise ConfigurationError(f"{label} must be a byte snapshot within {MAX_CSV_BYTES} bytes")
    try:
        text = raw.decode("utf-8", "strict")
        if text.startswith("\ufeff"):
            raise ConfigurationError(f"{label} CSV must not contain a BOM")
        if any(not line.strip() for line in text.splitlines()):
            raise ConfigurationError(f"{label} CSV must not contain blank physical rows")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        rows: list[_CSVRow] = []
        for number, cells in enumerate(reader, 1):
            if number > maximum:
                raise ConfigurationError(f"{label} exceeds {maximum} rows")
            if len(cells) != 6:
                raise ConfigurationError(f"{label} CSV row {number} requires six columns")
            values = tuple(
                _cell(value, row=number, column=index) for index, value in enumerate(cells, 1)
            )
            question, *remaining = values
            choices = remaining[:4]
            answer = remaining[4]
            if answer not in _LETTERS:
                raise ConfigurationError(f"{label} CSV row {number} answer must be A, B, C or D")
            if len({choice.casefold() for choice in choices}) != 4:
                raise ConfigurationError(f"{label} CSV row {number} choices must be distinct")
            rows.append(
                _CSVRow(
                    question,
                    (choices[0], choices[1], choices[2], choices[3]),
                    _LETTERS.index(answer),
                )
            )
    except (UnicodeError, csv.Error) as error:
        raise ConfigurationError(f"{label} CSV is not strict UTF-8 CSV: {error}") from error
    if not rows:
        raise ConfigurationError(f"{label} CSV contains no rows")
    return tuple(rows)


def _stem_key(row: _CSVRow) -> str:
    return " ".join(unicodedata.normalize("NFKC", row.question).casefold().split())


def _unique_stems(rows: tuple[_CSVRow, ...], label: str) -> set[str]:
    keys = {_stem_key(row) for row in rows}
    if len(keys) != len(rows):
        raise ConfigurationError(f"{label} contains duplicate normalized question stems")
    return keys


def _context(subject: str, rows: tuple[_CSVRow, ...]) -> str | None:
    if not rows:
        return None
    parts = [f"Examples about {subject.replace('_', ' ')} (answers shown for examples only):"]
    for row in rows:
        parts.append(
            "\n".join(
                (
                    f"Question: {row.question}",
                    *(
                        f"{letter}. {choice}"
                        for letter, choice in zip(_LETTERS, row.choices, strict=True)
                    ),
                    f"Answer: {_LETTERS[row.answer]}",
                )
            )
        )
    return "\n\n".join(parts)


def _examples(
    subject: str, rows: tuple[_CSVRow, ...], context: str | None
) -> tuple[BenchmarkExample, ...]:
    return tuple(
        BenchmarkExample(
            example_id=f"{subject}-{index:04d}",
            format=BenchmarkFormat.MMLU,
            prompt=row.question,
            choices=row.choices,
            answer=row.answer,
            category=subject,
            few_shot_context=context,
        )
        for index, row in enumerate(rows, 1)
    )


@dataclass(frozen=True, slots=True)
class PreparedMMLUCSV:
    """Immutable in-memory artifact with answer-free prompt digests in evidence."""

    plan: MMLUCSVPreparationPlan
    dev_rows: int
    test_rows: int
    selected_shots: int
    dev_canonical_sha256: str
    test_canonical_sha256: str
    examples: tuple[BenchmarkExample, ...]

    def benchmark_jsonl(self) -> bytes:
        """Return public-score-compatible JSONL; it contains private test gold labels."""
        raw = b"".join(_json(example.to_dict()) + b"\n" for example in self.examples)
        if len(raw) > MAX_OUTPUT_BYTES:
            raise ConfigurationError("prepared MMLU artifact exceeds output byte limit")
        return raw

    def evidence_json(self) -> bytes:
        """Return aggregate provenance, without prompt text or gold answers."""
        artifact = self.benchmark_jsonl()
        report = {
            "protocol": PROTOCOL,
            "plan": self.plan.to_dict(),
            "dev_rows": self.dev_rows,
            "test_rows": self.test_rows,
            "selected_shots": self.selected_shots,
            "dev_canonical_sha256": self.dev_canonical_sha256,
            "test_canonical_sha256": self.test_canonical_sha256,
            "artifact_sha256": _sha(artifact),
            "provider_prompt_sha256": [
                {"id": example.example_id, "sha256": _sha(_prompt(example)[0].encode("utf-8"))}
                for example in self.examples
            ],
        }
        return _json(report) + b"\n"


def prepare_mmlu_csv(
    dev_source: bytes,
    test_source: bytes,
    plan: MMLUCSVPreparationPlan,
    *,
    dev_name: str,
    test_name: str,
) -> PreparedMMLUCSV:
    """Prepare exact pinned snapshots; no model, tokenizer, or provider is invoked."""
    if not isinstance(plan, MMLUCSVPreparationPlan):
        raise ConfigurationError("MMLU preparation requires an explicit plan")
    if dev_name != f"{plan.subject}_dev.csv" or test_name != f"{plan.subject}_test.csv":
        raise ConfigurationError("declared subject does not match dev/test CSV basenames")
    if type(dev_source) is not bytes or type(test_source) is not bytes:
        raise ConfigurationError("MMLU sources must be immutable byte snapshots")
    if _sha(dev_source) != plan.dev.source_sha256 or _sha(test_source) != plan.test.source_sha256:
        raise ConfigurationError("MMLU source SHA-256 differs from caller pin")
    dev = _csv_rows(dev_source, label="dev", maximum=MAX_DEV_ROWS)
    test = _csv_rows(test_source, label="test", maximum=MAX_TEST_ROWS)
    if plan.requested_shots > len(dev):
        raise ConfigurationError("requested_shots exceeds available dev rows")
    dev_keys = _unique_stems(dev, "dev")
    test_keys = _unique_stems(test, "test")
    if dev_keys & test_keys:
        raise ConfigurationError("dev/test normalized question stems overlap")
    for selected in range(plan.requested_shots, -1, -1):
        context = _context(plan.subject, dev[:selected])
        examples = _examples(plan.subject, test, context)
        if all(
            len(_prompt(example)[0].encode("utf-8", "strict")) <= plan.max_prompt_bytes
            for example in examples
        ):
            result = PreparedMMLUCSV(
                plan=plan,
                dev_rows=len(dev),
                test_rows=len(test),
                selected_shots=selected,
                dev_canonical_sha256=_sha(_json([row.to_dict() for row in dev])),
                test_canonical_sha256=_sha(_json([row.to_dict() for row in test])),
                examples=examples,
            )
            result.benchmark_jsonl()
            return result
    raise ConfigurationError("test question exceeds prompt byte budget even with zero shots")


def verify_mmlu_csv_preparation(
    artifact: bytes,
    evidence: bytes,
    dev_source: bytes,
    test_source: bytes,
    plan: MMLUCSVPreparationPlan,
    *,
    dev_name: str,
    test_name: str,
) -> bool:
    """Replay exact sources and compare both output artifacts byte-for-byte."""
    if type(artifact) is not bytes or type(evidence) is not bytes:
        raise ConfigurationError("prepared artifact and evidence must be byte snapshots")
    expected = prepare_mmlu_csv(
        dev_source, test_source, plan, dev_name=dev_name, test_name=test_name
    )
    return artifact == expected.benchmark_jsonl() and evidence == expected.evidence_json()
