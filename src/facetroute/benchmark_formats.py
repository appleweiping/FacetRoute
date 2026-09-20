"""Strict adapters for common offline LLM benchmark record formats.

The adapters intentionally stop at request construction.  They never execute a
model and never treat a benchmark answer as a routing outcome.  This keeps
evaluation data provenance explicit and prevents answer leakage into prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .errors import ConfigurationError
from .types import RouteRequest


class BenchmarkFormat(StrEnum):
    """Supported public benchmark record families."""

    MMLU = "mmlu"
    GSM8K = "gsm8k"
    MT_BENCH = "mt-bench"


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One normalized benchmark prompt and its held-out reference answer."""

    example_id: str
    format: BenchmarkFormat
    prompt: str
    choices: tuple[str, ...] = ()
    answer: int | str | None = None
    category: str | None = None
    turns: tuple[str, ...] = ()
    few_shot_context: str | None = None

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ConfigurationError("benchmark example_id cannot be empty")
        if not self.prompt.strip():
            raise ConfigurationError("benchmark prompt cannot be empty")
        if self.few_shot_context is not None and (
            self.format not in {BenchmarkFormat.MMLU, BenchmarkFormat.GSM8K}
            or type(self.few_shot_context) is not str
            or not self.few_shot_context.strip()
        ):
            raise ConfigurationError("few_shot_context requires bounded MMLU or GSM8K text")
        if self.few_shot_context is not None:
            try:
                if len(self.few_shot_context.encode("utf-8", "strict")) > 32 * 1024:
                    raise ConfigurationError("few_shot_context exceeds 32 KiB")
            except UnicodeError:
                raise ConfigurationError("few_shot_context must be strict UTF-8") from None
        if self.format is BenchmarkFormat.MMLU:
            if len(self.choices) < 2:
                raise ConfigurationError("MMLU examples require at least two choices")
            if self.answer is None:
                raise ConfigurationError("MMLU examples require an answer")
            if isinstance(self.answer, int):
                if not 0 <= self.answer < len(self.choices):
                    raise ConfigurationError("MMLU answer index is outside choices")
            elif self.answer not in self.choices:
                raise ConfigurationError("MMLU answer text is not one of choices")
        elif self.format is BenchmarkFormat.GSM8K:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ConfigurationError("GSM8K examples require a non-empty answer")
        elif self.format is BenchmarkFormat.MT_BENCH:
            if not self.turns:
                raise ConfigurationError("MT-Bench examples require at least one turn")

    def to_request(
        self, *, user_id: str = "benchmark", expected_output_tokens: int = 512
    ) -> RouteRequest:
        """Build a provider-neutral request without placing the answer in its query."""

        if (
            isinstance(expected_output_tokens, bool)
            or not isinstance(expected_output_tokens, int)
            or expected_output_tokens < 0
        ):
            raise ConfigurationError("expected_output_tokens must be a non-negative integer")
        metadata: dict[str, Any] = {
            "benchmark_format": self.format.value,
            "benchmark_example_id": self.example_id,
        }
        if self.category is not None:
            metadata["benchmark_category"] = self.category
        if self.choices:
            metadata["benchmark_choices"] = list(self.choices)
        return RouteRequest(
            query=self.prompt,
            user_id=user_id,
            expected_output_tokens=expected_output_tokens,
            request_id=f"{self.format.value}:{self.example_id}",
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.example_id,
            "format": self.format.value,
            "prompt": self.prompt,
        }
        if self.choices:
            result["choices"] = list(self.choices)
        if self.answer is not None:
            result["answer"] = self.answer
        if self.category is not None:
            result["category"] = self.category
        if self.turns:
            result["turns"] = list(self.turns)
        if self.few_shot_context is not None:
            result["few_shot_context"] = self.few_shot_context
        return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _format_for(record: dict[str, Any], requested: BenchmarkFormat | str) -> BenchmarkFormat:
    if requested != "auto":
        try:
            return BenchmarkFormat(requested)
        except ValueError as exc:
            raise ConfigurationError(f"unknown benchmark format: {requested}") from exc
    keys = set(record)
    if "format" in record and "prompt" in record:
        try:
            return BenchmarkFormat(record["format"])
        except ValueError as exc:
            raise ConfigurationError(f"unknown benchmark format: {record['format']}") from exc
    if {"question", "choices", "answer"} <= keys:
        return BenchmarkFormat.MMLU
    if {"question", "answer"} <= keys:
        return BenchmarkFormat.GSM8K
    if {"question_id", "turns"} <= keys:
        return BenchmarkFormat.MT_BENCH
    raise ConfigurationError(
        "cannot detect benchmark format; expected MMLU question/choices/answer, "
        "GSM8K question/answer, or MT-Bench question_id/turns"
    )


def _parse_record(
    raw: object, line_number: int, requested: BenchmarkFormat | str
) -> BenchmarkExample:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"benchmark record {line_number} must be an object")
    format_name = _format_for(raw, requested)
    try:
        if format_name is BenchmarkFormat.MMLU:
            subject = raw.get("subject")
            category = raw.get("category")
            if subject is not None and category is not None and subject != category:
                raise ConfigurationError("MMLU subject and category disagree")
            choices_raw = raw["choices"]
            if not isinstance(choices_raw, list) or not all(
                isinstance(choice, str) and choice.strip() for choice in choices_raw
            ):
                raise ConfigurationError("MMLU choices must be a non-empty string array")
            answer = raw["answer"]
            if isinstance(answer, bool) or not isinstance(answer, (int, str)):
                raise ConfigurationError("MMLU answer must be an integer index or choice text")
            return BenchmarkExample(
                example_id=_string(raw.get("id", f"mmlu-{line_number}"), "MMLU id"),
                format=format_name,
                prompt=_string(raw.get("question", raw.get("prompt")), "MMLU question"),
                choices=tuple(choice.strip() for choice in choices_raw),
                answer=answer,
                category=(
                    _string(subject if subject is not None else category, "MMLU subject")
                    if subject is not None or category is not None
                    else None
                ),
                few_shot_context=raw.get("few_shot_context"),
            )
        if format_name is BenchmarkFormat.GSM8K:
            return BenchmarkExample(
                example_id=_string(raw.get("id", f"gsm8k-{line_number}"), "GSM8K id"),
                format=format_name,
                prompt=_string(raw.get("question", raw.get("prompt")), "GSM8K question"),
                answer=_string(raw["answer"], "GSM8K answer"),
                few_shot_context=raw.get("few_shot_context"),
            )
        turns_raw = raw.get("turns")
        if (
            not isinstance(turns_raw, list)
            or not turns_raw
            or not all(isinstance(turn, str) and turn.strip() for turn in turns_raw)
        ):
            raise ConfigurationError("MT-Bench turns must be a non-empty string array")
        turns = tuple(turn.strip() for turn in turns_raw)
        return BenchmarkExample(
            example_id=_string(raw.get("question_id", raw.get("id")), "MT-Bench question_id"),
            format=format_name,
            prompt="\n\n".join(f"Turn {index}: {turn}" for index, turn in enumerate(turns, 1)),
            category=(
                _string(raw["category"], "MT-Bench category")
                if raw.get("category") is not None
                else None
            ),
            turns=turns,
        )
    except KeyError as exc:
        raise ConfigurationError(
            f"benchmark record {line_number} is missing {exc.args[0]}"
        ) from exc


def _records(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ConfigurationError("benchmark input must be a JSON array/object or JSONL")


def _validate_limits(max_bytes: int, max_records: int) -> None:
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records <= 0
    ):
        raise ValueError("benchmark limits must be positive")


def _examples_from_bytes(
    raw: bytes, *, source_name: str, format: BenchmarkFormat | str, max_records: int
) -> tuple[BenchmarkExample, ...]:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"invalid benchmark UTF-8 in {source_name}: {exc}") from exc
    try:
        try:
            payload = loads_strict(text)
            raw_records = _records(payload)
        except (ValueError, json.JSONDecodeError):
            raw_records = []
            for _line_number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    raw_records.append(loads_strict(line))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid benchmark JSON in {source_name}: {exc}") from exc
    if not raw_records:
        raise ConfigurationError("benchmark input contains no records")
    if len(raw_records) > max_records:
        raise ConfigurationError(f"benchmark input exceeds {max_records} records")
    examples = tuple(_parse_record(raw, index, format) for index, raw in enumerate(raw_records, 1))
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("benchmark example IDs must be unique")
    formats = {example.format for example in examples}
    if len(formats) != 1:
        raise ConfigurationError("benchmark input cannot mix MMLU, GSM8K, and MT-Bench records")
    return examples


def load_benchmark_examples_bytes(
    raw: bytes,
    *,
    format: BenchmarkFormat | str = "auto",
    max_bytes: int = 128 * 1024 * 1024,
    max_records: int = 1_000_000,
) -> tuple[BenchmarkExample, ...]:
    """Parse one already-read byte snapshot under the same strict adapter rules."""

    _validate_limits(max_bytes, max_records)
    if not isinstance(raw, bytes):
        raise ConfigurationError("benchmark byte snapshot must be bytes")
    if len(raw) > max_bytes:
        raise ConfigurationError(f"benchmark input exceeds {max_bytes} bytes")
    return _examples_from_bytes(raw, source_name="<bytes>", format=format, max_records=max_records)


def load_benchmark_examples(
    path: str | Path,
    *,
    format: BenchmarkFormat | str = "auto",
    max_bytes: int = 128 * 1024 * 1024,
    max_records: int = 1_000_000,
) -> tuple[BenchmarkExample, ...]:
    """Load MMLU, GSM8K, or MT-Bench JSON/JSONL with strict limits."""

    _validate_limits(max_bytes, max_records)
    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ConfigurationError(f"cannot read benchmark input {source}: {exc}") from exc
    if len(raw) > max_bytes:
        raise ConfigurationError(f"benchmark input exceeds {max_bytes} bytes")
    return _examples_from_bytes(
        raw, source_name=str(source), format=format, max_records=max_records
    )


def write_benchmark_examples(path: str | Path, examples: tuple[BenchmarkExample, ...]) -> None:
    """Write canonical JSONL for provenance-stable offline workflows."""

    if not examples:
        raise ConfigurationError("cannot write an empty benchmark")
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(
                json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in examples
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigurationError(f"cannot write benchmark output {destination}: {exc}") from exc
