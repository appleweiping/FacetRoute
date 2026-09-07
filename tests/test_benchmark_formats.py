from __future__ import annotations

import json

import pytest

from facetroute.benchmark_formats import (
    BenchmarkFormat,
    load_benchmark_examples,
    write_benchmark_examples,
)
from facetroute.errors import ConfigurationError


def test_mmlu_json_array_is_normalized_without_answer_leakage(tmp_path):
    source = tmp_path / "mmlu.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question": "Which planet is closest to the Sun?",
                    "choices": ["Earth", "Mercury", "Mars", "Venus"],
                    "answer": 1,
                    "subject": "astronomy",
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = load_benchmark_examples(source)
    assert examples[0].format is BenchmarkFormat.MMLU
    assert examples[0].answer == 1
    request = examples[0].to_request()
    assert "Mercury" not in request.query
    assert request.metadata["benchmark_choices"] == ["Earth", "Mercury", "Mars", "Venus"]


def test_gsm8k_jsonl_and_mt_bench_multiline_object(tmp_path):
    gsm = tmp_path / "gsm8k.jsonl"
    gsm.write_text('{"question":"If x is 2, what is x+2?","answer":"4"}\n', encoding="utf-8")
    examples = load_benchmark_examples(gsm)
    assert examples[0].format is BenchmarkFormat.GSM8K
    assert examples[0].to_request().request_id == "gsm8k:gsm8k-1"

    mt = tmp_path / "mt.json"
    mt.write_text(
        json.dumps(
            {
                "question_id": "q-7",
                "category": "writing",
                "turns": ["Draft a title.", "Now explain the choice."],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mt_examples = load_benchmark_examples(mt)
    assert mt_examples[0].format is BenchmarkFormat.MT_BENCH
    assert mt_examples[0].prompt.startswith("Turn 1: Draft a title.")
    assert mt_examples[0].category == "writing"


def test_format_override_and_validation_reject_ambiguous_or_duplicate_records(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "x", "question": "q", "choices": ["a", "b"], "answer": 0}),
                json.dumps({"id": "x", "question": "q2", "choices": ["a", "b"], "answer": 1}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unique"):
        load_benchmark_examples(source, format="mmlu")

    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps({"question": "only a question"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot detect"):
        load_benchmark_examples(ambiguous)


def test_canonical_writer_round_trips(tmp_path):
    source = tmp_path / "mmlu.json"
    source.write_text(
        json.dumps({"id": "m", "question": "q", "choices": ["a", "b"], "answer": 0}),
        encoding="utf-8",
    )
    examples = load_benchmark_examples(source, format=BenchmarkFormat.MMLU)
    output = tmp_path / "out.jsonl"
    write_benchmark_examples(output, examples)
    assert load_benchmark_examples(output, format=BenchmarkFormat.MMLU) == examples
