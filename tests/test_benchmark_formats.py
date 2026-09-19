from __future__ import annotations

import json

import pytest

from facetroute.benchmark_formats import (
    BenchmarkExample,
    BenchmarkFormat,
    load_benchmark_examples,
    write_benchmark_examples,
)
from facetroute.errors import ConfigurationError


def test_benchmark_example_invariants_and_request_limits():
    with pytest.raises(ConfigurationError, match="example_id"):
        BenchmarkExample(" ", BenchmarkFormat.GSM8K, "prompt", answer="answer")
    with pytest.raises(ConfigurationError, match="prompt"):
        BenchmarkExample("id", BenchmarkFormat.GSM8K, " ", answer="answer")
    with pytest.raises(ConfigurationError, match="choices"):
        BenchmarkExample("id", BenchmarkFormat.MMLU, "prompt", choices=("one",), answer=0)
    with pytest.raises(ConfigurationError, match="answer"):
        BenchmarkExample("id", BenchmarkFormat.MMLU, "prompt", choices=("one", "two"))
    with pytest.raises(ConfigurationError, match="outside"):
        BenchmarkExample("id", BenchmarkFormat.MMLU, "prompt", choices=("one", "two"), answer=2)
    with pytest.raises(ConfigurationError, match="not one"):
        BenchmarkExample("id", BenchmarkFormat.MMLU, "prompt", choices=("one", "two"), answer="x")
    with pytest.raises(ConfigurationError, match="GSM8K"):
        BenchmarkExample("id", BenchmarkFormat.GSM8K, "prompt", answer=" ")
    with pytest.raises(ConfigurationError, match="turn"):
        BenchmarkExample("id", BenchmarkFormat.MT_BENCH, "prompt")

    example = BenchmarkExample("id", BenchmarkFormat.GSM8K, "prompt", answer="4")
    with pytest.raises(ConfigurationError, match="expected_output_tokens"):
        example.to_request(expected_output_tokens=True)
    with pytest.raises(ConfigurationError, match="expected_output_tokens"):
        example.to_request(expected_output_tokens=-1)


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

    with pytest.raises(ConfigurationError, match="unknown benchmark format"):
        load_benchmark_examples(source, format="not-a-format")

    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(
            [
                {"question": "q", "choices": ["a", "b"], "answer": 0},
                {"question": "q", "answer": "a"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="mix"):
        load_benchmark_examples(mixed)


def test_limits_and_io_errors_are_explicit(tmp_path):
    source = tmp_path / "mmlu.json"
    source.write_text(
        json.dumps({"question": "q", "choices": ["a", "b"], "answer": 0}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="limits"):
        load_benchmark_examples(source, max_bytes=0)
    with pytest.raises(ValueError, match="limits"):
        load_benchmark_examples(source, max_records=0)
    with pytest.raises(ConfigurationError, match="exceeds"):
        load_benchmark_examples(source, max_bytes=1)
    two = tmp_path / "two.jsonl"
    two.write_text(
        '{"question":"q","choices":["a","b"],"answer":0}\n'
        '{"question":"q2","choices":["a","b"],"answer":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="exceeds"):
        load_benchmark_examples(two, max_records=1)
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_benchmark_examples(tmp_path / "missing.json")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no records"):
        load_benchmark_examples(empty)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"question":"q"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid benchmark JSON"):
        load_benchmark_examples(malformed, format="mmlu")


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


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"format": "unknown", "prompt": "q"}, "unknown benchmark format"),
        ([{"question": "q", "answer": "a"}, 7], "must be an object"),
        ({"question": "q", "choices": ["a", 4], "answer": 0}, "choices"),
        ({"question": "q", "choices": ["a", "b"], "answer": True}, "answer"),
        ({"format": "mmlu", "prompt": "q", "choices": ["a", "b"]}, "missing answer"),
        ({"question": "q"}, "cannot detect"),
        ({"question_id": "q", "turns": []}, "turns"),
        ({"question_id": "q", "turns": [" "]}, "turns"),
        ({"question_id": "q", "turns": ["x"], "category": 5}, "category"),
        (7, "must be an object"),
    ],
)
def test_benchmark_adapter_rejects_malformed_official_records(tmp_path, record, message):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_benchmark_examples(source)


def test_benchmark_writer_rejects_empty_and_directory_destination(tmp_path):
    with pytest.raises(ConfigurationError, match="empty"):
        write_benchmark_examples(tmp_path / "out.jsonl", ())
    examples = (BenchmarkExample("id", BenchmarkFormat.GSM8K, "q", answer="a"),)
    with pytest.raises(ConfigurationError, match="cannot write"):
        write_benchmark_examples(tmp_path, examples)


def test_benchmark_loader_limits_bytes_read_not_only_an_earlier_file_stat(tmp_path, monkeypatch):
    source = tmp_path / "racing.json"
    source.write_text(json.dumps({"question": "q" * 300, "answer": "1"}), encoding="utf-8")
    original_stat = type(source).stat

    class ReportedSize:
        st_size = 1

    def stale_stat(path, *args, **kwargs):
        if path == source:
            return ReportedSize()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "stat", stale_stat)
    with pytest.raises(ConfigurationError, match="exceeds"):
        load_benchmark_examples(source, max_bytes=100)


def test_benchmark_loader_reports_invalid_utf8(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_bytes(b"\xff")
    with pytest.raises(ConfigurationError, match="UTF-8"):
        load_benchmark_examples(source)


@pytest.mark.parametrize(
    "limits", [{"max_bytes": 1.5}, {"max_records": True}, {"max_records": 1.5}]
)
def test_benchmark_loader_rejects_noninteger_resource_limits(tmp_path, limits):
    source = tmp_path / "valid.json"
    source.write_text('{"question":"q","answer":"1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="limits"):
        load_benchmark_examples(source, **limits)
