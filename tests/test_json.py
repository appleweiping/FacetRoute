from __future__ import annotations

from pathlib import Path

import pytest

from facetroute._json import _MAX_JSON_NESTING, loads_strict
from facetroute.cli import main
from facetroute.errors import ConfigurationError, PersistenceError
from facetroute.similarity import SimilarityModel
from facetroute.traces import load_traces


@pytest.mark.parametrize(
    "payload",
    [
        b"1e999",
        b"-1e999",
        b'{"value":1e999}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
    ],
)
def test_strict_json_rejects_every_non_finite_number_spelling(payload: bytes) -> None:
    with pytest.raises(ValueError, match="non-finite JSON number"):
        loads_strict(payload)


def test_strict_json_accepts_finite_exponents() -> None:
    assert loads_strict(b'{"high":1e308,"low":-1e-308}') == {
        "high": 1e308,
        "low": -1e-308,
    }


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value":"\\ud800"}',
        b'{"value":"\\udfff"}',
        b'{"\\ud800":"value"}',
        '"\ud800"',
    ],
)
def test_strict_json_rejects_unpaired_surrogates(payload: str | bytes) -> None:
    with pytest.raises(ValueError, match="valid Unicode scalars"):
        loads_strict(payload)


def test_strict_json_accepts_valid_unicode_and_surrogate_pairs() -> None:
    assert loads_strict(b'{"value":"\\ud83d\\ude00"}') == {"value": "😀"}
    assert loads_strict('{"value":"数据"}') == {"value": "数据"}


def test_strict_json_snapshots_a_string_subclass_without_calling_its_hooks() -> None:
    class HookedString(str):
        def __iter__(self):
            return iter("[" * 1_000)

        def encode(self, *args, **kwargs):
            return b"null"

    assert loads_strict(HookedString('{"value":1}')) == {"value": 1}


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value":"\xff"}',
        bytearray(b'{"value":"\xfe"}'),
        '{"value":1}'.encode("utf-16"),
    ],
)
def test_strict_json_rejects_non_utf8_byte_inputs(payload: bytes | bytearray) -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        loads_strict(payload)


def test_strict_json_nesting_limit_is_platform_independent() -> None:
    at_limit = "[" * _MAX_JSON_NESTING + "0" + "]" * _MAX_JSON_NESTING
    assert isinstance(loads_strict(at_limit), list)

    too_deep = "[" * (_MAX_JSON_NESTING + 1) + "0" + "]" * (_MAX_JSON_NESTING + 1)
    with pytest.raises(ValueError, match="nesting exceeds"):
        loads_strict(too_deep)

    assert loads_strict('{"literal":"[[[{{{","escaped":"\\"["}') == {
        "literal": "[[[{{{",
        "escaped": '"[',
    }


def _trace(*, query: str = '"safe"', metadata: str = "{}") -> bytes:
    return (
        '{"request_id":"r1","request":{"query":'
        + query
        + ',"request_id":"r1","metadata":'
        + metadata
        + '},"outcomes":{"alpha":{"quality":1,"cost_usd":0,'
        '"latency_ms":1,"success":true}},"preferred_model":"alpha"}\n'
    ).encode("ascii")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_trace(metadata='{"overflow":1e999}'), "non-finite JSON number"),
        (_trace(query='"\\ud800"'), "valid Unicode scalars"),
        (b"[" * 300 + b"0" + b"]" * 300 + b"\n", "nesting exceeds"),
    ],
)
def test_trace_loader_wraps_strict_json_failures_as_configuration_errors(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(content)

    with pytest.raises(ConfigurationError, match=message):
        load_traces(path)


@pytest.mark.parametrize("content", [b'{"x":1e999}', b'{"x":"\\ud800"}'])
def test_similarity_loader_wraps_strict_json_failures_as_persistence_errors(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(content)

    with pytest.raises(PersistenceError, match="Cannot decode similarity state"):
        SimilarityModel.load(path)


@pytest.mark.parametrize(
    "request_json",
    [
        '{"query":"safe","metadata":{"overflow":1e999}}',
        '{"query":"\\ud800"}',
    ],
)
def test_route_cli_reports_strict_json_failures_without_a_traceback(
    capsys: pytest.CaptureFixture[str], request_json: str
) -> None:
    models = Path(__file__).parents[1] / "examples" / "models.json"

    assert main(["route", "--models", str(models), "--request-json", request_json]) == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    assert "JSON" in error or "non-finite" in error
