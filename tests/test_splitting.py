from __future__ import annotations

from dataclasses import replace

import pytest

from facetroute.errors import ConfigurationError
from facetroute.splitting import TracePartitions, split_traces, write_trace_partitions
from facetroute.traces import (
    RouteTrace,
    TraceOutcome,
    file_sha256,
    load_traces,
    traces_sha256,
    write_traces,
)
from facetroute.types import RouteRequest


def _traces(users: int = 6, records_per_user: int = 2) -> tuple[RouteTrace, ...]:
    result = []
    for user in range(users):
        for record in range(records_per_user):
            result.append(
                RouteTrace(
                    RouteRequest(
                        "offline input",
                        request_id=f"u{user}-r{record}",
                        user_id=f"u{user}",
                        metadata={"domain": f"d{user // 2}"},
                    ),
                    {
                        "strong": TraceOutcome(0.9, 0.02, 400, True),
                        "weak": TraceOutcome(0.6, 0.002, 80, True),
                    },
                    preferred_model="strong" if user % 2 else "weak",
                    route_score=(user + 1) / (users + 2),
                    strong_model="strong",
                    weak_model="weak",
                )
            )
    return tuple(result)


def _ids(values: tuple[RouteTrace, ...]) -> set[str]:
    return {trace.request.request_id for trace in values}


def test_group_split_is_deterministic_disjoint_and_keeps_users_together() -> None:
    traces = _traces()
    first = split_traces(traces, seed=31, group_by="user_id")
    second = split_traces(traces, seed=31, group_by="user_id")
    assert first == second
    assert _ids(first.train) | _ids(first.calibration) | _ids(first.test) == _ids(traces)
    assert not (_ids(first.train) & _ids(first.calibration))
    assert not (_ids(first.train) & _ids(first.test))
    assert not (_ids(first.calibration) & _ids(first.test))
    user_partitions: dict[str, set[str]] = {}
    for name in ("train", "calibration", "test"):
        for trace in first.traces(name):
            assert trace.request.user_id is not None
            user_partitions.setdefault(trace.request.user_id, set()).add(name)
    assert all(len(names) == 1 for names in user_partitions.values())


def test_group_membership_is_independent_of_source_row_order() -> None:
    traces = _traces()
    forward = split_traces(traces, seed=31, group_by="user_id")
    reverse = split_traces(tuple(reversed(traces)), seed=31, group_by="user_id")

    for name in ("train", "calibration", "test"):
        assert _ids(forward.traces(name)) == _ids(reverse.traces(name))


def test_metadata_group_split_never_crosses_a_domain() -> None:
    partitions = split_traces(_traces(), group_by="metadata:domain")
    domain_partitions: dict[str, set[str]] = {}
    for name in ("train", "calibration", "test"):
        for trace in partitions.traces(name):
            domain = trace.request.metadata["domain"]
            assert isinstance(domain, str)
            domain_partitions.setdefault(domain, set()).add(name)
    assert all(len(names) == 1 for names in domain_partitions.values())


def test_writer_emits_replayable_files_and_source_provenance(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    traces = _traces()
    write_traces(source, traces)
    source.write_bytes(b"\n" + source.read_bytes())
    partitions = split_traces(traces, group_by="user_id", seed=9)
    manifest = write_trace_partitions(
        tmp_path / "split",
        partitions,
        source_path=source,
        dataset_name="synthetic contract fixture",
        source_uri="local:generated-test-data",
        license_name="CC0-1.0",
    )
    assert manifest["dataset"]["input_file_sha256"] == file_sha256(source)
    assert manifest["dataset"]["canonical_sha256"] == traces_sha256(traces)
    assert manifest["dataset"]["input_file_sha256"] != manifest["dataset"]["canonical_sha256"]
    assert manifest["split"]["algorithm"] == "sha256-size-aware-deficit-v2"
    assert manifest["split"]["actual_fractions"] == {
        "train": len(partitions.train) / len(traces),
        "calibration": len(partitions.calibration) / len(traces),
        "test": len(partitions.test) / len(traces),
    }
    assert sum(manifest["split"]["group_counts"].values()) == 6
    for name in ("train", "calibration", "test"):
        path = tmp_path / "split" / f"{name}.jsonl"
        assert len(load_traces(path)) == manifest["partitions"][name]["records"]
        assert file_sha256(path) == manifest["partitions"][name]["file_sha256"]
    assert (tmp_path / "split" / "manifest.json").is_file()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"train_fraction": 0}, "train_fraction"),
        ({"calibration_fraction": float("nan")}, "calibration_fraction"),
        ({"train_fraction": 0.8, "calibration_fraction": 0.2}, "sum"),
        ({"seed": True}, "integer"),
        ({"group_by": "unknown"}, "group_by"),
    ],
)
def test_split_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        split_traces(_traces(), **kwargs)  # type: ignore[arg-type]


def test_split_requires_three_complete_groups() -> None:
    with pytest.raises(ConfigurationError, match="at least three"):
        split_traces(_traces(users=2), group_by="user_id")
    with pytest.raises(ConfigurationError, match="non-empty string"):
        split_traces(_traces(users=3), group_by="metadata:missing")


def test_manifest_requires_declared_provenance(tmp_path) -> None:
    traces = _traces()
    source = tmp_path / "source.jsonl"
    write_traces(source, traces)
    with pytest.raises(ConfigurationError, match="dataset_name"):
        write_trace_partitions(
            tmp_path / "split",
            split_traces(traces),
            source_path=source,
            dataset_name="",
            source_uri="local:test",
            license_name="CC0-1.0",
        )


def test_size_aware_split_assigns_a_dominant_group_to_the_largest_target() -> None:
    traces = tuple(
        RouteTrace(
            RouteRequest("x", request_id=f"{user}-{record}", user_id=user),
            {
                "strong": TraceOutcome(0.9, 0.02, 400, True),
                "weak": TraceOutcome(0.6, 0.002, 80, True),
            },
            preferred_model="strong",
            route_score=0.5,
            strong_model="strong",
            weak_model="weak",
        )
        for user, size in (("large", 100), ("small-a", 20), ("small-b", 20))
        for record in range(size)
    )

    partitions = split_traces(traces, seed=4, group_by="user_id")

    assert len(partitions.train) == 100
    assert sorted((len(partitions.calibration), len(partitions.test))) == [20, 20]


def test_trace_partitions_validate_group_and_configuration_invariants() -> None:
    traces = _traces(users=3)
    with pytest.raises(ConfigurationError, match="occurs in both"):
        TracePartitions(
            train=(traces[0],),
            calibration=(traces[1],),
            test=(traces[2],),
            seed=17,
            group_by="user_id",
            train_fraction=0.6,
            calibration_fraction=0.2,
        )
    with pytest.raises(ConfigurationError, match="split seed"):
        TracePartitions(
            train=(traces[0],),
            calibration=(traces[2],),
            test=(traces[4],),
            seed=True,
            group_by="user_id",
            train_fraction=0.6,
            calibration_fraction=0.2,
        )


def test_writer_rejects_source_output_collision_before_writing(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    traces = _traces()
    write_traces(source, traces)
    before = source.read_bytes()

    with pytest.raises(ConfigurationError, match="collides"):
        write_trace_partitions(
            tmp_path,
            split_traces(traces),
            source_path=source,
            dataset_name="fixture",
            source_uri="local:test",
            license_name="CC0-1.0",
        )

    assert source.read_bytes() == before
    assert not (tmp_path / "calibration.jsonl").exists()
    assert not (tmp_path / "test.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_writer_rejects_outputs_that_are_hardlinks_to_each_other(tmp_path) -> None:
    traces = _traces()
    source = tmp_path / "source.jsonl"
    write_traces(source, traces)
    output = tmp_path / "split"
    output.mkdir()
    anchor = output / "train.jsonl"
    anchor.write_text("sentinel\n", encoding="utf-8")
    (output / "calibration.jsonl").hardlink_to(anchor)
    (output / "test.jsonl").hardlink_to(anchor)

    with pytest.raises(ConfigurationError, match="path collides"):
        write_trace_partitions(
            output,
            split_traces(traces),
            source_path=source,
            dataset_name="fixture",
            source_uri="local:test",
            license_name="CC0-1.0",
        )

    assert anchor.read_text(encoding="utf-8") == "sentinel\n"
    assert not (output / "manifest.json").exists()


def test_writer_rejects_partitions_that_do_not_derive_from_source(tmp_path) -> None:
    traces = _traces()
    source = tmp_path / "source.jsonl"
    write_traces(source, _traces(users=7))
    output = tmp_path / "split"

    with pytest.raises(ConfigurationError, match="do not match the declared source"):
        write_trace_partitions(
            output,
            split_traces(traces, group_by="user_id"),
            source_path=source,
            dataset_name="fixture",
            source_uri="local:test",
            license_name="CC0-1.0",
        )

    assert not output.exists()


def test_writer_compares_canonical_records_not_python_numeric_equality(tmp_path) -> None:
    traces = _traces()
    source_traces = tuple(
        replace(
            trace,
            request=replace(trace.request, metadata={**trace.request.metadata, "fold": 1}),
        )
        for trace in traces
    )
    partition_traces = tuple(
        replace(
            trace,
            request=replace(trace.request, metadata={**trace.request.metadata, "fold": 1.0}),
        )
        for trace in traces
    )
    assert source_traces == partition_traces
    source = tmp_path / "source.jsonl"
    write_traces(source, source_traces)

    with pytest.raises(ConfigurationError, match="do not match the declared source"):
        write_trace_partitions(
            tmp_path / "split",
            split_traces(partition_traces, group_by="user_id"),
            source_path=source,
            dataset_name="fixture",
            source_uri="local:test",
            license_name="CC0-1.0",
        )
