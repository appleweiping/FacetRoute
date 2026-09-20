from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from facetroute.errors import ConfigurationError
from facetroute.live_profile import LiveProfile, LiveProvider, main, run_live_profile
from facetroute.providers import ProviderError
from facetroute.public_scores import PublicScoreProvenance


class _Server(ThreadingHTTPServer):
    calls: list[dict[str, Any]]
    bad_response: bool


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        server = cast(_Server, self.server)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.calls.append(
            {"path": self.path, "authorization": self.headers.get("Authorization"), "body": request}
        )
        body = json.dumps(
            {
                "id": "local",
                "object": "chat.completion",
                "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}}],
            }
        ).encode()
        if server.bad_response:
            body = b"not-json"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _server() -> Iterator[_Server]:
    server = _Server(("127.0.0.1", 0), _Handler)
    server.calls = []
    server.bad_response = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _source(tmp_path: Path) -> tuple[Path, PublicScoreProvenance]:
    source = tmp_path / "licensed.jsonl"
    raw = b'{"id":"m1","question":"Choose A.","choices":["A","B"],"answer":0,"subject":"demo"}\n'
    source.write_bytes(raw)
    return source, PublicScoreProvenance(
        "fixture:licensed", "CC-BY-4.0", hashlib.sha256(raw).hexdigest()
    )


def _profile(
    tmp_path: Path, base_url: str, *, max_calls: int = 0, acknowledge_cost: bool = False
) -> LiveProfile:
    source, provenance = _source(tmp_path)
    return LiveProfile(
        source,
        tmp_path / "checkpoint.json",
        provenance,
        LiveProvider("weak", "upstream-weak", "rev-1", base_url, "WEAK_KEY"),
        LiveProvider("strong", "upstream-strong", "rev-1", base_url, "STRONG_KEY"),
        max_calls=max_calls,
        acknowledge_cost=acknowledge_cost,
    )


def test_zero_call_preflight_needs_no_secret_and_does_not_contact_server(
    tmp_path: Path, monkeypatch: Any
) -> None:
    with _server() as server:
        monkeypatch.setitem(sys.modules, "httpx", None)
        monkeypatch.setitem(sys.modules, "facetroute.async_http", None)
        profile = _profile(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        progress = asyncio.run(run_live_profile(profile, environment={}))
        assert progress.completed_calls == 0
        assert progress.total_calls == 2
        assert not progress.finished
        assert server.calls == []
        assert profile.checkpoint_path.exists()


def test_zero_call_checkpoint_resumes_with_paid_provider(tmp_path: Path) -> None:
    with _server() as server:
        profile = _profile(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        preflight = asyncio.run(run_live_profile(profile, environment={}))
        assert preflight.completed_calls == 0 and server.calls == []
        paid = replace(profile, max_calls=1, acknowledge_cost=True)
        progressed = asyncio.run(
            run_live_profile(paid, environment={"WEAK_KEY": "weak", "STRONG_KEY": "strong"})
        )
        assert (
            progressed.completed_calls == 1
            and progressed.manifest_sha256 == preflight.manifest_sha256
        )
        assert len(server.calls) == 1


def test_explicit_cost_confirmation_precedes_network_and_checkpoint(tmp_path: Path) -> None:
    with _server() as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with pytest.raises(ConfigurationError, match="acknowledge-cost"):
            _profile(tmp_path, base_url, max_calls=1)
        assert server.calls == []
        assert not (tmp_path / "checkpoint.json").exists()


def test_live_http_call_resume_and_secret_redaction(tmp_path: Path) -> None:
    with _server() as server:
        profile = _profile(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/v1",
            max_calls=1,
            acknowledge_cost=True,
        )
        env = {"WEAK_KEY": "weak-private-token", "STRONG_KEY": "strong-private-token"}
        first = asyncio.run(run_live_profile(profile, environment=env))
        assert (first.completed_calls, first.total_calls, first.finished) == (1, 2, False)
        second = asyncio.run(run_live_profile(profile, environment=env))
        assert (second.completed_calls, second.total_calls, second.finished) == (2, 2, True)
        assert (second.weak_accuracy, second.strong_accuracy) == (1.0, 1.0)
        assert [call["body"]["model"] for call in server.calls] == [
            "upstream-weak",
            "upstream-strong",
        ]
        assert [call["authorization"] for call in server.calls] == [
            "Bearer weak-private-token",
            "Bearer strong-private-token",
        ]
        assert all("answer" not in call["body"]["messages"][0] for call in server.calls)
        saved = profile.checkpoint_path.read_text()
        assert "private-token" not in saved
        assert json.loads(saved)["manifest"]["models"][0]["revision"].startswith("rev-1@")
        again = asyncio.run(run_live_profile(profile, environment=env))
        assert again == second and len(server.calls) == 2


def test_missing_key_is_redacted_before_network(tmp_path: Path) -> None:
    with _server() as server:
        profile = _profile(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/v1",
            max_calls=1,
            acknowledge_cost=True,
        )
        with pytest.raises(ConfigurationError, match="environment variable is unset") as exc:
            asyncio.run(run_live_profile(profile, environment={"WEAK_KEY": "private-token"}))
        assert "private-token" not in str(exc.value)
        assert server.calls == []
        assert not profile.checkpoint_path.exists()


def test_endpoint_change_cannot_resume_existing_checkpoint(tmp_path: Path) -> None:
    with _server() as server:
        profile = _profile(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        asyncio.run(run_live_profile(profile, environment={}))
        changed = replace(
            profile,
            weak=replace(profile.weak, base_url="https://other.example.org/v1"),
        )
        with pytest.raises(ConfigurationError, match="manifest mismatch"):
            asyncio.run(run_live_profile(changed, environment={}))
        assert server.calls == []


def test_ambiguous_http_result_requires_duplicate_cost_consent(tmp_path: Path) -> None:
    with _server() as server:
        profile = _profile(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/v1",
            max_calls=1,
            acknowledge_cost=True,
        )
        env = {"WEAK_KEY": "weak-secret", "STRONG_KEY": "strong-secret"}
        server.bad_response = True
        with pytest.raises(ProviderError):
            asyncio.run(run_live_profile(profile, environment=env))
        assert len(server.calls) == 1
        server.bad_response = False
        with pytest.raises(ConfigurationError, match="ambiguous pending"):
            asyncio.run(run_live_profile(profile, environment=env))
        assert len(server.calls) == 1
        with pytest.raises(ConfigurationError, match="acknowledge-duplicate-cost"):
            replace(profile, allow_ambiguous_retry=True)
        retried = replace(profile, allow_ambiguous_retry=True, acknowledge_duplicate_cost=True)
        progress = asyncio.run(run_live_profile(retried, environment=env))
        assert progress.completed_calls == 1 and len(server.calls) == 2


@pytest.mark.parametrize("revision", ["", " ", "r" * 181, 7])
def test_bad_revision_rejected(revision: object) -> None:
    with pytest.raises(ConfigurationError, match="provider revision"):
        LiveProvider("weak", "upstream", revision, "https://example.com/v1", "KEY")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["bad-key", "lower_key", "", 7, "A" * 65])
def test_bad_key_locator_rejected(name: object) -> None:
    with pytest.raises(ConfigurationError, match="API-key environment name"):
        LiveProvider("weak", "upstream", "rev", "https://example.com/v1", name)  # type: ignore[arg-type]


def test_limits_and_retry_cost_consent(tmp_path: Path) -> None:
    profile = _profile(tmp_path, "https://example.com/v1")
    for changes, message in (
        ({"max_calls": 101}, "max_calls"),
        ({"max_calls": True}, "max_calls"),
        ({"max_http_response_bytes": 0}, "HTTP response byte limit"),
        ({"max_http_response_bytes": 1024 * 1024 + 1}, "HTTP response byte limit"),
        ({"timeout_seconds": float("nan")}, "timeout_seconds"),
        ({"allow_ambiguous_retry": True}, "acknowledge-duplicate-cost"),
    ):
        kwargs = {field: getattr(profile, field) for field in profile.__dataclass_fields__}
        kwargs.update(changes)
        with pytest.raises(ConfigurationError, match=message):
            LiveProfile(**kwargs)


def test_non_loopback_http_rejected_even_for_preflight(tmp_path: Path) -> None:
    profile = _profile(tmp_path, "http://example.com/v1")
    with pytest.raises(ConfigurationError, match="loopback"):
        asyncio.run(run_live_profile(profile, environment={}))
    assert not profile.checkpoint_path.exists()


def test_cli_default_preflight_and_redacted_error(tmp_path: Path, capsys: Any) -> None:
    with _server() as server:
        profile = _profile(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        args = [
            "--source",
            str(profile.source_path),
            "--checkpoint",
            str(profile.checkpoint_path),
            "--source-uri",
            profile.provenance.source_uri,
            "--license-id",
            profile.provenance.license_id,
            "--source-sha256",
            profile.provenance.source_sha256,
            "--weak-base-url",
            profile.weak.base_url,
            "--weak-upstream-model",
            profile.weak.upstream_model,
            "--weak-revision",
            profile.weak.revision,
            "--strong-base-url",
            profile.strong.base_url,
            "--strong-upstream-model",
            profile.strong.upstream_model,
            "--strong-revision",
            profile.strong.revision,
        ]
        assert main(args) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["completed_calls"] == 0 and output["total_calls"] == 2
        assert server.calls == []
        assert main([*args, "--max-calls", "1"]) == 2
        captured = capsys.readouterr()
        assert "acknowledge-cost" in captured.err and not captured.out
        assert server.calls == []
