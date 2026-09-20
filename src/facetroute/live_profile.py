"""Explicit, secret-safe bridge from local public scores to async HTTP providers.

No provider is contacted by importing this module or by a zero-call preflight.
The optional HTTP transport is imported only for a positive-call run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .async_client import AsyncChatCompletionProvider, AsyncProviderTarget
from .errors import ConfigurationError, FacetRouteError
from .providers import _validate_base_url
from .public_scores import (
    PublicScoreLimits,
    PublicScoreModel,
    PublicScoreProgress,
    PublicScoreProvenance,
    run_public_scores,
)

_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_MAX_HTTP_RESPONSE_BYTES = 1024 * 1024
_MAX_CALLS = 100


class _NoCallsProvider:
    """Manifest-compatible preflight transport that cannot use the network."""

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        raise ConfigurationError("zero-call preflight cannot contact a provider")

    def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        raise ConfigurationError("zero-call preflight cannot contact a provider")


@dataclass(frozen=True, slots=True)
class LiveProvider:
    """One declared upstream identity and an environment-only secret locator."""

    model_id: str
    upstream_model: str
    revision: str
    base_url: str
    api_key_env: str

    def __post_init__(self) -> None:
        if type(self.api_key_env) is not str or not _ENV_NAME.fullmatch(self.api_key_env):
            raise ConfigurationError(
                "API-key environment name must use uppercase letters and digits"
            )
        if type(self.base_url) is not str or not self.base_url.strip():
            raise ConfigurationError("provider base URL must be a non-empty string")
        if type(self.revision) is not str or not self.revision.strip() or len(self.revision) > 180:
            raise ConfigurationError("provider revision must be a bounded non-empty string")


@dataclass(frozen=True, slots=True)
class LiveProfile:
    """One bounded invocation; execution requires explicit cost acknowledgement."""

    source_path: Path
    checkpoint_path: Path
    provenance: PublicScoreProvenance
    weak: LiveProvider
    strong: LiveProvider
    max_calls: int = 0
    acknowledge_cost: bool = False
    allow_ambiguous_retry: bool = False
    acknowledge_duplicate_cost: bool = False
    timeout_seconds: float = 30.0
    max_http_response_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if type(self.max_calls) is not int or not 0 <= self.max_calls <= _MAX_CALLS:
            raise ConfigurationError(f"max_calls must be between 0 and {_MAX_CALLS}")
        if type(self.acknowledge_cost) is not bool or (
            self.max_calls > 0 and not self.acknowledge_cost
        ):
            raise ConfigurationError("provider calls require --acknowledge-cost")
        if (
            type(self.allow_ambiguous_retry) is not bool
            or type(self.acknowledge_duplicate_cost) is not bool
        ):
            raise ConfigurationError("ambiguous-retry controls must be boolean")
        if self.allow_ambiguous_retry and not self.acknowledge_duplicate_cost:
            raise ConfigurationError("ambiguous retry requires --acknowledge-duplicate-cost")
        if (
            type(self.max_http_response_bytes) is not int
            or not 1 <= self.max_http_response_bytes <= _MAX_HTTP_RESPONSE_BYTES
        ):
            raise ConfigurationError("HTTP response byte limit is out of range")
        PublicScoreLimits(timeout_seconds=self.timeout_seconds)


async def run_live_profile(
    profile: LiveProfile, *, environment: Mapping[str, str] | None = None
) -> PublicScoreProgress:
    """Run a preflight or explicitly acknowledged provider-backed score batch.

    Credentials are read only from the supplied environment mapping. Nothing
    prints or serializes a key; transport errors are redacted by the provider.
    """

    if not isinstance(profile, LiveProfile):
        raise ConfigurationError("profile must be a LiveProfile")
    env = os.environ if environment is None else environment

    def bind(binding: LiveProvider) -> PublicScoreModel:
        _validate_base_url(binding.base_url, allow_insecure_http=False)
        provider: AsyncChatCompletionProvider
        if profile.max_calls == 0:
            provider = _NoCallsProvider()
        else:
            key = env.get(binding.api_key_env)
            if not key:
                raise ConfigurationError(
                    "configured provider API-key environment variable is unset"
                )
            try:
                from .async_http import AsyncOpenAICompatibleProvider
            except ImportError:
                raise ConfigurationError("install facetroute[async] for the HTTP profile") from None
            provider = AsyncOpenAICompatibleProvider(
                binding.base_url,
                api_key=key,
                max_response_bytes=profile.max_http_response_bytes,
            )
        return PublicScoreModel(
            AsyncProviderTarget(binding.model_id, binding.upstream_model, provider),
            f"{binding.revision}@{hashlib.sha256(binding.base_url.encode('utf-8')).hexdigest()}",
        )

    return await run_public_scores(
        profile.source_path,
        profile.checkpoint_path,
        provenance=profile.provenance,
        weak=bind(profile.weak),
        strong=bind(profile.strong),
        limits=PublicScoreLimits(timeout_seconds=profile.timeout_seconds),
        max_calls=profile.max_calls,
        allow_ambiguous_retry=profile.allow_ambiguous_retry,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetroute-public-score-live",
        description="Opt-in public scoring; zero calls unless cost is explicitly acknowledged",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    for role in ("weak", "strong"):
        parser.add_argument(f"--{role}-base-url", required=True)
        parser.add_argument(f"--{role}-upstream-model", required=True)
        parser.add_argument(f"--{role}-revision", required=True)
        parser.add_argument(f"--{role}-api-key-env", default=f"FACETROUTE_{role.upper()}_API_KEY")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--allow-ambiguous-retry", action="store_true")
    parser.add_argument("--acknowledge-duplicate-cost", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-http-response-bytes", type=int, default=64 * 1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = LiveProfile(
            source_path=args.source,
            checkpoint_path=args.checkpoint,
            provenance=PublicScoreProvenance(args.source_uri, args.license_id, args.source_sha256),
            weak=LiveProvider(
                "weak",
                args.weak_upstream_model,
                args.weak_revision,
                args.weak_base_url,
                args.weak_api_key_env,
            ),
            strong=LiveProvider(
                "strong",
                args.strong_upstream_model,
                args.strong_revision,
                args.strong_base_url,
                args.strong_api_key_env,
            ),
            max_calls=args.max_calls,
            acknowledge_cost=args.acknowledge_cost,
            allow_ambiguous_retry=args.allow_ambiguous_retry,
            acknowledge_duplicate_cost=args.acknowledge_duplicate_cost,
            timeout_seconds=args.timeout_seconds,
            max_http_response_bytes=args.max_http_response_bytes,
        )
        progress = asyncio.run(run_live_profile(profile))
    except (FacetRouteError, OSError, ValueError) as error:
        print(f"facetroute-public-score-live: error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "completed_calls": progress.completed_calls,
                "total_calls": progress.total_calls,
                "finished": progress.finished,
                "weak_accuracy": progress.weak_accuracy,
                "strong_accuracy": progress.strong_accuracy,
                "manifest_sha256": progress.manifest_sha256,
                "source_file_sha256": progress.source_file_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
