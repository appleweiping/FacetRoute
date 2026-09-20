# Opt-in MMLU/GSM8K public-score workflow

`facetroute.public_scores.run_public_scores` generates weak and strong model
responses for a caller-supplied local MMLU or GSM8K JSON/JSONL file and scores
those responses. It is a separate Python API, not the offline `benchmark` or
`normalize-benchmark` CLI. Nothing is downloaded and no provider is selected
automatically. Calling this API with a network-backed provider can incur cost.

Prepare a licensed local dataset with the [supported record shapes](benchmark-methodology.md).
The caller must declare a public `source_uri`, a `license_id`, and the exact
lowercase SHA-256 of the input bytes. FacetRoute verifies that hash against
the same bounded byte snapshot it parses. The license declaration is an audit
record, not a legal determination. The raw file and ordered normalized rows
are both hashed in the checkpoint manifest.

```python
import asyncio
from facetroute.async_client import AsyncProviderTarget
from facetroute.public_scores import (
    PublicScoreModel,
    PublicScoreProvenance,
    run_public_scores,
)

# `weak_client` and `strong_client` are caller-injected asynchronous providers.
# They can be test fakes, or opt-in facetroute.async_http transports.
weak = PublicScoreModel(AsyncProviderTarget("weak", "model-a", weak_client), "revision-a")
strong = PublicScoreModel(AsyncProviderTarget("strong", "model-b", strong_client), "revision-b")
progress = asyncio.run(
    run_public_scores(
        "licensed-local-questions.jsonl",
        "results/public-score-checkpoint.json",
        provenance=PublicScoreProvenance(
            source_uri="https://example.org/dataset-release",
            license_id="CC-BY-4.0",
            source_sha256="<64 lowercase hex characters from the exact local file>",
        ),
        weak=weak,
        strong=strong,
        max_calls=100,
    )
)
print(progress.completed_calls, progress.total_calls, progress.finished)
if progress.finished:
    print(progress.weak_accuracy, progress.strong_accuracy)
```

The example placeholders must be replaced before running. A native HTTP
provider is available through the optional `facetroute[async]` extra and an
explicit import from `facetroute.async_http`; the default package/CLI remain
free of HTTPX. API keys belong in application environment/secret storage,
never in the dataset, source URI, or checkpoint.

The workflow preflights every record and prompt before making any call. MMLU
provider payloads contain the question and labeled choices, but not which
choice is correct. GSM8K payloads contain the question but never the reference
solution. A leading isolated option letter scores MMLU; the final signed
decimal number (preferably after `####`) scores GSM8K. Missing or malformed
predictions count as incorrect. The completion's reported model must match
the configured upstream model exactly; aliases that return another name need
an explicitly configured exact identity before this workflow can be used.
The model `revision` is declared by the caller, not independently verified
against the provider endpoint or model weights. Change it when the deployment,
endpoint, or weights change; otherwise a resumed checkpoint may silently mix
different models under one manifest. Scores are fractions in `[0, 1]` and are
published only after both models have completed every row. The response text,
prediction, correctness, request digest, model IDs/revisions, source hashes,
source URI, and declared license are retained in the local checkpoint. Never
put that checkpoint in a public repository if responses are restricted.

Every call is preceded by an atomic checkpoint replacement containing a
`pending` marker. On POSIX, the workflow fsyncs the file and containing
directory before sending the request to improve crash durability. On Windows,
replacement is atomic but the API cannot guarantee persistence across sudden
power loss. A successful response replaces the marker with a scored result.
A certified pre-send failure clears the marker; timeout, cancellation,
response error, or a crash
may have reached the provider and leaves it pending. Resume refuses to resend
such a call unless the caller explicitly passes `allow_ambiguous_retry=True`,
accepting potential duplicate cost. Already committed results are never
regenerated on a normal resume. `max_calls` limits new calls in one invocation
and can be set to zero to inspect progress without calling a provider.
Inter-process locking rejects concurrent writers, and resume validates the
manifest, checkpoint checksum, exact result sequence, request digests, and
recomputed scores before any call. Input and checkpoint paths must differ.

Defaults bound source bytes (8 MiB), records (1,000), prompt bytes (32 KiB),
stored completion-text bytes (8 KiB), checkpoint bytes (32 MiB), and provider
time (30 s per call). `max_response_bytes` is only a limit on the extracted
completion text; it does not bound the full provider response. Configure a
separate transport-level response-body limit on each injected provider (for
example, `AsyncOpenAICompatibleProvider(max_response_bytes=...)`) when using
untrusted endpoints. `PublicScoreLimits` permits deliberate changes within
hard ceilings; limits affecting the run are part of the checkpoint identity. The workflow
runs sequentially, so the maximum number of new provider calls is explicit.

This is an independent, transparent scoring protocol, not a reproduction of
RouteLLM's MMLU few-shot/tokenizer setup, GSM8K prompt setup, contamination
filter, provider revisions, published dataset license, or official scores.
Those require a separate pinned evaluation protocol and validation.
