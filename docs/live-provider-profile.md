# Explicit live-provider public-score profile

`facetroute-public-score-live` connects the local MMLU/GSM8K
[public-score workflow](public-score-workflow.md) to two caller-declared
OpenAI-compatible `/v1` endpoints. Install `facetroute[async]` to make
network calls. Importing the module or requesting `--help` does not import
HTTPX or contact a provider. The default `--max-calls 0` performs only local
input/checkpoint validation and writes the initial checkpoint.

Prepare a licensed local JSON/JSONL file in a supported
[benchmark shape](benchmark-methodology.md), compute the SHA-256 of its exact
bytes, and set each credential in a separate environment variable. For
example, after configuring `FACETROUTE_WEAK_API_KEY` and
`FACETROUTE_STRONG_API_KEY` in your secret manager:

```bash
facetroute-public-score-live \
  --source licensed-questions.jsonl \
  --checkpoint private-results/checkpoint.json \
  --source-uri https://example.org/licensed-dataset-release \
  --license-id CC-BY-4.0 \
  --source-sha256 '<exact lowercase SHA-256>' \
  --weak-base-url https://weak.example.org/v1 \
  --weak-upstream-model declared-weak-model \
  --weak-revision declared-weak-revision \
  --strong-base-url https://strong.example.org/v1 \
  --strong-upstream-model declared-strong-model \
  --strong-revision declared-strong-revision
```

The example endpoints and declarations are placeholders, not evaluated
providers. The command's output is aggregate progress; it does not print
credentials or raw responses. The checkpoint *does* contain response text,
predictions, source identifiers and model declarations, so keep it private.
Only HTTPS or loopback HTTP is accepted. Credentials are read from the
environment variables named by `--weak-api-key-env` and
`--strong-api-key-env`; literal credential flags are intentionally absent.

After checking the zero-call preflight and your provider's price, add
`--max-calls 10 --acknowledge-cost` to make at most ten new calls in this
invocation. The cap is 100 per invocation and applies to both models
together, not to rows. A later invocation with the same declarations resumes
committed results without repeating them. If a request may have reached a
provider but no response was committed, resuming refuses to resend it.
`--allow-ambiguous-retry --acknowledge-duplicate-cost` explicitly accepts
that potential duplicate charge. A safe failure may be retried without
those flags. `--timeout-seconds` and `--max-http-response-bytes` bound each
call; the latter defaults to 64 KiB and is capped at 1 MiB. The workflow
also has its own source, prompt, response-text and checkpoint limits.

The upstream model and revision are caller declarations. The checkpoint
identity also includes a SHA-256 of each declared base URL, so changing the
endpoint refuses to silently continue the same run. The application
checks that the response's reported model matches the declared upstream
model, but it cannot authenticate weights or revisions. This profile has
local loopback-server tests; the CI does **not** call a commercial provider
or reproduce official benchmark scores. It does not download datasets or
establish their license for you.
