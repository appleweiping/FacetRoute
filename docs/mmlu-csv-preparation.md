# Offline MMLU-shaped CSV few-shot preparation

`facetroute-mmlu-prepare` prepares **caller-owned** headerless, six-column
`<subject>_dev.csv` and `<subject>_test.csv` snapshots. The columns are
`question, A, B, C, D, answer-letter`; the answer letter must be A–D. It uses
only dev answers as few-shot demonstrations, never a test answer. The resulting
JSONL can be read by FacetRoute's existing
[public-score workflow](public-score-workflow.md). Preparation itself is
offline: it does not download datasets, invoke a provider, or require a key.
The repository includes only a tiny synthetic example, not licensed MMLU data.

## Reproducible synthetic example

From the repository root, using the committed fixture byte snapshots:

```bash
facetroute-mmlu-prepare \
  --dev examples/mmlu_csv/demo_math_dev.csv \
  --test examples/mmlu_csv/demo_math_test.csv \
  --subject demo_math \
  --dev-sha256 60812464d52bd3c5b379d48956ab903e9bab3cd239375f8c0dc3cbe4c4a19082 \
  --test-sha256 33ac8df529bf4b1d3b36b8fa25cdadd5b297f419b1751a1dfdaa3abf45fa02fa \
  --dev-source-uri repository:examples/mmlu_csv/demo_math_dev.csv \
  --test-source-uri repository:examples/mmlu_csv/demo_math_test.csv \
  --license MIT --shots 2 --max-prompt-bytes 8192 \
  --output prepared-demo.jsonl > prepared-demo-evidence.json
```

The output is create-only. Remove or choose a **new** output name before
rerunning; an existing file is never overwritten. The CLI exits 2 on
validation or handled write failure. The output is published by a hard link
only after its complete bytes have been flushed, but a directory-fsync or
stdout failure can occur **after** that publication and still produce a
nonzero exit. In that case, do not assume the file is absent: inspect the
output path and SHA-256, and replay against the original sources before using
it. The JSON evidence printed to stdout includes
the declared provenance, exact artifact hash, canonical row hashes, selected
shot count, and SHA-256 of each answer-free provider prompt. It does not print
raw prompts or answers. For a programmatic byte-for-byte replay, call
`verify_mmlu_csv_preparation(artifact, evidence, dev_bytes, test_bytes, plan,
dev_name=..., test_name=...)` with the original pinned snapshots and plan.
The JSONL itself **does include test gold answers** for local scoring: treat it
as private evaluation data, and do not publish it for a licensed dataset.

## Protocol and limits

The subject is a lowercase ASCII slug and must exactly match both source
basenames. This check cannot establish the real semantic subject of a
mislabelled file: the caller is responsible for provenance and licensing.
Both byte snapshots must match the caller-supplied SHA-256 and be strict UTF-8
without BOM. The parser rejects malformed rows, empty/control-character
cells, duplicate choices, duplicate normalized question stems within a split,
and any normalized question-stem overlap between dev and test. It preserves
CSV row order; only the first requested dev rows can become examples.

The tool accepts at most 1 MiB and 20 rows for dev, 1 MiB and 500 rows for
test, 4 KiB per cell, 0–5 requested shots, a 64–32,768 **UTF-8 byte**
provider-prompt ceiling, and a 4 MiB JSONL artifact. It tries the requested
shot count and deterministically removes the last dev example until **every**
test prompt fits; if even zero-shot prompts do not fit, it fails. The selected
count is global across the subject and appears in the evidence. A byte limit
is **not** a tokenizer or model-context limit; callers must separately check
the actual provider/model context window before opting into live scoring.

The generated `few_shot_context` is a separate, bounded MMLU-only field in
the normalized record. It is prepended to the existing public-score MMLU
question/options prompt. The target answer **label** remains in the local
scoring record; it is not added as an answer annotation to
`RouteRequest.query`, metadata, or provider messages. The candidate option
texts are necessarily present in both metadata and the provider prompt.
Changing only a test answer changes local scoring and artifact digests but
not provider prompt bytes. Manually edited or third-party JSONL is not
covered by this guarantee: inspect untrusted content before a live run.

This is FacetRoute's original `facet-mmlu-csv-fewshot-v1` protocol, **not**
the official MMLU evaluation, RouteLLM's prompt/tokenizer setup, published
model responses, or a comparable score. It does not perform contamination
screening, prompt-injection detection, subject-wide aggregation, or paid
provider calls. Do not select shots or providers after seeing test scores and
then report that test as an untouched holdout.
