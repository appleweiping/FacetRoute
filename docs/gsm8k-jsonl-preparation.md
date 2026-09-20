# Caller-pinned GSM8K-shaped few-shot preparation

`facetroute-gsm8k-prepare` converts caller-owned train/test JSONL byte snapshots
into a bounded input for FacetRoute's existing injected-provider public-score
workflow. This is a local protocol, **not** the frozen RouteLLM/SGLang GSM8K
prompt, tokenizer, provider revision, official dataset, or published score.
No model or provider is called during preparation.

Each source row is exactly `{"question":"...","answer":"... #### 5"}`. The final
`####` marker must be followed by one finite signed decimal; earlier reasoning
text may appear in `answer`. Sources must be UTF-8 JSONL without blank rows,
duplicate keys, duplicate normalized questions, or train/test question overlap.
The demo inputs under `examples/gsm8k_jsonl/` are fictional, not GSM8K data.

From the repository root, compute the hashes of the exact local files, then
prepare a new output path:

```bash
TRAIN_SHA=$(sha256sum examples/gsm8k_jsonl/demo_train.jsonl | cut -d ' ' -f 1)
TEST_SHA=$(sha256sum examples/gsm8k_jsonl/demo_test.jsonl | cut -d ' ' -f 1)
facetroute-gsm8k-prepare \
  --train examples/gsm8k_jsonl/demo_train.jsonl \
  --test examples/gsm8k_jsonl/demo_test.jsonl \
  --train-sha256 "$TRAIN_SHA" --test-sha256 "$TEST_SHA" \
  --train-source-uri repository:examples/gsm8k_jsonl/demo_train.jsonl \
  --test-source-uri repository:examples/gsm8k_jsonl/demo_test.jsonl \
  --license MIT --shots 2 --output prepared-gsm8k.jsonl \
  > prepared-gsm8k-evidence.json
```

Both source URI and license are **caller declarations**, not verified
authenticity or legal determinations. Treat the prepared JSONL as private: it
contains test gold labels and train demonstration answers. The evidence
contains source/artifact hashes, row and selected-shot counts, and **per-prompt**
hashes, but no raw questions or answers. These digests do not guarantee privacy:
someone with candidate questions or prompts can test guesses against them.
Review the evidence before sharing it. Preserve the original source bytes and
evidence to replay with `verify_gsm8k_jsonl_preparation`; hashes are not digital
signatures.

The first requested train rows become worked Q/A demonstrations. If any test
prompt exceeds the declared byte budget, preparation removes the last
demonstration until **all** test prompts fit or zero-shot also fails. No test
answer enters a provider prompt. Existing GSM8K examples without
`few_shot_context` retain their exact zero-shot request format. The injected
weak/strong provider workflow, resumable checkpoint, and offline threshold
audit are described in [Public-score workflow](public-score-workflow.md) and
[Public-score audit](public-score-audit.md). An injected-provider test checks
that changing only a test answer changes correctness but not request bytes.

Sources are capped at 1 MiB each, 64 train and 500 test rows, 4 KiB per
question/answer, 8 selected shots, and 32 KiB per provider prompt. Output is
create-only and capped at 4 MiB. Source paths are re-read before publication,
but this is not a transactional guarantee against concurrent writers. Use
immutable, access-controlled inputs for consequential evaluation. The caller
must choose licensed, temporally appropriate and uncontaminated partitions;
this tool does not certify any of those properties. Live external-provider
verification remains separate and requires explicit credentials and cost
consent.
