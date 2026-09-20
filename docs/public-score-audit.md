# Offline public-score checkpoint audit

`facetroute-public-score-audit` connects a **completed** MMLU/GSM8K
public-score checkpoint to a separate, caller-supplied router-score file. It
never contacts a provider, modifies the checkpoint, downloads a benchmark, or
claims an official leaderboard result. The existing
[`facetroute-public-score-live`](live-provider-profile.md) command can produce
the checkpoint only after explicit provider/cost consent; synthetic providers
can also produce one without network access.

For every source example, provide exactly one newline-terminated JSONL row:

```json
{"id":"question-1","strong_win_rate":0.8}
```

`id` must match the normalized benchmark example ID. Scores must be finite
numbers in `[0, 1]`, but the audit does **not** assume they are calibrated
probabilities. The caller is responsible for generating them without reading
the held-out answers or responses in the checkpoint. All source IDs must be
present exactly once, including any rows later excluded; this prevents an
incomplete score file from silently selecting only favorable questions.

Optionally, provide a newline-terminated exclusions JSONL file, one digest
per normalized prompt:

```json
{"prompt_sha256":"<lowercase SHA-256 of normalized UTF-8 question text>"}
```

The audit rejects unknown or duplicate digests and excludes every source row
with a matching prompt. It does not determine contamination; an operator must
define, license, and justify the comparison corpus and digest-production
method. The exclusions file's exact SHA-256 is included in the report. Omitting
it means **no decontamination**. Supplying exclusions and scores post hoc can
bias a comparison; register and pin them before examining outcomes.

```bash
facetroute-public-score-audit \
  --source licensed-questions.jsonl \
  --checkpoint private-results/checkpoint.json \
  --route-scores private-results/route-scores.jsonl \
  --exclusions private-results/exclusions.jsonl \
  > private-results/audit.json
```

`--exclusions` is optional. The command emits deterministic strict JSON to
standard output, with source URI/license, source/checkpoint/scores/exclusions
SHA-256 digests, declared model identities, included/excluded counts, overall
weak/strong baselines, and per-category baselines. It emits **no** prompt,
reference answer, predicted answer, or completion text. Category labels and
source/model declarations are echoed and may themselves be sensitive; review
them before sharing the report. Keep the checkpoint private if its response
texts or dataset license require that.

The first point (`threshold: null`) selects the weak model for every included
row. The other points are the sorted unique observed scores. At threshold
`t`, the strong model is selected iff `score >= t`; ties therefore move as a
group. `accuracy` is the fraction of selected outcomes scored correct. For a
point with `k` strong calls, `oracle_accuracy_at_same_calls` starts from all
weak outcomes and replaces the `k` rows with the largest **realized**
strong-minus-weak correctness gains. It is an information-leaking, fixed-call
upper bound, not a feasible routing policy. `regret_to_oracle` is their
difference. The report does not estimate dollar cost, latency, uncertainty,
provider drift, or causal/generalization performance.

Inputs are read once as bounded byte snapshots. The audit verifies exact
source hash, checkpoint JSON seal, manifest/protocol and resource limits,
ordered request digests, per-response predictions and correctness, and
completeness before computing any aggregate. Score and exclusions files use
strict JSONL, bounded file/line/record sizes, exact ID/hash matches, and
file-byte digests in the output. The audit's hard byte ceilings match the
generator's 32 MiB source and 256 MiB checkpoint ceilings. Router-score files
are capped at 64 MiB with each line capped at 32 MiB plus 1 KiB; exclusion
files are capped at 2 MiB with 256-byte lines. Both auxiliary files are also
capped at the number of source records (at most 10,000). A pending or partial
checkpoint cannot be audited. This checks local consistency, not the honesty
of the declared license, model revision, prompt protocol, or supplied router
scores.

RouteLLM's frozen evaluation uses its own prompt construction, model
responses, contamination files, router predictions, and score conventions.
This workflow has not imported those assets or reproduced official MMLU or
GSM8K published scores. It is a reproducible original local comparison
surface; official parity and real-provider verification remain separate work.
