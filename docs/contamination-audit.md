# Offline contamination-similarity audit

`facetroute-contamination-audit` checks whether each evaluation embedding has
a high-cosine-similarity neighbor in a declared training set. It does not
generate embeddings, use an OpenAI service, download benchmark data, or prove
training-set leakage. The caller must lawfully obtain and
encode both prompt sets with the **same** model and revision. The audit checks
the declaration and dimensions, not the external encoder's true identity.

Each UTF-8 JSON file is a `facet-embedding-set-v1` object:

```json
{
  "schema": "facet-embedding-set-v1",
  "model_id": "fixture-encoder",
  "model_revision": "r1",
  "dimension": 2,
  "records": [
    {"id": "train-1", "embedding": [3, 4]},
    {"id": "train-2", "embedding": [0, 1]}
  ]
}
```

The training and evaluation files use the same schema. IDs are short ASCII
tokens and must be unique within a file. Do not put prompt text or personal
information in IDs. Vectors must be finite and nonzero. For a self-contained
smoke, save the object above as `train.json`; create `eval.json` with the same
metadata and `[{"id":"eval-1","embedding":[1,0]}]` as its `records` value.
Then run:

```bash
facetroute-contamination-audit --training train.json --evaluation eval.json --threshold 0.6
```

Or run the committed synthetic files directly:

```bash
facetroute-contamination-audit --training examples/contamination-train.json --evaluation examples/contamination-eval.json
```

The result is one versioned JSON line containing exact SHA-256 hashes of the
input bytes, model declaration, record counts, scalar-operation preflight,
and at most one nearest training ID and cosine similarity per evaluation ID.
The default threshold is 0.95; inclusive values in [-1, 1] are accepted.
Exact ties select the lexicographically smallest training ID. No vectors or
prompt texts are emitted. A high score is a triage signal that requires
separate provenance and human review; a low score does not rule out semantic
contamination, and model/vector quality is outside this command's proof.

The command snapshots each file (128 MiB maximum), rejects duplicate JSON
keys, non-finite coordinates, invalid/duplicate IDs, mismatched model labels
or dimensions, and enforces 20,000 records and 8 million coordinates per set.
It checks `training_count * evaluation_count * dimension <= 100 million`
before any pairwise comparisons. It does not issue network calls or mutate
inputs. Large reference-scale experiments need an explicitly designed
streaming/indexed path rather than silently bypassing this memory/work bound.

## Link screened IDs to public-score exclusions

The separate `facetroute-contamination-exclusions` command joins **every**
evaluation embedding ID to a normalized MMLU/GSM8K source ID and writes the
prompt-digest JSONL understood by `facetroute-public-score-audit`. This is
useful only if the caller can justify that those vectors actually encode the
named prompts. Equal IDs, model labels, and dimensions are declarations, not
cryptographic proof of a trustworthy encoder or training corpus.

The committed example is wholly synthetic:

```bash
facetroute-contamination-exclusions \
  --source examples/contamination-questions.jsonl \
  --training examples/contamination-train.json \
  --evaluation examples/contamination-eval.json \
  --output exclusions.jsonl
```

The command creates `exclusions.jsonl` only if it does not exist; it never
overwrites an existing path. Standard output is one evidence JSON line with
the exact source, training, evaluation, and exclusion-file SHA-256 digests,
declared encoder identity, threshold, comparison-work count, and matched and
excluded record counts. It contains no prompts, answers, or vectors. Duplicate
normalized prompts share one digest: one flagged ID excludes **all** source
rows with that same prompt, and the reported excluded count reflects this.
The source and evaluation ID sets must match exactly. An explicit zero-hit
screen produces an empty, hash-pinned exclusion file, which the public-score
audit accepts. Preserve the evidence line alongside the source and embedding
provenance before examining weak/strong outcomes; post-hoc choices can bias
the comparison. This still does not reproduce an official benchmark
contamination protocol or prove training-set leakage.
