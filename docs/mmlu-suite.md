# Offline MMLU-shaped multi-subject suite

`facetroute-mmlu-suite` combines 1–64 caller-owned headerless six-column
`<subject>_dev.csv` and `<subject>_test.csv` pairs into one private-gold JSONL
file for the existing public-score workflow. Each subject uses only its own
dev examples as demonstrations; no test answer is placed in a provider prompt.
It does not download data, invoke a model, reproduce RouteLLM's tokenizer or
official MMLU protocol, or establish a comparable published score.

Run the committed **fictional** two-subject example from the repository root:

```bash
facetroute-mmlu-suite \
  --manifest examples/mmlu_csv/suite-demo.json \
  --output prepared-suite.jsonl > prepared-suite-evidence.json
```

The manifest is strict UTF-8 JSON with one `subjects` array. Each entry has
exactly `subject`, `dev`, `test`, `dev_sha256`, `test_sha256`,
`dev_source_uri`, `test_source_uri`, `license`, `shots`, and
`max_prompt_bytes`. Source paths are relative to the manifest directory,
must remain inside it even through symlinks, and must have the corresponding
`<subject>_dev.csv`/`<subject>_test.csv` basename. Source bytes must match
the declared lowercase SHA-256 hashes. The source URI and license are caller
declarations, not proof that the data are authentic or licensed.

Subjects are sorted by name in the artifact, independent of manifest order.
The per-subject [few-shot preparation rules](mmlu-csv-preparation.md) still
apply, including per-source byte, row, prompt and shot limits. This layer
also rejects a normalized test question that appears in *another* subject's
dev split, so a demonstration cannot silently leak a held-out question.
It accepts at most 64 subjects, 64 MiB combined source bytes, a 1 MiB
manifest, and a 64 MiB final JSONL. The output is create-only; failed
validation leaves no output, but a post-publication directory-sync/stdout
failure can leave a valid published file despite a nonzero exit. Inspect the
artifact hash and replay before using it.

The stdout evidence contains the exact manifest hash, aggregate artifact
hash, ordered subject names, record count, and each subject's source,
canonical-row, artifact, and answer-free provider-prompt hashes. It omits
prompt text and gold answers. `verify_mmlu_suite(artifact, evidence,
sources)` recomputes and byte-compares the API-level suite evidence; for CLI
output, compare its nested `suite` object to the parsed API evidence and its
`manifest_sha256` to the exact manifest bytes. The JSONL itself **contains
test gold labels** for local scoring and must not be published when using
licensed evaluation data.

This feature supports the shape of a many-subject evaluation, not official
data or official score parity. The repository includes only the two tiny
fictional subjects above. A real licensed MMLU corpus, provider identities,
prompt protocol, contamination screening, and external-provider CI remain
outside this slice.
