# Pairwise low-rank router

`FactorizationRouter` learns a conditional route preference from strict,
labelled counterfactual traces. It is an offline-first baseline, not a claim to
reproduce RouteLLM's neural/text-encoder architecture. It has no external ML
runtime or remote model call. Use it when each training request has a
`preferred_model` and observed outcomes for at least one competing model.

## Model and optimization

The fitted encoder makes a bounded TF-IDF request vector `x` from **training
only**. The encoder configuration and vocabulary are stored inside the factor
state. With `d` latent dimensions, `W` is a `d × features` projection and
`P[r]` is a `d`-vector for route `r`:

```text
score(r, x) = P[r] · (W x)
margin(winner, loser, x) = score(winner, x) - score(loser, x)
loss = log(1 + exp(-margin))
```

For each trace, the preferred route is compared with each other observed
route. Training minimizes the logistic pairwise loss using deterministic,
request-ID-sorted SGD with shrinkage on updated latent factors and active
projection coordinates. SHA-256-derived initial weights avoid
ambient randomness. `seed`, dimension, epochs, learning rate,
regularization, feature/record/pair limits, and the derived update-work limit
are explicit. The fitted state includes a training digest and final loss.

This is a low-rank bilinear classifier over lexical/request features. It can
learn context-dependent preferences, but cannot recognize semantic synonyms
outside its vocabulary, infer a route absent from training, or estimate
counterfactual quality where outcomes were not observed.

## Training and held-out evaluation

```bash
facetroute train-factorization \
  --train-traces artifacts/split/train.jsonl \
  --held-out-traces artifacts/split/test.jsonl \
  --group-by user_id \
  --dimension 8 \
  --epochs 30 \
  --output artifacts/factor-model.json \
  --report artifacts/factor-report.json

facetroute route \
  --models examples/models.json \
  --policy factorization \
  --factor-model artifacts/factor-model.json \
  --query "Design edge cases for a parser"
```

Only training traces fit the vocabulary and weights. The optional held-out
file is scored once and must be disjoint from training at `request_id`,
`user_id`, or a declared `metadata:<field>` group key. The report records
training and held-out digests, top-1 and pairwise accuracy, and pairwise
log-loss. Top-1 accuracy ranks only the routes with outcomes observed on that
row; rows with fewer than two observed routes are excluded and the report
records `top1_evaluable_records` (or `null` accuracy if none qualify). A
held-out label or observed route absent from the fitted route set is rejected,
not silently removed from the denominator. Choose hyperparameters
on a separate validation partition before using the final test partition.

`benchmark --policy factorization --factor-model ...` replays the same
observed traces as the other policies and hashes the exact loaded state
snapshot in its manifest. `simulate` and `serve` accept the same policy/model
pair as `route`.

## Routing and state boundary

Capability, context, region, budget, enabled-state, restricted-data, and
profile-block constraints run before learned ranking. The highest logit among
eligible trained routes wins. When no trained route remains eligible, the
existing objective scorer is used. Soft profile/rule bonuses are recorded in
the explanation but do not overwrite the learned logit; hard profile blocks
still filter. Ties use stable route-ID order.

The state is versioned JSON containing the exact encoder, sorted routes, route
factors, projection, bounded settings, and canonical SHA-256 checksum. Writes
are atomic and preflight the 32 MiB state limit; loads reject duplicate keys,
oversized files, mismatched checksums, malformed dimensions, non-finite
numbers, and unknown catalog routes. State produced by a different
implementation/schema is not silently accepted.
