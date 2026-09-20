# Changelog

## Unreleased

### Added

- **`AttentionSurgeon` / `prune_attention_heads`** — the compression half of
  attention-head redundancy: physically removes head rows from `query`,
  `key`, `value` (weight and bias) and the matching columns from
  `attention.output.dense`, so heads `head_analysis.AttentionHeadAnalyzer`
  flags as redundant can actually be cut, not just visualized. There is no
  merge/bias-compensation option here (unlike `FFNSurgeon`) — a head's
  contribution is a function of the input, not a per-neuron constant a bias
  can absorb.
- Fixed the `config.num_attention_heads` gotcha the surgery would otherwise
  hit (see 0.2.0's "Deferred" below): `AttentionLayerAdapter.num_attention_heads`
  is now a per-layer method reading the self-attention module's own
  `num_attention_heads`/`attention_head_size`/`all_head_size`, matching how
  HuggingFace's own `prune_heads()` tracks a post-prune head count — and
  `set_attention_heads` updates those three attributes together, since
  resizing the weights without them gives silently wrong attention, not a
  shape error. `head_analysis.get_flat_heads` now derives head size from the
  query weight's own shape rather than the model-wide `hidden_size`, so
  analysis stays correct on a layer that has already been pruned.
- The larger open question from 0.2.0's deferral — cosine similarity on raw
  query weight is a weak redundancy signal; the OV circuit or an
  attention-pattern similarity on real data would be stronger — is still
  open. `prune_attention_heads` takes head indices directly rather than a
  selector, so whichever criterion answers that question can drive it
  without a package change.

## 0.2.0 — 2026-09-19

A correctness, criteria, and infrastructure pass over the whole project.
Section 10.1 of [`KNOWLEDGE_TRANSFER.md`](KNOWLEDGE_TRANSFER.md) carries the
same bug list in onboarding form; this file additionally records the
*evidence* each claim rests on, and two mistakes made and corrected during
the work itself.

### Correctness fixes

Each of these produced plausible-looking output rather than an error, which
is why they survived. Anything measured before this commit should be
re-measured.

| # | Bug | Consequence |
|---|---|---|
| 1 | `JaccardTwinFinder` dead-neuron branch was unreachable — two never-firing neurons have `intersection = union = 0`, and the divide-by-zero guard mapped that to similarity 0, which no threshold accepts | Stage 09 always printed "0 dead neurons", and it read as a finding |
| 2 | `head_analysis.compute_distance_matrix` called `.numpy()` without `.cpu()` | Crashed on any CUDA model; `compute_similarity` had always done it correctly, so nobody hit it |
| 3 | `FFNSurgeon` built bare `nn.Linear`s in default float32 | Silently converted a half-precision model's FFN, mismatching every other layer |
| 4 | Experiments 05/10 reused one `Trainer` across surgery; its cached optimizer held pre-surgery `nn.Parameter` objects | "Healing" stepped detached parameters — the healed number measured nothing |
| 5 | Nothing was seeded | On MRPC, "pruning cost 1.2%" is indistinguishable from run-to-run variance |
| 6 | `SaliencySelector` truncated the keep count with `int()` | One-directional: keeps one neuron fewer than asked whenever `n·(1−p/100)` isn't whole (768 @ 40% → 460, not 461). Never errs the other way, so a whole-model sweep drifts past the nominal budget |

Surgery additionally now preserves `requires_grad`, handles `bias=None`, and
validates keep-indices and FFN-pair consistency.

### Pruning quality

Selectors now receive an `FFNContext` (both projections, both biases, optional
calibration statistics) and return a `Selection` (keep-list plus an optional
merge map), rather than taking a bare `intermediate.weight` and returning a
list. The old signature structurally limited every criterion to half the
neuron: a neuron's contribution is `W_out[:,i] · act(W_in[i] @ x + b_i)`.

New criteria and corrections:

- **`InOutNormSelector`** — `‖W_in[i]‖ · ‖W_out[:,i]‖`. Product form, so a
  near-zero factor on either side sinks the neuron; a sum would let a large
  input norm mask a dead output column.
- **`ActivationAwareSelector`** — `‖W_out[:,i]‖ · E[|a_i|]`, via the new
  `calibration.py`. Measures use rather than capacity.
- **Bias compensation** — deleting neuron `i` removes `W_out[:,i]·a_i`; its
  expectation is a constant, and a constant is exactly what a bias represents
  exactly, so the layer's mean output is preserved for free.
- **Merging** — fold a removed neuron's output column into a survivor instead
  of discarding it.
- **`prune_model_ffn(allocation="global")`** — ranks every neuron model-wide
  instead of giving each layer the same fixed fraction.

### Performance

- `ActivationRecorder`: batched (was batch-size-1), `bool` storage (4×),
  padding masked, post-activation hook, no longer stateful across calls
  (activations used to accumulate on the instance, so a second `record()`
  returned the first call's rows too).
- `JaccardTwinFinder`: vectorized upper-triangle selection, no numpy
  round-trip, no Python loop over the full matrix.
- `NetworkSaliencyScanner`: score caching, one host sync per layer instead of
  three.

### Infrastructure

- `tests/` — 86 tests, ~0.45s, no GPU and no HuggingFace. `conftest.py` builds
  `StubBert` by hand because the adapters need BERT's `encoder.layer[i]`
  *shape*, not HuggingFace itself.
- `pyproject.toml`; `requirements.txt` given lower bounds (`transformers>=4.46`
  for `eval_strategy`).
- `experiments/_mrpc.py` — shared `MrpcHarness`, replacing ~50 lines duplicated
  between stages 05 and 10. Two copies of a training setup is two chances for
  the baselines to drift, which would make the criterion comparison meaningless.
- Experiments 05 and 10 rewritten to compare variants from one shared trained
  baseline; new stage 11 contrasts uniform vs global allocation.

### Verification

What the test suite actually pins, beyond "it runs":

- Merging an interchangeable neuron is **exactly** output-preserving on every
  input, not merely on average.
- Bias compensation **exactly** preserves the layer's mean output.
- Merge and compensation do not double-count (compensation adds only the
  residual `E[a_j] − s·E[a_i]`, which is zero when `s = E[a_j]/E[a_i]`).
- Surgery preserves dtype and `requires_grad`; `bias=None` survives.
- Global allocation honours its budget exactly and floors each layer.

Smoke run on a random-weight stub (3 layers × 16 neurons, 40% cut, mean
absolute output deviation from the unpruned model):

| Criterion | uniform | global |
|---|---|---|
| Max-3 | 0.1147 | 0.1252 |
| Lp-norm | 0.1147 | 0.1139 |
| In/out norm | 0.1092 | 0.1202 |
| Activation-aware | **0.0913** | 0.0945 |

Twin pruning on a planted exact twin: dropping gives max|Δ| = 0.278, merging
gives **0.000000**.

> **These are mechanism checks, not research results.** The weights are random
> and the model is a stub. They confirm the arithmetic is right; they say
> nothing about whether activation-aware pruning beats Max-3 on a real
> fine-tuned BERT. No accuracy number has been produced in this repo layout
> yet — run stages 05, 10 and 11 before citing anything.

### Two corrections made during the work

Recorded because both were confidently stated before being checked.

1. **A tie-handling bug in global allocation, found by a failing test.** The
   first implementation picked a threshold as the k-th largest score and kept
   everything `>= threshold`. Any tie *at* the threshold is then kept in full,
   so a model with many equal scores kept 40 neurons against a 34 budget. Now
   uses exact global top-k by index.
2. **The `int()` truncation example was wrong.** It had been written up — in a
   code comment, a test docstring, and the onboarding doc — as "`int(10 * 0.6)`
   keeps 5, a 50% cut for a 40% ask". `10 * 0.6` is exactly `6.0`. The bug is
   real (`int()` differs from `round()` in 19,283 of 45,056 size/percent
   combinations) but the mechanism is the duller one now documented: a
   consistent off-by-one toward over-pruning. The test was also not
   distinguishing `int` from `round` on its example values; it now does.

### Deferred

- **Attention-head pruning surgery.** Implemented since — see "Unreleased"
  above.
- **Least-squares merge scales.** `TwinRedundancySelector` uses
  `s = E[a_j]/E[a_i]`, which only matches the *means*. Minimizing
  `E[(a_j − s·a_i)²]` gives `s* = E[a_i·a_j]/E[a_i²]`. The denominator is
  already stored (`rms²`); only the off-diagonal `E[a_i·a_j]` is missing, and
  `JaccardTwinFinder`'s Gram matrix can't supply it (it is over *binarized*
  firing). The intercept form `c = E[a_j] − s·E[a_i]` is already exactly what
  `FFNSurgeon._bias_compensation` folds into the bias, so only `s` needs to
  change.
