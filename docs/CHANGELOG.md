# Changelog

## Unreleased

### Added

- **`AttentionSurgeon` / `prune_attention_heads`** — the compression half of
  attention-head redundancy: physically removes head rows from `query`,
  `key`, `value` (weight and bias) and the matching columns from
  `attention.output.dense`, so heads `head_analysis.AttentionHeadAnalyzer`
  flags as redundant can actually be cut, not just visualized. There is no
  merge option (unlike `FFNSurgeon`). Bias compensation was also left out
  at first, on reasoning that turned out to be wrong; it now exists (see
  "Head criteria, calibration and model-wide head pruning" below).
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
- The larger open question from 0.2.0's deferral was that cosine
  similarity on raw query weight is a weak redundancy signal, and the OV
  circuit or a data-driven signal would be stronger. It is answered on the
  implementation side below. Which criterion actually wins on a trained
  model is still an empirical question for stage 12.
- **Least-squares merge scales** — `TwinRedundancySelector` /
  `WeightClusterRedundancySelector` now use the variance-minimizing scale
  `s* = E[a_dropped·a_survivor]/E[a_survivor²]` when it's available, instead
  of always falling back to the mean-matching `s = E[a_dropped]/E[a_survivor]`
  (see 0.2.0's "Deferred" below for the math). `E[a_survivor²]` was already
  `CalibrationStats.rms²`; the missing piece,
  `FFNCalibrator.collect_cross_moments(batches, layer_idx, pairs)`, computes
  the off-diagonal `E[a_i·a_j]` for exactly the pairs a caller names —
  never a full `(neurons, neurons)` Gram matrix, which nothing else needs
  and would cost `O(neurons²)` memory unconditionally. Both scale-preserving
  guarantees this replaces still hold: `_bias_compensation` doesn't assume
  which formula produced `scale`, it just corrects whatever mean residual
  results, so mean-output preservation is exact under either scale — the
  least-squares scale's actual payoff is a smaller *pointwise* error, which
  comparing means can't see (see
  `test_least_squares_merge_scale_reduces_pointwise_error_versus_mean_matching`
  in `tests/test_ffn_surgery.py`). Threaded through as an opt-in
  `cross_moments`/`cross_moments_by_layer` argument on `FFNContext`,
  `prune_ffn_layer`, and `prune_model_ffn`; a caller that never collects
  cross moments sees no change in behavior.
- **Head criteria, calibration and model-wide head pruning**, which close
  the gap between `head_analysis` (diagnosis) and `attention_surgery`
  (compression). Until now, every caller of `prune_attention_heads` had to
  invent its own rule for which heads to drop.
  - `head_selectors.py`: `HeadSelector` / `HeadImportanceSelector` (same
    rounding and `min_keep` floor as the FFN selectors), plus four criteria:
    `RedundantHeadSelector`, which greedily drops the weaker head of the
    most similar surviving pair (`similarity="ov"`, or `"query"` for the
    stage-06 baseline); `OVNormHeadSelector` (`‖W_O[:, h] W_V[h]‖_F`);
    `ActivationAwareHeadSelector` (measured contribution, `rms` or `std`);
    and `GradientHeadSelector` (Michel, Levy & Neubig, NeurIPS 2019).
  - `context.HeadContext`: the head counterpart of `FFNContext`.
  - `head_calibration.py`: `HeadCalibrator.collect` / `collect_many` →
    `HeadStats` (`context_mean`, `contribution_rms`, `contribution_std`).
    Streaming: per batch it keeps only a `(heads, d)` sum and a
    `(heads, d, d)` second moment, from which
    `E‖W_O[:, h] ctx_h‖² = tr(W_O[:, h]ᵀ W_O[:, h] E[ctx_h ctx_hᵀ])` is
    exact. `collect_gradient_importance` gives `E_x |∂L(x)/∂ξ_h|`, with the
    gate `ξ_h` applied by a forward pre-hook on the attention output
    projection. transformers 5.x removed `head_mask`, which the paper's code
    used for this. Gradients are taken w.r.t. the gates only, so no
    parameter `.grad` is touched. Per-example gradients are recovered from
    a mean-reduced loss, so batch size doesn't change the result.
  - `head_analysis.ov_norms` / `ov_similarity`, plus
    `AttentionHeadAnalyzer.compute_ov_norms` / `compute_ov_similarity`.
    They go through `(head_dim, head_dim)` Gram blocks
    (`<A_h B_h, A_g B_g>_F = Σ (A_hᵀA_g) ⊙ (B_h B_gᵀ)`) rather than
    materialising every head's `(hidden, hidden)` OV product.
  - `AttentionSurgeon.resize(..., stats, compensate_bias)`: folds each
    dropped head's `W_O[:, h] @ E[ctx_h]` into the output bias.
  - `workflow.prune_attention_layer` (selector-driven, one layer) and
    `prune_model_attention` (uniform or global). Global ranking and floors
    now share one implementation (now `allocation.global_keep_from_scores`) with
    `prune_model_ffn`. `normalize="l2"` (the per-layer normalization Michel
    et al. use) is available to both.
  - `AttentionLayerAdapter.num_layers()`: an optional default, as on
    `FFNLayerAdapter`.
- **`FFNCalibrator.collect_many`** — every layer's stats from one pass over
  the corpus (stage 11 used to make one full pass per layer). The two
  copies of the hook-and-forward loop in `ffn_calibration.py` are now one
  shared helper (now `hooks.run_hooked`), which the head calibrator reuses.
- **`prune_model_ffn` accepts `{layer_index: NeuronSelector}`** (uniform
  allocation only). A `TwinRedundancySelector` holds *one* layer's twin
  pairs, so handing a single instance to `prune_model_ffn` silently applied
  layer 0's twins to every layer. `TwinRedundancySelector.merge_pairs`
  exposes the `(dropped, survivor)` folds, which is what
  `collect_cross_moments` must measure. These are not the twin list
  itself: a chain `(i, j), (j, k)` folds `k` into `i`.
- **Experiments.** New stage 12 compares seven head-pruning variants on
  BERT-small, from one trained baseline. New stage 13 compares FFN-only,
  heads-only and joint compression, with parameter savings. Stage 05 adds
  `WeightClusterRedundancySelector`; stage 10 adds the least-squares merge
  scale; stage 11 calibrates in one pass; stage 06 plots OV-circuit
  similarity next to query similarity. `MrpcHarness.calibration_batches`
  gains `with_labels=` for gradient importance.

### Experiments layout

`experiments/` is grouped by what a script costs to run, not by topic,
since the topic is already in each file name:

| Folder | Stages | What running one means |
|---|---|---|
| `experiments/diagnostics/` | 01–04, 06–09 | no training, seconds to minutes (01–02 need torchvision; 06–08 plot) |
| `experiments/accuracy/` | 05, 10–13 | fine-tune → prune → heal on MRPC; needs a GPU; the only source of research numbers |

Stage numbers are unchanged, so "stage 12" still names one file.
`_mrpc.py` stays at the top because both groups import it (stage 09 uses
`set_seeds`). Each importing script's `sys.path` line now points one level
up. Run scripts by their new path, e.g.
`python experiments/accuracy/05_bert_mrpc_full_experiment.py`; any notebook
calling the old flat path needs updating.

`tests/test_experiments_layout.py` (5 tests) parses the scripts without
running them. It fails if a script that calls `train_baseline` sits outside
`accuracy/` (or one that doesn't sits inside), if stage numbers collide,
if a script doesn't parse, or if an `_mrpc` importer lost its path line.
Both failure modes that matter were checked by breaking them on purpose.
Each `_mrpc` importer was also run for real from an unrelated working
directory and got past `import _mrpc`. The experiment smoke run reproduces
the previous run's numbers.

The test suite stays flat, one file per module, named after it. Grouping it
was considered and declined at this size: the names already map 1:1 to
modules, the cross-cutting files (`test_extension_points`, both layout
tests) belong to no single folder, and three files import `conftest`
directly, which gets fragile in subfolders.

### Package layout

The 21 flat modules of `pruning_transformer/` are now grouped by role into
six subpackages. They form layers, each importing only from the ones before
it, and `tests/test_package_layout.py` enforces the order:

| Subpackage | Role | Modules |
|---|---|---|
| `models/` | reaching into a model | `adapters`, `layers` |
| `measurement/` | running data through it | `hooks`, `ffn_calibration`, `head_calibration`, `activation_recording` |
| `analysis/` | signals from weights / firings | `scoring`, `network_scanner`, `head_analysis`, `redundancy`, `clustering` |
| `selection/` | criteria | `context`, `budget`, `ffn_selectors`, `head_selectors` |
| `surgery/` | resizing modules | `linear_ops`, `ffn_surgery`, `attention_surgery` |
| `pipeline/` | orchestration | `allocation`, `workflow` |

**The public API is unchanged:** everything is still imported from
`pruning_transformer`, and no experiment changed. **Submodule paths did
change**, so code importing e.g. `pruning_transformer.selectors` directly
must use the new path. The project never supported that (the API rule has
always been "import from the top level"), and only one test did it.

Four modules were renamed. They either stuttered inside their folder or
broke an FFN/head naming pair. The others kept their names, and git
records every move as a rename:

| Old | New |
|---|---|
| `model_adapter.py` | `models/adapters.py` |
| `calibration.py` | `measurement/ffn_calibration.py` (pairs with `head_calibration.py`) |
| `selectors.py` | `selection/ffn_selectors.py` (pairs with `head_selectors.py`) |
| `pruning_workflow.py` | `pipeline/workflow.py` |

Renaming `selectors.py` also removes a real hazard. With the package
directory on `sys.path` (as when running Python from inside it), the module
shadowed the standard-library `selectors` that `subprocess` imports, and
the interpreter failed at import. The test files for the renamed modules
follow suit: `test_ffn_calibration.py`, `test_ffn_selectors.py`,
`test_workflow.py`.

`tests/test_package_layout.py` (4 tests) parses every import, including
`TYPE_CHECKING` ones. It fails on any upward or top-level import, on a
module left outside the six subpackages, and on a duplicate module name
(docs refer to modules by bare name). A deliberate violation of each was
checked to fail with the offending `file:line`.

### SOLID pass

A structural pass over the package, driven by concrete findings rather
than taste. Each item below was a specific violation found in the code.
The public API is backward compatible: every existing signature, default,
string shorthand and error message still works. The test suite was not
edited to accommodate any of it.

- **Dependency Inversion: adapters are resolved, not hardcoded.** Nine call
  sites (both calibrators, the recorder, the analyzer, five workflow
  functions) each defaulted to `adapter or BertLayerAdapter(model)`, so
  high-level code named the one concrete adapter and a second model family
  meant passing `adapter=` at every call. Now every consumer calls
  `adapters.resolve_adapter(model, adapter)`, backed by a registry:
  `register_adapter(matches, factory)` teaches every consumer a new family
  at once (Open/Closed), and returns an `unregister` function. Resolution
  also unwraps a HuggingFace task model via `base_model`. That removes the
  `HeadCalibrator(model, adapter=BertLayerAdapter(model.bert))` workaround
  gradient importance needed: `HeadCalibrator(model)` now hooks the
  encoder while the loss comes from the classifier.
- **Open/Closed: budget allocation is a strategy.** `prune_model_ffn` and
  `prune_model_attention` were near-duplicates, each with an
  `if allocation == "uniform" / "global"` switch. New `allocation.py` has
  `AllocationStrategy` (`UniformAllocation`, `GlobalAllocation`) working
  over a three-method `PrunableStructure` interface. `workflow`
  implements that interface once for FFN neurons and once for heads. Both
  functions accept a strategy object as `allocation=`, so a new policy is
  a class, not an edit; `"uniform"` / `"global"` remain shorthands. A side
  effect: `prune_model_attention` accepts per-layer selector mappings too,
  since that logic now lives in `UniformAllocation` rather than in one of
  the two functions.
- **Open/Closed: registries instead of `if`/`elif` chains**, following the
  existing `LP_METRICS` convention. `allocation.SCORE_NORMALIZERS` replaces
  the score-normalization chain; `head_selectors.HEAD_SIMILARITIES` replaces
  `RedundantHeadSelector`'s `ov`/`query` branch. A new normalization or
  redundancy signal is a new entry.
- **Interface Segregation: `AttentionLayerAdapter` asks for three members,
  not five.** `hidden_size` was abstract, so every adapter had to implement
  it, yet nothing read it. It is no longer required (`BertLayerAdapter`
  keeps it). `get_query_weight` is now a default derived from
  `get_attention_heads` rather than a fourth thing to write.
  `num_layers()`, defined twice with two different messages, now lives
  once on a shared `LayerStackAdapter` base, with `resolve_layers()`
  replacing four copies of "`None` means every layer".
- **Single Responsibility: one home per mechanism.**
  - `hooks.py`: the hook-and-forward loop, padding masking and device
    lookup. Before, this was a third copy in `ActivationRecorder`, a second
    `_model_device`, and private helpers `head_calibration` imported from
    `calibration`.
  - `budget.py`: budget validation and keep-count rounding, as
    `PruneBudgetMixin`. Four selectors each validated and stored the budget
    themselves, and `head_selectors` imported `selectors`' private helpers.
  - `linear_ops.rebuild_linear`: building a resized `nn.Linear` with the
    source's dtype, device and `requires_grad` flags. These guarantees
    (0.2.0's fp16 bug lived here) were written out three times across the
    two surgeons. `FFNSurgeon.resize` is now split into
    validate / fold-merges / compensate / rebuild steps.
  - `experiments/_mrpc.py`: `train_baseline`, `run_variant` and `compare`.
    Five scripts each repeated the clone → prune → fresh Trainer →
    evaluate → heal loop. That loop is exactly where 0.2.0's bug 4 lived
    (a reused Trainer healed detached parameters), and the fresh-Trainer
    rule is now enforced in one place instead of by convention in five.
    Each script is now just its list of variants. They also pass the task
    model directly instead of reaching into `model.bert`.
- `context.py` typed its stats fields as `Any` "to avoid a cycle" that did
  not exist; they are now real annotations under `TYPE_CHECKING`.

**Verification.** `tests/test_extension_points.py` adds 20 tests that use
each extension point as an extender would:

- a brand-new model layout registered once and then calibrated, pruned
  model-wide and run, with no `adapter=` anywhere;
- a three-method attention adapter serving the analyzer and head pruning;
- a custom allocation strategy driving both FFN and head pruning;
- new registry entries for score normalization and head similarity;
- identical budget validation across all four budgeted selectors;
- `rebuild_linear`'s guarantees.

The experiment smoke run now subclasses the real `MrpcHarness`, so the new
protocol is exercised. It reproduced the pre-refactor run's kept counts
and gradient importances exactly. Against a real transformers 5.17
`BertForSequenceClassification`, `resolve_adapter` unwraps to the encoder
and gives bit-identical gradient importance to the explicit-adapter path.

### Fixed

- **`TwinRedundancySelector` depended on pair order.** Each pair was
  resolved upward from its lower index on its own, so `(4, 6)` arriving
  before `(1, 4)` (as `JaccardTwinFinder`'s overlap ordering allows)
  mapped 6 into 4 and then dropped 4. That is a merge into a removed
  neuron, and `Selection.of` raises on it. Pairs sharing a neuron could
  also overwrite each other's survivor. Now a union-find in which every
  connected twin group collapses onto its lowest index.
- **`WeightClusterRedundancySelector` over-pruned when K-Means left clusters
  empty**, which always happens when there are fewer distinct directions
  than clusters (duplicate or all-zero rows): one survivor per non-empty
  cluster kept fewer neurons than `prune_percent` allows. The shortfall
  is now refilled with the highest-L1 remaining neurons.
- **`kmeans_assign` mixed devices on GPU**, in two places: the initial
  assignment and the `k == n` shortcut were created on the CPU, and
  `torch.equal` raises across devices.

### Correction

- **Heads *can* be bias-compensated.** The attention-surgery entry above,
  `attention_surgery.py`'s docstring and the README all said there was no
  compensation for heads because "a head's contribution is a function of
  the input, not a per-neuron constant a bias can absorb". That argument
  proves too much. An FFN neuron's `W_out[:, i] * a_i` is just as
  input-dependent, and FFN compensation never claimed to absorb it: in
  both cases only the *expectation* goes into the bias. For a head that
  turns zero-ablation into mean-ablation, the standard and much gentler
  way interpretability work removes heads. The mean-preservation tests
  cover heads now, in `tests/test_attention_surgery.py` and
  `tests/test_pruning_workflow.py`. Merging heads is still not offered,
  for a different and sound reason: two heads that attend alike can read
  and write different subspaces, and no survivor's `head_dim`-wide value
  space can absorb another's in general.

### Verification

- 169 tests at the time (189 after the SOLID pass above; one CUDA-only, skipped without a GPU), under a second, still
  no GPU or HuggingFace. Two checks are
  independent of the implementation: `HeadStats` against an explicit
  per-token computation, and gradient importance against central finite
  differences. The finite differences scale a head's `W_O` columns, which
  is exactly scaling its gate.
- Offline against a real transformers 5.17 `BertForSequenceClassification`
  (random init, built from a config; eager and SDPA attention):
  - pruned logits equal zero-gating the same heads, max |Δ| 7e-9;
  - gradient importance on the model's own loss matches finite
    differences, relative error ≤ 1e-6;
  - compensated head pruning preserves the layer's mean output, drift 4e-9;
  - heads, then re-calibrated FFN, then backward still runs.
- Stages 05 and 10–13 were run end to end against a stand-in harness: a
  real HF BERT, synthetic batches, and a few SGD steps in place of
  `Trainer`. That checks their wiring, nothing more. **Still no accuracy
  number has been produced in this repo layout**; that needs a GPU run of
  the real scripts.

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

Both items below are implemented now — see "Unreleased" above — kept here
for the original reasoning.

- **Attention-head pruning surgery.**
- **Least-squares merge scales.** `TwinRedundancySelector` uses
  `s = E[a_j]/E[a_i]`, which only matches the *means*. Minimizing
  `E[(a_j − s·a_i)²]` gives `s* = E[a_i·a_j]/E[a_i²]`. The denominator is
  already stored (`rms²`); only the off-diagonal `E[a_i·a_j]` is missing, and
  `JaccardTwinFinder`'s Gram matrix can't supply it (it is over *binarized*
  firing). The intercept form `c = E[a_j] − s·E[a_i]` is already exactly what
  `FFNSurgeon._bias_compensation` folds into the bias, so only `s` needs to
  change.
