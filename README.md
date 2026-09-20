# Transformer Pruning

Extends the pruning criteria from `pruning_framwork_v4` (the thesis
framework: "Pruning and Quantization Techniques for Deep Neural Network
Acceleration", NITK, July 2025) from VGG/ResNet conv channels onto
transformer models (BERT).

This is a separate project from `pruning_framwork_v4` because the two
share a research idea, not code: `pruning_framwork_v4` is a config.ini-driven
framework built around torchvision VGG/ResNet + CIFAR10/IntelIC, while this
project runs on HuggingFace `transformers`/`datasets` against BERT + GLUE.

New to this codebase? See [`docs/KNOWLEDGE_TRANSFER.md`](docs/KNOWLEDGE_TRANSFER.md)
for a full developer onboarding doc — architecture rationale, a
module-by-module reference, extension recipes, and known gotchas.

## Architecture

The package is built around a few small interfaces so that adding a new
pruning criterion, or targeting a new model family, never means touching
scanning, surgery, or orchestration code.

### Interfaces

- **`SaliencyScorer`** (`scoring.py`) — scores a weight tensor's output
  units. `TopKMagnitudeScorer(k)` generalizes the thesis's Max-3 rule
  (`Max3SaliencyScorer` is `k=3`) across conv and linear weights in one
  code path; `LpNormScorer` is the standard magnitude baseline, kept here
  so an experiment can report both from the same scanner.
- **`NeuronSelector`** (`selectors.py`) — decides which FFN neurons
  survive, given an `FFNContext`. Returns a `Selection`.
- **`FFNLayerAdapter` / `AttentionLayerAdapter`** (`model_adapter.py`) —
  reach into a model to get/set a layer's FFN `(intermediate, output)`
  pair, or to read its attention query weight and head config. Split in
  two because no consumer needs both (Interface Segregation).
  `BertLayerAdapter` implements both — it also covers RoBERTa, which
  shares BERT's layout — but a differently-shaped architecture is a new
  adapter class, not an edit to any consumer.

### What a selector gets to see

A selector receives an **`FFNContext`** (`context.py`), not a bare weight
tensor. That matters because a neuron's actual contribution to the layer
output is

```
W_out[:, i] * act(W_in[i] @ x + b_i)
```

so a criterion that sees only `W_in` is reasoning about half the neuron.
The context carries both projections, both biases, and — when the caller
has calibrated — per-neuron activation statistics.

A selector returns a **`Selection`**: the indices to keep, plus an
optional `merge_map`. The map is what lets a redundancy criterion do
something better than deleting its loser: if neuron `j` was flagged
*because* it behaves like neuron `i`, then `W_out[:, j] * a_j` is not
noise to discard, it is signal `W_out[:, i] * a_i` can carry.

### The criteria

| Selector | Sees | Idea |
|---|---|---|
| `SaliencySelector` | `W_in` | Any `SaliencyScorer` — the thesis's Max-3 rule, an Lp norm, or `CSDScorer` |
| `InOutNormSelector` | `W_in`, `W_out` | `‖W_in[i]‖ · ‖W_out[:,i]‖` — a neuron nothing reads is worthless however hard it's driven |
| `ActivationAwareSelector` | `W_out`, activations | `‖W_out[:,i]‖ · E[\|a_i\|]` — measures *use*, not just capacity |
| `TwinRedundancySelector` | firing patterns | Behavioural redundancy (co-activation on real data); can merge rather than drop |
| `WeightClusterRedundancySelector` | `W_in` direction | Weight-space redundancy (K-means on incoming-weight direction); can also merge |

`CSDScorer` scores a neuron by the L1 dispersion of `W_in[i]` from its own
mean, rather than magnitude — a row of near-uniform weights contributes a
roughly constant signal (indistinguishable from a bias), so low dispersion
means low informativeness. Adapted as a one-shot post-hoc criterion from the
group-regularization loss in Thaker & Mohan, *"Enhancing Deep Compression of
CNNs"* (IEEE Access, 2024), which used the same dispersion measure as a
training-time penalty instead.

`WeightClusterRedundancySelector` is the weight-space counterpart to
`TwinRedundancySelector`: instead of behavioral co-activation, it clusters
neurons by `W_in` direction (L2-normalized, via from-scratch k-means in
`clustering.py`) and keeps the highest-L1-norm neuron per cluster, adapted
from the K-Means channel-selection method in Thaker & Mohan, *"Channel
Pruning of Transfer Learning Models Using Novel Techniques"* (IEEE Access,
2024). Where that paper had to condense a 2D conv kernel into a per-channel
feature vector before clustering, a BERT FFN neuron's `W_in` row already
*is* that feature vector, so no condensing step is needed.

Weight-only criteria measure how strongly a neuron is *wired*. They
cannot see that a well-wired neuron may almost never fire on the target
distribution — common in a fine-tuned model, where the pretrained FFN
carries capacity the downstream task never exercises. That is what
calibration supplies.

### Making a cut cost less

Two corrections apply to any criterion, both in `FFNSurgeon`:

- **Bias compensation.** Deleting neuron `i` removes `W_out[:, i] * a_i`
  from the layer output. Its expectation `W_out[:, i] * E[a_i]` is a
  constant, and a constant is exactly what a bias represents exactly — so
  folding it into `output.bias` preserves the layer's mean output for
  free, leaving only the zero-mean residual as real damage. Costs one
  calibration pass.
- **Merging.** Fold a removed neuron's output column into a surviving one.
  When the two are genuinely interchangeable this is output-preserving on
  *every* input, not merely on average.

### Everything else

- **`layers.py`** — `discover_layers`: finds `Conv2d`/`Linear` modules by
  type and name, nothing else.
- **`network_scanner.py`** — `NetworkSaliencyScanner(layers, scorer)`:
  per-layer saliency stats and weakest units for *any* layer set + scorer
  combination.
- **`ffn_surgery.py`** — `FFNSurgeon`: the only code that knows how to
  physically resize an FFN's `(intermediate, output)` Linear pair.
  Written once; every selector shares it.
- **`pruning_workflow.py`** — `prune_ffn_layer` (one layer) and
  `prune_model_ffn` (the whole stack). The one place selection and
  surgery meet.
- **`head_analysis.py`** — `AttentionHeadAnalyzer`: cosine similarity and
  Lp-distance between attention heads, with a `normalize` flag.
- **`attention_surgery.py`** — `AttentionSurgeon`: the compression half of
  head redundancy — physically removes head rows from `query`/`key`/`value`
  and the matching columns from `attention.output.dense`. No merge or bias
  compensation here: a head's contribution is a function of the input, not
  a per-neuron constant. `pruning_workflow.prune_attention_heads` wires it
  to a live model.
- **`calibration.py`** — `FFNCalibrator`: streams batches and accumulates
  per-neuron activation moments without materialising the full
  `(tokens, neurons)` tensor.
- **`activation_recording.py`** — `ActivationRecorder`: records which
  neurons fire on real inputs, keeping full per-token detail (which
  `JaccardTwinFinder` needs and no summary statistic preserves).
- **`redundancy.py`** — `JaccardTwinFinder`: flags neuron pairs with
  near-identical firing patterns (true Jaccard/IoU) as redundant "twins",
  and reports neurons that never fire at all.
- **`clustering.py`** — `kmeans_assign`: from-scratch Lloyd's-algorithm
  k-means, no external ML dependency. Used by
  `WeightClusterRedundancySelector`; a standalone utility because
  clustering is a strategy in its own right, not selector bookkeeping.

## Usage

```python
from pruning_transformer import (
    ActivationAwareSelector, FFNCalibrator, Max3SaliencyScorer,
    SaliencySelector, prune_ffn_layer, prune_model_ffn,
)

# The thesis criterion, unchanged.
prune_ffn_layer(model.bert, 0, SaliencySelector(Max3SaliencyScorer(), prune_percent=40))

# Calibrated, with the mean output restored after the cut.
stats = FFNCalibrator(model.bert).collect(batches, layer_idx=0)
prune_ffn_layer(
    model.bert, 0, ActivationAwareSelector(prune_percent=40),
    stats=stats, compensate_bias=True,
)

# Whole stack, with every neuron ranked against every other rather than
# each layer losing the same fixed fraction.
kept = prune_model_ffn(
    model.bert, ActivationAwareSelector(prune_percent=40),
    allocation="global", stats_by_layer=stats_by_layer, compensate_bias=True,
)  # -> {0: 1900, 1: 1640, 2: 2100, ...}
```

Uniform allocation gives every layer the same fraction; global lets
layers that turn out to be redundant give up more. FFN redundancy is
generally not spread evenly across depth, so uniform over-cuts the layers
carrying the model and under-cuts the ones that aren't.

## Experiments

`experiments/` — runnable scripts, one per exploration stage:

| Script | What it validates |
|---|---|
| `01_vgg_saliency_demo.py` | Vectorized Max-3 score on one VGG16 layer |
| `02_vgg_network_scan.py` | Max-3 stats across all of VGG16 |
| `03_bert_universal_scan.py` | Max-3 crossed over onto BERT-tiny FFN |
| `04_bert_pruning_surgery_demo.py` | Neuron surgery doesn't break a forward pass |
| `05_bert_mrpc_full_experiment.py` | Real accuracy: four criteria compared, baseline → prune → heal, on MRPC |
| `06_attention_head_similarity.py` | Cosine similarity between attention heads |
| `07_attention_head_distance.py` | Manhattan vs. Euclidean head distance |
| `08_attention_scaled_distribution.py` | Distance on unit-normalized heads |
| `09_coactivation_twin_scan.py` | Behavior-based twin-neuron and dead-neuron detection |
| `10_twin_neuron_pruning_experiment.py` | Real accuracy: twins dropped vs. twins merged, on MRPC |
| `11_global_multilayer_pruning.py` | Uniform vs. global budget allocation at equal compression |

Stages 05, 10 and 11 train. Each compares its variants from one shared
trained baseline at one fixed seed (`experiments/_mrpc.py`), so the only
difference between rows is the criterion under test. The interesting
column is *pre-heal* accuracy — healing can paper over a bad cut given
enough epochs, so the pre-heal number is what measures how much the
criterion actually knew.

## Running

No GPU/PyTorch on the primary dev machine for this project (same
constraint as `pruning_framwork_v4`) — the experiment scripts are meant
to run on Colab/Kaggle:

```
pip install -r requirements.txt
python experiments/05_bert_mrpc_full_experiment.py
```

### Tests

The test suite needs neither a GPU nor HuggingFace — the adapters only
need a module tree with BERT's `encoder.layer[i]` *shape*, which
`tests/conftest.py` builds by hand. It runs in well under a second:

```
pip install -e ".[dev]"
pytest
```

## Status

The pruning mechanics are covered by the test suite — surgery preserves
dtype and `requires_grad`, merging identical neurons is exactly
output-preserving, bias compensation exactly preserves the mean output,
budgets are honoured, and dead-neuron detection works.

The **research numbers are not**. No experiment has been run end-to-end
in this repo's layout yet; before citing any accuracy figure, actually
run the corresponding script. Same caveat as `pruning_framwork_v4`'s
unvalidated numbers.

One correctness fix carried over from the original extraction: the
notebook's `find_twins` computed `intersection / (fires_A + fires_B)`,
which is off by a factor vs. true Jaccard/IoU (`intersection / union`,
where `union = fires_A + fires_B - intersection`). `redundancy.py` uses
the corrected formula, so twin-pair counts will differ slightly from the
original notebook's printed output.
