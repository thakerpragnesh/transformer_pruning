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

The package is built around three small interfaces so that adding a new
pruning criterion, or targeting a new model family, never means touching
scanning, surgery, or orchestration code:

- **`SaliencyScorer`** (`scoring.py`) — scores a weight tensor's output
  units. `Max3SaliencyScorer` is the only implementation so far (a
  `torch.topk`-based, GPU-friendly rewrite of `pruning_framwork_v4`'s
  `compute_max3_saliency_score_channel`), but an SVD- or K-means-based
  scorer could be added here without changing anything downstream.
- **`NeuronSelector`** (`selectors.py`) — decides which neuron indices in
  a BERT FFN layer survive pruning. `SaliencySelector` wraps any
  `SaliencyScorer`; `TwinRedundancySelector` instead uses behavioral
  (co-activation) redundancy. Both are interchangeable wherever a
  selector is expected.
- **`FFNLayerAdapter` / `AttentionLayerAdapter`** (`model_adapter.py`) —
  reach into a model to get/set a layer's FFN `(intermediate, output)`
  pair, or to read its attention query weight and head config,
  respectively. Split in two because no consumer needs both:
  `prune_ffn_layer` and `ActivationRecorder` depend only on
  `FFNLayerAdapter`; `AttentionHeadAnalyzer` depends only on
  `AttentionLayerAdapter` (Interface Segregation). `BertLayerAdapter`
  implements both (exposed together as `TransformerLayerAdapter`) since
  BERT has both blocks at the same `encoder.layer[i]` layout, but a
  future adapter needing only FFN support wouldn't have to stub out
  attention methods it has no use for. It's the only implementation so
  far — it also covers RoBERTa, which shares BERT's layout — but a
  differently-shaped architecture is a new adapter class, not an edit to
  any of the three consumers.

Everything else is plain, single-purpose modules consuming those
interfaces:

- **`layers.py`** — `discover_layers`: finds `Conv2d`/`Linear` modules by
  type and name, nothing else.
- **`network_scanner.py`** — `NetworkSaliencyScanner(layers, scorer)`:
  reports per-layer saliency stats and weakest units for *any* layer set
  + scorer combination — this one class covers what used to be two
  near-duplicate classes (a VGG-only scanner and a VGG+BERT scanner).
- **`ffn_surgery.py`** — `FFNSurgeon`: the only code that knows how to
  physically resize a BERT FFN's `(intermediate, output)` Linear pair,
  given a `keep_indices` list. Written once; every selector shares it.
- **`pruning_workflow.py`** — `prune_ffn_layer(model, layer_index,
  selector)`: wires a `NeuronSelector` to `FFNSurgeon` via a
  `TransformerLayerAdapter`. This is the one place selection and surgery
  meet.
- **`attention_similarity` → `head_analysis.py`** —
  `AttentionHeadAnalyzer`: cosine similarity and Lp-distance between
  attention heads, with a `normalize` flag replacing what used to be two
  copy-pasted classes (raw vs. unit-normalized distance).
- **`activation_recording.py`** — `ActivationRecorder`: hooks a layer and
  records which neurons fire on real inputs. Pure data collection.
- **`redundancy.py`** — `JaccardTwinFinder`: pure analysis over a
  recorded firing tensor, flags neuron pairs with near-identical firing
  patterns (true Jaccard/IoU) as redundant "twins" — a criterion not in
  the thesis, exploring whether *behavior* catches redundancy that
  weight-based scoring misses.

Example: the two full pruning experiments differ only in which selector
they hand to the same `prune_ffn_layer`/`FFNSurgeon`:

```python
# Max-3 saliency pruning
selector = SaliencySelector(Max3SaliencyScorer(), prune_percent=40)
prune_ffn_layer(model.bert, layer_index=0, selector=selector)

# Co-activation twin pruning
twins, _ = JaccardTwinFinder().find(recorded_fires, threshold=0.95)
prune_ffn_layer(model.bert, layer_index=0, selector=TwinRedundancySelector(twins))
```

`experiments/` — runnable scripts, one per exploration stage:

| Script | What it validates |
|---|---|
| `01_vgg_saliency_demo.py` | Vectorized Max-3 score on one VGG16 layer |
| `02_vgg_network_scan.py` | Max-3 stats across all of VGG16 |
| `03_bert_universal_scan.py` | Max-3 crossed over onto BERT-tiny FFN |
| `04_bert_pruning_surgery_demo.py` | Neuron surgery doesn't break a forward pass |
| `05_bert_mrpc_full_experiment.py` | Real accuracy: baseline → prune → heal, on MRPC |
| `06_attention_head_similarity.py` | Cosine similarity between attention heads |
| `07_attention_head_distance.py` | Manhattan vs. Euclidean head distance |
| `08_attention_scaled_distribution.py` | Distance on unit-normalized heads |
| `09_coactivation_twin_scan.py` | Behavior-based twin-neuron detection |
| `10_twin_neuron_pruning_experiment.py` | Real accuracy: prune twins → heal, on MRPC |

## Running

No GPU/PyTorch on the primary dev machine for this project (same
constraint as `pruning_framwork_v4`) — these scripts are meant to run on
Colab/Kaggle. Install with:

```
pip install -r requirements.txt
```

Then run any script under `experiments/` directly, e.g.:

```
python experiments/05_bert_mrpc_full_experiment.py
```

## Status

All modules are freshly extracted/cleaned from an exploratory Colab
notebook and have not yet been run end-to-end in this repo's layout —
treat results as needing a real run before citing, same caveat as
`pruning_framwork_v4`'s unvalidated numbers.

One correctness fix made during extraction: the notebook's `find_twins`
computed `intersection / (fires_A + fires_B)`, which is off by a factor
vs. true Jaccard/IoU (`intersection / union`, where
`union = fires_A + fires_B - intersection`). This repo's
`co_activation.py` uses the corrected formula, so twin-pair counts/overlap
percentages will differ slightly from the original notebook's printed
output.
