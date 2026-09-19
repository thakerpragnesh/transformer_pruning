"""High-level orchestration: apply any `NeuronSelector` to a
transformer's FFN blocks via `FFNSurgeon`.

This is the one place that wires a selection strategy to surgery
mechanics. Everything else in this package depends only on the abstract
`NeuronSelector` / `SaliencyScorer` / `FFNSurgeon` / `FFNLayerAdapter`
contracts, never on each other's concrete classes -- adding a new pruning
criterion, or targeting a new model family, never requires touching this
module.

`prune_ffn_layer` handles one layer. `prune_model_ffn` handles the whole
stack, and offers the choice that matters at model scale: whether the
budget is spent *uniformly* (every layer loses the same fraction) or
*globally* (every neuron in the model competes against every other, so
layers that turn out to be redundant give up more than layers that are
carrying the model). Uniform is the simpler baseline and the one the
thesis's single-layer experiments generalise to; global usually buys more
accuracy at the same compression, because FFN redundancy is not evenly
distributed across depth.
"""
import torch

from .context import FFNContext, Selection
from .ffn_surgery import FFNSurgeon
from .model_adapter import BertLayerAdapter
from .selectors import ImportanceSelector


def prune_ffn_layer(model, layer_index, selector, surgeon=None, adapter=None, stats=None,
                    compensate_bias=False):
    """Prune one layer's FFN. Returns the number of neurons kept."""
    surgeon = surgeon or FFNSurgeon()
    adapter = adapter or BertLayerAdapter(model)

    intermediate, output = adapter.get_ffn(layer_index)
    ctx = FFNContext.from_layers(intermediate, output, stats=stats, layer_index=layer_index)
    selection = selector.select(ctx)
    new_intermediate, new_output = surgeon.resize(
        intermediate, output, selection, stats=stats, compensate_bias=compensate_bias
    )
    adapter.set_ffn(layer_index, new_intermediate, new_output)
    return len(selection.keep_indices)


def prune_model_ffn(model, selector, layer_indices=None, allocation="uniform", surgeon=None,
                    adapter=None, stats_by_layer=None, compensate_bias=False,
                    normalize="mean", min_keep_ratio=0.1):
    """Prune every FFN block in the model. Returns `{layer_index: n_kept}`.

    `allocation="uniform"` applies `selector` to each layer
    independently -- each layer loses the selector's own
    `prune_percent`.

    `allocation="global"` instead treats the selector's `prune_percent`
    as a *model-wide* budget: it scores every neuron in every layer, ranks
    them all together, and removes the globally weakest. This requires an
    `ImportanceSelector` (a criterion that produces a per-neuron score);
    a purely structural criterion like `TwinRedundancySelector` has no
    scalar to rank across layers and is rejected with an explanation.

    `normalize` controls how per-layer scores are made comparable before
    that global ranking. Raw scores are not: different layers have
    different weight scales, so ranking them directly tends to delete one
    or two low-magnitude layers wholesale rather than finding the genuinely
    redundant neurons. `"mean"` (default) divides each layer's scores by
    that layer's mean, `"median"` is its outlier-resistant counterpart,
    and `"none"` ranks raw scores for callers whose criterion is already
    scale-free.

    `min_keep_ratio` floors how much of any single layer the global mode
    may take, so an unlucky ranking cannot collapse a layer entirely.
    """
    adapter = adapter or BertLayerAdapter(model)
    surgeon = surgeon or FFNSurgeon()
    stats_by_layer = stats_by_layer or {}

    if layer_indices is None:
        layer_indices = list(range(adapter.num_layers()))

    if allocation == "uniform":
        return {
            idx: prune_ffn_layer(
                model, idx, selector, surgeon=surgeon, adapter=adapter,
                stats=stats_by_layer.get(idx), compensate_bias=compensate_bias,
            )
            for idx in layer_indices
        }

    if allocation != "global":
        raise ValueError(f"allocation must be 'uniform' or 'global', got {allocation!r}")

    if not isinstance(selector, ImportanceSelector):
        raise ValueError(
            f"allocation='global' needs a per-neuron score to rank layers against each other, "
            f"but {type(selector).__name__} is not an ImportanceSelector. Use a scoring "
            "criterion (SaliencySelector, InOutNormSelector, ActivationAwareSelector), or "
            "allocation='uniform'."
        )

    keep_by_layer = _global_keep_sets(
        model, selector, layer_indices, adapter, stats_by_layer, normalize, min_keep_ratio
    )

    kept = {}
    for idx in layer_indices:
        intermediate, output = adapter.get_ffn(idx)
        selection = Selection.of(keep_by_layer[idx], num_neurons=intermediate.weight.shape[0])
        new_intermediate, new_output = surgeon.resize(
            intermediate, output, selection, stats=stats_by_layer.get(idx),
            compensate_bias=compensate_bias,
        )
        adapter.set_ffn(idx, new_intermediate, new_output)
        kept[idx] = len(selection.keep_indices)
    return kept


def _global_keep_sets(model, selector, layer_indices, adapter, stats_by_layer, normalize,
                      min_keep_ratio):
    """Rank every neuron model-wide and return `{layer: keep_indices}`.

    Scoring is done up front, before any surgery, so that every layer is
    ranked against the *original* model. Scoring lazily inside the
    resize loop would rank later layers against a model already mutated
    by earlier cuts.
    """
    per_layer_scores = {}
    for idx in layer_indices:
        intermediate, output = adapter.get_ffn(idx)
        ctx = FFNContext.from_layers(
            intermediate, output, stats=stats_by_layer.get(idx), layer_index=idx
        )
        scores = selector.importance(ctx).detach().float().cpu()
        per_layer_scores[idx] = _normalize_scores(scores, normalize, idx)

    flat = torch.cat([per_layer_scores[idx] for idx in layer_indices])
    total = flat.numel()
    n_keep_total = max(1, round(total * (1 - selector.prune_percent / 100)))
    if n_keep_total >= total:
        return {idx: list(range(per_layer_scores[idx].numel())) for idx in layer_indices}

    # Select the global top-k by index rather than by thresholding on the
    # k-th score. With a `>= threshold` test, any tie *at* the threshold
    # is kept in full, so a model with many equal scores (a uniformly
    # weighted layer, or a block of exact zeros) silently keeps more
    # neurons than the budget asked for.
    survivors = torch.zeros(total, dtype=torch.bool)
    survivors[torch.topk(flat, k=n_keep_total).indices] = True

    keep_by_layer = {}
    offset = 0
    for idx in layer_indices:
        scores = per_layer_scores[idx]
        n = scores.numel()
        layer_mask = survivors[offset:offset + n]
        offset += n
        floor = min(n, max(selector.min_keep, int(round(n * min_keep_ratio))))
        if int(layer_mask.sum()) < floor:
            # This layer was hit harder than the floor allows: keep its own
            # strongest `floor` neurons instead of the global verdict. This
            # deliberately spends more than the budget rather than let a
            # ranking artifact collapse a layer.
            keep = torch.topk(scores, k=floor).indices
        else:
            keep = layer_mask.nonzero(as_tuple=True)[0]
        keep_by_layer[idx] = keep.tolist()
    return keep_by_layer


def _normalize_scores(scores, normalize, layer_index):
    if normalize == "none":
        return scores
    if normalize == "mean":
        scale = scores.mean()
    elif normalize == "median":
        scale = scores.median()
    else:
        raise ValueError(f"normalize must be 'mean', 'median' or 'none', got {normalize!r}")
    if not torch.isfinite(scale) or scale.abs() < 1e-12:
        # An all-zero layer would turn into NaNs and poison the global
        # ranking; leaving it unnormalised keeps it (correctly) at the
        # bottom instead.
        return scores
    return scores / scale
