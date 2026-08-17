"""High-level orchestration: apply any NeuronSelector to a transformer
layer's FFN block via FFNSurgeon.

This is the one place that wires a selection strategy to surgery
mechanics. Everything else in this package depends only on the abstract
`NeuronSelector` / `SaliencyScorer` / `FFNSurgeon` / `FFNLayerAdapter`
contracts, never on each other's concrete classes -- adding a new
pruning criterion, or targeting a new model family, never requires
touching this function. Only `FFNLayerAdapter` is needed here, not the
full `TransformerLayerAdapter` -- this function has no use for attention
internals.
"""
from .ffn_surgery import FFNSurgeon
from .model_adapter import BertLayerAdapter


def prune_ffn_layer(model, layer_index, selector, surgeon=None, adapter=None):
    surgeon = surgeon or FFNSurgeon()
    adapter = adapter or BertLayerAdapter(model)

    intermediate, output = adapter.get_ffn(layer_index)
    keep_indices = selector.select_keep_indices(intermediate.weight.data)
    new_intermediate, new_output = surgeon.resize(intermediate, output, keep_indices)
    adapter.set_ffn(layer_index, new_intermediate, new_output)
    return len(keep_indices)
