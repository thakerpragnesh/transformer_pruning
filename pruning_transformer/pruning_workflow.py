"""High-level orchestration: apply any NeuronSelector to a BERT layer's
FFN block via FFNSurgeon.

This is the one place that wires a selection strategy to surgery
mechanics. Everything else in this package depends only on the abstract
`NeuronSelector` / `SaliencyScorer` / `FFNSurgeon` contracts, never on
each other's concrete classes -- adding a new pruning criterion never
requires touching this function.
"""
from .ffn_surgery import FFNSurgeon


def prune_ffn_layer(model, layer_index, selector, surgeon=None):
    surgeon = surgeon or FFNSurgeon()
    bert_layer = model.encoder.layer[layer_index]
    intermediate = bert_layer.intermediate.dense
    output = bert_layer.output.dense

    keep_indices = selector.select_keep_indices(intermediate.weight.data)
    new_intermediate, new_output = surgeon.resize(intermediate, output, keep_indices)

    bert_layer.intermediate.dense = new_intermediate
    bert_layer.output.dense = new_output
    return len(keep_indices)
