"""Abstraction over "how to reach into a transformer model", so pruning
orchestration, attention-head analysis, and activation recording never
need to know a specific model family's attribute layout.

Before this, `prune_ffn_layer`, `AttentionHeadAnalyzer`, and
`ActivationRecorder` each hardcoded `model.encoder.layer[i]...` -- fine
while only BERT existed, but it meant those three modules (high-level
policy) depended directly on one concrete model shape (a low-level
detail) instead of on an abstraction (Dependency Inversion). Supporting
a differently-shaped architecture meant editing all three.

Now they depend on adapter interfaces instead. Those interfaces are
split in two -- `FFNLayerAdapter` (get/set the FFN pair) and
`AttentionLayerAdapter` (head config, query weight) -- because no
consumer needs both: `prune_ffn_layer` and `ActivationRecorder` only
touch the FFN side, `AttentionHeadAnalyzer` only touches the attention
side. Keeping them separate means a future adapter for a model that only
needs FFN pruning support implements `FFNLayerAdapter` alone, instead of
being forced to also stub out attention methods it has no use for
(Interface Segregation). `BertLayerAdapter` is the only implementation
so far and satisfies both, since BERT has both blocks at the same
`encoder.layer[i]` layout.
"""
from abc import ABC, abstractmethod

import torch.nn as nn


class FFNLayerAdapter(ABC):
    @abstractmethod
    def get_ffn(self, layer_idx: int):
        """Return (intermediate_linear, output_linear) for the layer's FFN block."""

    @abstractmethod
    def set_ffn(self, layer_idx: int, intermediate: nn.Linear, output: nn.Linear) -> None:
        """Install a resized (intermediate, output) pair back onto the model."""


class AttentionLayerAdapter(ABC):
    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @abstractmethod
    def get_query_weight(self, layer_idx: int):
        """Return the layer's attention query projection weight."""


class TransformerLayerAdapter(FFNLayerAdapter, AttentionLayerAdapter):
    """Convenience union for an adapter that supports both sides, as
    `BertLayerAdapter` does. Consumers should still depend on whichever
    single sub-interface they actually need.
    """


class BertLayerAdapter(TransformerLayerAdapter):
    """Adapter for HuggingFace's `BertModel` layout (`encoder.layer[i]`),
    which RoBERTa's model class also uses.
    """

    def __init__(self, model):
        self.model = model

    @property
    def num_attention_heads(self) -> int:
        return self.model.config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def get_ffn(self, layer_idx: int):
        layer = self.model.encoder.layer[layer_idx]
        return layer.intermediate.dense, layer.output.dense

    def set_ffn(self, layer_idx: int, intermediate: nn.Linear, output: nn.Linear) -> None:
        layer = self.model.encoder.layer[layer_idx]
        layer.intermediate.dense = intermediate
        layer.output.dense = output

    def get_query_weight(self, layer_idx: int):
        return self.model.encoder.layer[layer_idx].attention.self.query.weight.data
