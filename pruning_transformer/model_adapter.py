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

    def get_activation_module(self, layer_idx: int) -> nn.Module:
        """Return the module whose output is the FFN's *post-activation*
        hidden tensor -- i.e. exactly what the output projection consumes.

        Deliberately not abstract. Only `calibration.FFNCalibrator` and
        `activation_recording.ActivationRecorder` need it, so an adapter
        written purely to enable weight-based pruning shouldn't be forced
        to implement it (same Interface Segregation reasoning that split
        `FFNLayerAdapter` from `AttentionLayerAdapter` in the first
        place). The default explains what to override rather than
        failing obscurely.

        Note this is *not* `get_ffn(...)[0]`: that returns the
        intermediate `nn.Linear`, whose output is pre-activation. For
        sign-based firing detection the two agree (GELU and ReLU are both
        positive exactly where their input is), but for any magnitude
        statistic they differ, which is why calibration hooks here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_activation_module(), so it cannot "
            "be used for activation calibration or recording. Override it to return the "
            "module emitting the post-activation FFN hidden state (for BERT: "
            "`model.encoder.layer[i].intermediate`)."
        )

    def num_layers(self) -> int:
        """How many transformer layers the model has.

        Non-abstract for the same reason as `get_activation_module`: only
        the whole-model `prune_model_ffn` entry point needs it, and an
        adapter used solely for single-layer work shouldn't have to
        supply it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement num_layers(), which "
            "`pruning_workflow.prune_model_ffn` needs to enumerate layers. Either override it "
            "or pass an explicit `layer_indices=` list."
        )


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

    def get_activation_module(self, layer_idx: int) -> nn.Module:
        # BertIntermediate == dense + intermediate_act_fn, so its output
        # is post-activation; `layer.intermediate.dense` would not be.
        return self.model.encoder.layer[layer_idx].intermediate

    def num_layers(self) -> int:
        return len(self.model.encoder.layer)

    def get_query_weight(self, layer_idx: int):
        return self.model.encoder.layer[layer_idx].attention.self.query.weight.data
