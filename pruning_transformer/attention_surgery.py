"""Physically removes attention heads from a transformer layer's Q/K/V and
output projections.

`head_analysis.AttentionHeadAnalyzer` finds which heads look redundant;
this is the other half -- turning a list of head indices into a smaller
model. Q, K, and V each partition their *output* rows into `num_heads`
contiguous blocks of `head_dim` rows (HuggingFace's layout: head `h` owns
rows `[h*head_dim, (h+1)*head_dim)`); the attention output projection
partitions its *input* columns the same way, since its input is the heads'
concatenated context vectors. Removing head `h` therefore means dropping
those rows from Q/K/V (weight and bias) and those columns from the output
projection -- never partial rows or columns, or the surviving heads' math
would be corrupted.

Unlike `ffn_surgery.FFNSurgeon`, there is no merge or bias-compensation
option here: a dropped head's contribution is a function of the input (its
attention pattern over the sequence), not a per-neuron constant a bias term
can absorb, so there is no equivalent cheap correction to apply at surgery
time. Updating a model's head-count bookkeeping (`num_attention_heads` /
`attention_head_size` / `all_head_size`) after this resize is the caller's
job, via the adapter -- see `model_adapter.BertLayerAdapter.set_attention_heads`
and `pruning_workflow.prune_attention_heads`.
"""
import torch
import torch.nn as nn


class AttentionSurgeon:
    def resize(self, query: nn.Linear, key: nn.Linear, value: nn.Linear, output: nn.Linear,
               num_heads: int, head_indices):
        """Return `(query, key, value, output, num_heads)` with the given
        heads removed.

        `query`/`key`/`value` are `(hidden, hidden)` Linear modules whose
        output rows partition into `num_heads` head-sized blocks. `output`
        is the attention block's output projection, whose *input* columns
        partition the same way. `head_indices` are the heads to drop.
        """
        out_features = query.weight.shape[0]
        if out_features % num_heads:
            raise ValueError(
                f"query output size {out_features} is not divisible by num_heads {num_heads}; "
                "num_heads does not match this layer's actual Q/K/V layout."
            )
        if output.weight.shape[1] != out_features:
            raise ValueError(
                f"Attention block is inconsistent: query/key/value produce {out_features} "
                f"features but the output projection expects {output.weight.shape[1]} inputs. "
                "Check the adapter's get_attention_heads() is returning a matching quartet."
            )
        head_dim = out_features // num_heads

        drop = {int(h) for h in head_indices}
        out_of_range = sorted(h for h in drop if h < 0 or h >= num_heads)
        if out_of_range:
            raise ValueError(
                f"head_indices contains out-of-range head indices {out_of_range[:5]} "
                f"for a layer with {num_heads} heads."
            )
        keep_heads = [h for h in range(num_heads) if h not in drop]
        if not keep_heads:
            raise ValueError(
                "Selection would remove every head, which collapses the attention block. "
                "Keep at least one head."
            )

        device = query.weight.device
        row_keep = torch.cat([
            torch.arange(h * head_dim, (h + 1) * head_dim, device=device) for h in keep_heads
        ])

        new_query = self._slice_out_features(query, row_keep)
        new_key = self._slice_out_features(key, row_keep)
        new_value = self._slice_out_features(value, row_keep)
        new_output = self._slice_in_features(output, row_keep)
        return new_query, new_key, new_value, new_output, len(keep_heads)

    @staticmethod
    def _slice_out_features(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
        """New Linear keeping only rows `keep` of weight and bias -- shrinks
        the layer's output dimension (Q/K/V's per-head rows).
        """
        new = nn.Linear(
            linear.in_features, keep.numel(),
            bias=linear.bias is not None, device=linear.weight.device, dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight.data[keep])
            if linear.bias is not None:
                new.bias.copy_(linear.bias.data[keep])
        new.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            new.bias.requires_grad_(linear.bias.requires_grad)
        return new

    @staticmethod
    def _slice_in_features(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
        """New Linear keeping only columns `keep` of weight -- shrinks the
        layer's input dimension (the output projection's per-head columns).
        Its bias is over the model's hidden size, not the head layout, so it
        survives whole.
        """
        new = nn.Linear(
            keep.numel(), linear.out_features,
            bias=linear.bias is not None, device=linear.weight.device, dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight.data[:, keep])
            if linear.bias is not None:
                new.bias.copy_(linear.bias.data)
        new.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            new.bias.requires_grad_(linear.bias.requires_grad)
        return new
