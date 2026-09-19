"""Physically resizes a paired (intermediate, output) FFN block.

This is the only place that knows how to reshape those weights. Every
pruning criterion -- Max-3 saliency, in/out norms, activation-aware
scoring, twin redundancy, or anything added later -- reduces to producing
a `Selection` and handing it to this one surgeon, so selection logic and
surgery mechanics never need to know about each other (Dependency
Inversion) and the mechanics themselves are written exactly once.

Beyond slicing, the surgeon can apply two corrections that make a cut
cost less accuracy than a naive delete:

- **Merging** -- fold a removed neuron's output column into a surviving
  one (`Selection.merge_map`), so a redundant neuron's contribution is
  transferred rather than dropped.
- **Bias compensation** -- add the removed neurons' *expected* output
  contribution back into the output bias. Deleting neuron `i` removes
  `W_out[:, i] * a_i` from the layer output; its expectation
  `W_out[:, i] * E[a_i]` is a constant, and a constant is exactly what a
  bias can represent exactly. So the layer's mean output is preserved for
  free, leaving only the zero-mean residual as real damage. This is
  ordinary practice in structured pruning and costs one calibration pass.
"""
import torch
import torch.nn as nn

from .context import Selection


class FFNSurgeon:
    def resize(self, intermediate: nn.Linear, output: nn.Linear, selection, stats=None,
               compensate_bias: bool = False):
        """Return a new `(intermediate, output)` pair with only the
        selected neurons.

        `selection` may be a `Selection` or a plain list of indices to
        keep. `stats` is a `calibration.CalibrationStats` for this layer,
        required only when `compensate_bias` is true or the selection
        carries merges whose scales were derived from calibration.
        """
        num_neurons = intermediate.weight.shape[0]
        if not isinstance(selection, Selection):
            selection = Selection.of(selection, num_neurons=num_neurons)
        keep_indices = selection.keep_indices
        if keep_indices[-1] >= num_neurons:
            raise ValueError(
                f"Selection keeps neuron {keep_indices[-1]} but the layer has only "
                f"{num_neurons} neurons."
            )
        if output.weight.shape[1] != num_neurons:
            raise ValueError(
                f"FFN pair is inconsistent: intermediate has {num_neurons} output neurons but "
                f"output projection expects {output.weight.shape[1]} inputs. Check the adapter's "
                "get_ffn() is returning a matching (intermediate, output) pair."
            )

        device = intermediate.weight.device
        # Preserve dtype explicitly. Building a bare nn.Linear would
        # default to float32, and copy_ would silently downcast-then-
        # upcast a half-precision model into an fp32 layer that then
        # mismatches every other layer at forward time.
        i_dtype = intermediate.weight.dtype
        o_dtype = output.weight.dtype

        keep = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
        n_keep = keep.numel()

        # Work from the original column values throughout, so that two
        # neurons merging into the same survivor, or a chain of merges,
        # can't make the result depend on dict iteration order.
        source_out_weight = output.weight.data
        new_out_weight_full = source_out_weight.clone()
        new_out_bias = None if output.bias is None else output.bias.data.clone()

        merge_map = selection.merge_map or {}
        if merge_map:
            for dropped, (survivor, scale) in merge_map.items():
                new_out_weight_full[:, survivor] += (
                    source_out_weight[:, dropped].to(torch.float32) * scale
                ).to(o_dtype)

        if compensate_bias:
            if new_out_bias is None:
                raise ValueError(
                    "compensate_bias=True but the output projection has no bias to fold the "
                    "removed neurons' mean contribution into. Pass compensate_bias=False, or "
                    "give the layer a bias."
                )
            if stats is None:
                raise ValueError(
                    "compensate_bias=True requires calibration stats. Collect them with "
                    "`calibration.FFNCalibrator(model).collect(batches, layer_idx)`."
                )
            if stats.num_neurons != num_neurons:
                raise ValueError(
                    f"Calibration stats cover {stats.num_neurons} neurons but the layer has "
                    f"{num_neurons}; they were collected against a different (or already "
                    "pruned) layer."
                )
            new_out_bias += self._bias_compensation(
                source_out_weight, stats, keep_indices, merge_map, num_neurons
            ).to(o_dtype)

        new_intermediate = nn.Linear(
            intermediate.in_features, n_keep,
            bias=intermediate.bias is not None, device=device, dtype=i_dtype,
        )
        new_output = nn.Linear(
            n_keep, output.out_features,
            bias=output.bias is not None, device=device, dtype=o_dtype,
        )
        with torch.no_grad():
            new_intermediate.weight.copy_(intermediate.weight.data[keep])
            if intermediate.bias is not None:
                new_intermediate.bias.copy_(intermediate.bias.data[keep])
            new_output.weight.copy_(new_out_weight_full[:, keep])
            if new_out_bias is not None:
                new_output.bias.copy_(new_out_bias)

        # A frozen layer must stay frozen across surgery, or a "heal"
        # fine-tune would quietly start training weights the experiment
        # meant to hold fixed.
        new_intermediate.weight.requires_grad_(intermediate.weight.requires_grad)
        new_output.weight.requires_grad_(output.weight.requires_grad)
        if intermediate.bias is not None:
            new_intermediate.bias.requires_grad_(intermediate.bias.requires_grad)
        if output.bias is not None:
            new_output.bias.requires_grad_(output.bias.requires_grad)

        return new_intermediate, new_output

    @staticmethod
    def _bias_compensation(source_out_weight, stats, keep_indices, merge_map, num_neurons):
        """Mean output contribution lost by this cut.

        For a plainly-deleted neuron `j` that is `W_out[:, j] * E[a_j]`.
        For one merged into `i` with scale `s`, the merge already
        reintroduces `s * W_out[:, j] * a_i`, so only the *residual*
        `W_out[:, j] * (E[a_j] - s * E[a_i])` is still missing -- which
        is zero exactly when the scale was chosen as `E[a_j] / E[a_i]`.
        Handling both in one expression keeps merge and compensation from
        double-counting each other.
        """
        w = source_out_weight.to(torch.float32)
        mean = stats.mean.to(w.device, torch.float32)
        dropped = torch.ones(num_neurons, dtype=torch.bool, device=w.device)
        dropped[torch.as_tensor(keep_indices, dtype=torch.long, device=w.device)] = False

        residual = mean * dropped  # E[a_j] for dropped neurons, 0 for kept
        for j, (survivor, scale) in merge_map.items():
            residual[j] = mean[j] - scale * mean[survivor]
        return w @ residual
