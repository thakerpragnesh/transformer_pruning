"""Physically resizes a paired (intermediate, output) BERT FFN block.

This is the only place that knows how to reshape those weights. Every
pruning criterion -- Max-3 saliency, activation-twin redundancy, or
anything added later -- reduces to producing a `keep_indices` list and
handing it to this one surgeon, so selection logic and surgery mechanics
never need to know about each other (Dependency Inversion) and the
mechanics themselves are written exactly once.
"""
import torch
import torch.nn as nn


class FFNSurgeon:
    def resize(self, intermediate: nn.Linear, output: nn.Linear, keep_indices):
        device = intermediate.weight.device
        keep_indices = sorted(keep_indices)
        n_keep = len(keep_indices)

        new_intermediate = nn.Linear(intermediate.in_features, n_keep)
        with torch.no_grad():
            new_intermediate.weight.copy_(intermediate.weight[keep_indices])
            new_intermediate.bias.copy_(intermediate.bias[keep_indices])

        new_output = nn.Linear(n_keep, output.out_features)
        with torch.no_grad():
            new_output.weight.copy_(output.weight[:, keep_indices])
            new_output.bias.copy_(output.bias)

        new_intermediate.to(device)
        new_output.to(device)
        return new_intermediate, new_output
