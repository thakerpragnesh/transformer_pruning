"""Saliency scoring strategies.

Each strategy scores a weight tensor's output units (conv filters or
linear neurons) behind one shared interface. Any consumer (a network
scanner, a neuron selector) accepts a `SaliencyScorer` and never needs to
know which concrete scoring rule it got (Liskov substitution) or be
edited when a new one is added (Open/Closed) -- e.g. an SVD or K-means
score could be added here later as another `SaliencyScorer` without
touching `NetworkSaliencyScanner` or `SaliencySelector`.
"""
from abc import ABC, abstractmethod

import torch


class SaliencyScorer(ABC):
    @abstractmethod
    def score(self, weight: torch.Tensor) -> torch.Tensor:
        """Return one score per output unit (dim 0 of `weight`)."""


class Max3SaliencyScorer(SaliencyScorer):
    """Per output unit, sum its 3 largest-magnitude incoming weights.

    For a conv filter (4D weight) that's summed again across input
    channels and kernel positions; for a linear neuron (2D weight) it's
    the 3 strongest incoming connections directly.
    """

    def score(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dim() == 4:
            out_ch, in_ch, kh, kw = weight.shape
            flat = weight.abs().view(out_ch, in_ch, kh * kw)
            k = min(3, flat.shape[2])
            return torch.topk(flat, k=k, dim=2).values.sum(dim=(1, 2))
        if weight.dim() == 2:
            k = min(3, weight.shape[1])
            return torch.topk(weight.abs(), k=k, dim=1).values.sum(dim=1)
        raise ValueError(f"Max3SaliencyScorer supports 2D/4D weights, got {weight.dim()}D")


def lowest_scoring(scorer: SaliencyScorer, weight: torch.Tensor, amount: int):
    """Convenience for one-off lookups: the `amount` lowest-scoring units
    for a single weight tensor, as [index, score] pairs.
    """
    scores = scorer.score(weight)
    amount = min(amount, scores.shape[0])
    values, indices = torch.topk(scores, k=amount, largest=False)
    return [[idx.item(), val.item()] for idx, val in zip(indices, values)]
