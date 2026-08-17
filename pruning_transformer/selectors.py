"""Neuron-selection strategies.

Each selector decides which neuron indices survive pruning, behind one
shared `select_keep_indices` contract. `SaliencySelector` and
`TwinRedundancySelector` are fully substitutable for each other (Liskov)
-- `prune_ffn_layer` never branches on which criterion is in play, and a
new criterion is a new class here, not a new `if` in the workflow
(Open/Closed).
"""
from abc import ABC, abstractmethod

import torch


class NeuronSelector(ABC):
    @abstractmethod
    def select_keep_indices(self, intermediate_weight: torch.Tensor) -> list:
        """Return the indices to KEEP, given the FFN's intermediate.weight."""


class SaliencySelector(NeuronSelector):
    """Keeps the highest-scoring neurons under a saliency criterion,
    dropping `prune_percent`% of the lowest scorers.
    """

    def __init__(self, scorer, prune_percent: float):
        self.scorer = scorer
        self.prune_percent = prune_percent

    def select_keep_indices(self, intermediate_weight: torch.Tensor) -> list:
        scores = self.scorer.score(intermediate_weight)
        current = intermediate_weight.shape[0]
        n_keep = int(current * (1 - self.prune_percent / 100))
        _, keep = torch.topk(scores, k=n_keep)
        return keep.tolist()


class TwinRedundancySelector(NeuronSelector):
    """Drops the higher-indexed neuron of every co-activation "twin"
    pair (see `redundancy.JaccardTwinFinder`) -- ignores raw weights
    entirely, since the redundancy signal here is behavioral.
    """

    def __init__(self, twin_pairs):
        self.drop_indices = {pair[1] for pair in twin_pairs}

    def select_keep_indices(self, intermediate_weight: torch.Tensor) -> list:
        current = intermediate_weight.shape[0]
        return [i for i in range(current) if i not in self.drop_indices]
