"""Reports per-layer saliency statistics and lowest-scoring units.

Takes its layer list and its scoring rule as dependencies (Dependency
Inversion) instead of hardcoding either, so the same class covers both a
pure-CNN pass and a mixed Conv2d+Linear pass -- what used to be two
near-duplicate classes (`SaliencyPruner`, `UniversalPruner`) differing
only in which layer types and scorer they hardcoded.
"""
import pandas as pd
import torch


class NetworkSaliencyScanner:
    def __init__(self, layers, scorer):
        self.layers = layers
        self.scorer = scorer

    def scan(self):
        rows = []
        for layer in self.layers:
            scores = self.scorer.score(layer.module.weight.data)
            rows.append(
                {
                    "Layer": layer.name,
                    "Type": layer.kind,
                    "Units": layer.module.weight.shape[0],
                    "Avg Score": scores.mean().item(),
                    "Min Score": scores.min().item(),
                    "StdDev": scores.std().item(),
                }
            )
        return pd.DataFrame(rows)

    def weakest_units(self, layer_name, n=5):
        layer = next((l for l in self.layers if l.name == layer_name), None)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found among scanned layers.")
        scores = self.scorer.score(layer.module.weight.data)
        values, indices = torch.topk(scores, k=n, largest=False)
        return list(zip(indices.tolist(), values.tolist()))
