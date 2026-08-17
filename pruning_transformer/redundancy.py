"""Pure analysis over a recorded activation-firing tensor.

Finds neuron pairs whose firing pattern overlaps almost perfectly
(true Jaccard/IoU: intersection over union) -- functionally redundant
regardless of their raw weight saliency. Not from the thesis; a new
criterion explored alongside the weight-based ones.
"""
import numpy as np
import torch


class JaccardTwinFinder:
    def find(self, fires: torch.Tensor, threshold: float = 0.95):
        intersection = torch.mm(fires.t(), fires)
        fires_per_neuron = fires.sum(dim=0)
        union = fires_per_neuron.unsqueeze(0) + fires_per_neuron.unsqueeze(1) - intersection
        union = torch.where(union == 0, torch.ones_like(union), union)

        jaccard = (intersection / union).numpy()
        rows, cols = np.where(jaccard >= threshold)

        twins, dead = [], []
        for r, c in zip(rows, cols):
            if r < c:
                if fires_per_neuron[r] == 0 and fires_per_neuron[c] == 0:
                    dead.append(int(r))
                else:
                    twins.append((int(r), int(c), float(jaccard[r, c])))
        return twins, list(set(dead))
