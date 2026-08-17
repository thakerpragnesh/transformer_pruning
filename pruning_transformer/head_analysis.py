"""Attention-head redundancy analysis.

Computes cosine similarity or Lp-distance between heads' flattened query
weight vectors, optionally on unit-normalized vectors to isolate
directional (as opposed to magnitude) redundancy. One class with a
`normalize` switch replaces what used to be two nearly-identical classes
(`SimilarityScanner`, `DistributionScanner`) differing only in whether
they normalized first -- the duplicated `cdist`/reshape code is now
written once.
"""
import torch


class AttentionHeadAnalyzer:
    def __init__(self, model):
        self.model = model

    def get_flat_heads(self, layer_idx: int) -> torch.Tensor:
        layer = self.model.encoder.layer[layer_idx]
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        w_q = layer.attention.self.query.weight.data
        return w_q.view(num_heads, head_dim, -1).reshape(num_heads, -1)

    def compute_similarity(self, layer_idx: int):
        heads = torch.nn.functional.normalize(self.get_flat_heads(layer_idx), p=2, dim=1)
        return torch.mm(heads, heads.t()).cpu().numpy()

    def compute_distance_matrix(self, layer_idx: int, metric: str = "manhattan", normalize: bool = False):
        heads = self.get_flat_heads(layer_idx)
        if normalize:
            heads = torch.nn.functional.normalize(heads, p=2, dim=1)
        p = 1 if metric == "manhattan" else 2
        dist = torch.cdist(heads.unsqueeze(0), heads.unsqueeze(0), p=p).squeeze(0)
        dist.fill_diagonal_(0.0)
        return dist.numpy()
