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

from .model_adapter import BertLayerAdapter

# Lp-norm order per named distance metric. A new metric is a new entry
# here, not a new branch in compute_distance_matrix (Open/Closed).
LP_METRICS = {
    "manhattan": 1,
    "euclidean": 2,
}


class AttentionHeadAnalyzer:
    def __init__(self, model=None, adapter=None):
        if adapter is None:
            if model is None:
                raise ValueError("AttentionHeadAnalyzer requires either `model` or `adapter`.")
            adapter = BertLayerAdapter(model)
        self.adapter = adapter

    def get_flat_heads(self, layer_idx: int) -> torch.Tensor:
        num_heads = self.adapter.num_attention_heads
        hidden = self.adapter.hidden_size
        if hidden % num_heads:
            raise ValueError(
                f"hidden_size {hidden} is not divisible by num_attention_heads {num_heads}; "
                "the adapter is reporting a head layout this model does not have."
            )
        head_dim = hidden // num_heads
        w_q = self.adapter.get_query_weight(layer_idx).detach()
        # W_q is (hidden_out, hidden_in) and the head partition runs down
        # the output rows, so reshaping dim 0 into (heads, head_dim) is
        # what groups each head's rows together.
        return w_q.reshape(num_heads, head_dim * w_q.shape[1]).float()

    def compute_similarity(self, layer_idx: int):
        heads = torch.nn.functional.normalize(self.get_flat_heads(layer_idx), p=2, dim=1)
        return torch.mm(heads, heads.t()).cpu().numpy()

    def compute_distance_matrix(self, layer_idx: int, metric: str = "manhattan", normalize: bool = False):
        if metric not in LP_METRICS:
            raise ValueError(f"Unknown metric '{metric}'; expected one of {sorted(LP_METRICS)}")
        heads = self.get_flat_heads(layer_idx)
        if normalize:
            heads = torch.nn.functional.normalize(heads, p=2, dim=1)
        dist = torch.cdist(heads.unsqueeze(0), heads.unsqueeze(0), p=LP_METRICS[metric]).squeeze(0)
        dist.fill_diagonal_(0.0)
        # .cpu() before .numpy(): on a CUDA model this raised
        # "can't convert cuda tensor to numpy" -- compute_similarity
        # already did this, compute_distance_matrix did not.
        return dist.cpu().numpy()
