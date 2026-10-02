"""Stage 6: cosine-similarity heatmaps between BERT-base attention heads
in one layer, to look for near-duplicate heads that could be pruned.

Two panels: the original query-weight cosine (where heads *look*) next to
the OV-circuit cosine (what heads *write*, `W_O[:, h] @ W_V[h]`). Two heads
can share a query pattern and still write different things; only the
second panel says they are interchangeable. Stage 12 measures what cutting
by each signal actually costs.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModel

from pruning_transformer import AttentionHeadAnalyzer

THRESHOLD = 0.90


def report_pairs(name, sim):
    print(f"\n--- Diagnosis ({name}) ---")
    rows, cols = np.where(sim > THRESHOLD)
    found = False
    for r, c in zip(rows, cols):
        if r < c:
            print(f"   Head {r} and Head {c} are {sim[r, c]:.2%} similar")
            found = True
    if not found:
        print(f"No near-duplicate heads found in this layer at the {THRESHOLD:.2f} threshold.")


if __name__ == "__main__":
    print("Loading BERT-base...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    scanner = AttentionHeadAnalyzer(model)

    target_layer = 10
    print(f"Scanning Redundancy in Layer {target_layer}...")
    query_sim = scanner.compute_similarity(target_layer)
    ov_sim = scanner.compute_ov_similarity(target_layer)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for axis, sim, title in (
        (ax[0], query_sim, "Query-weight cosine (where heads look)"),
        (ax[1], ov_sim, "OV-circuit cosine (what heads write)"),
    ):
        sns.heatmap(sim, annot=True, fmt=".2f", cmap="coolwarm", ax=axis)
        axis.set_title(f"{title}\nLayer {target_layer}")
        axis.set_xlabel("Head Index")
        axis.set_ylabel("Head Index")
    plt.tight_layout()
    plt.savefig("attention_head_similarity.png")

    report_pairs("query weights", query_sim)
    report_pairs("OV circuits", ov_sim)
