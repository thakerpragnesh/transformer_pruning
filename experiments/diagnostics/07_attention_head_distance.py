"""Stage 7: same idea as stage 6, but using the thesis's preferred
Manhattan (L1) distance metric side-by-side with Euclidean (L2), plus an
automated bottom-10%-percentile threshold to call out redundant pairs.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModel

from pruning_transformer import AttentionHeadAnalyzer

if __name__ == "__main__":
    print("Loading BERT-base...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    scanner = AttentionHeadAnalyzer(model)

    target_layer = 10
    print(f"Scanning Layer {target_layer}...")
    manhattan_dist = scanner.compute_distance_matrix(target_layer, metric="manhattan")
    euclidean_dist = scanner.compute_distance_matrix(target_layer, metric="euclidean")

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(manhattan_dist, annot=True, fmt=".0f", cmap="viridis_r", ax=ax[0])
    ax[0].set_title("Manhattan Distance (L1)\n(Lower = More Redundant)")
    ax[0].set_xlabel("Head Index")
    ax[0].set_ylabel("Head Index")

    sns.heatmap(euclidean_dist, annot=True, fmt=".1f", cmap="viridis_r", ax=ax[1])
    ax[1].set_title("Euclidean Distance (L2)\n(Lower = More Redundant)")
    ax[1].set_xlabel("Head Index")
    ax[1].set_ylabel("Head Index")

    plt.tight_layout()
    plt.savefig("attention_head_distance.png")

    print("\n--- Diagnosis (Manhattan) ---")
    flat_dist = manhattan_dist[manhattan_dist > 0]
    threshold = np.percentile(flat_dist, 10)
    print(f"Redundancy Threshold (Bottom 10%): {threshold:.2f}")

    rows, cols = np.where((manhattan_dist < threshold) & (manhattan_dist > 0))
    print("Potential Pruning Candidates (Most Similar Pairs):")
    for r, c in zip(rows, cols):
        if r < c:
            print(f"   Head {r} & Head {c} are nearly identical (Dist: {manhattan_dist[r, c]:.0f})")
