"""Stage 8: repeat the distance analysis on L2-unit-normalized head
vectors (removing magnitude from the comparison) to isolate redundancy
that comes from direction/distribution rather than raw scale.
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
    print(f"Scanning Layer {target_layer} for Distributional Divergence...")
    manhattan_dist = scanner.compute_distance_matrix(target_layer, metric="manhattan", normalize=True)
    euclidean_dist = scanner.compute_distance_matrix(target_layer, metric="euclidean", normalize=True)

    fig, ax = plt.subplots(1, 2, figsize=(22, 9))
    sns.heatmap(manhattan_dist, annot=True, fmt=".1f", cmap="viridis_r", ax=ax[0])
    ax[0].set_title("Scaled Manhattan Distance (L1)")
    sns.heatmap(euclidean_dist, annot=True, fmt=".2f", cmap="viridis_r", ax=ax[1])
    ax[1].set_title("Scaled Euclidean Distance (L2)")
    plt.tight_layout()
    plt.savefig("attention_scaled_distribution.png")

    print("\n--- Redundancy Report (Manhattan) ---")
    flat_dist = manhattan_dist[manhattan_dist > 0]
    threshold = np.percentile(flat_dist, 10)
    print(f"Deep Redundancy Threshold: {threshold:.2f}")

    rows, cols = np.where((manhattan_dist < threshold) & (manhattan_dist > 0))
    for r, c in zip(rows, cols):
        if r < c:
            print(f"   Pair Found: Head {r} & Head {c} are distributionally identical.")
