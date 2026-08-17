"""Stage 6: cosine-similarity heatmap between BERT-base attention heads
in one layer, to look for near-duplicate heads that could be merged/
pruned.
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
    print(f"Scanning Redundancy in Layer {target_layer}...")
    sim_matrix = scanner.compute_similarity(target_layer)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Cosine Similarity between Attention Heads (Layer {target_layer})")
    plt.xlabel("Head Index")
    plt.ylabel("Head Index")
    plt.tight_layout()
    plt.savefig("attention_head_similarity.png")

    print("\n--- Diagnosis ---")
    rows, cols = np.where(sim_matrix > 0.90)
    found = False
    for r, c in zip(rows, cols):
        if r < c:
            print(f"   Head {r} and Head {c} are {sim_matrix[r, c]:.2%} similar")
            found = True
    if not found:
        print("No near-duplicate heads found in this layer at the 0.90 threshold.")
