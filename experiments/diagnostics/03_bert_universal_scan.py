"""Stage 3: first crossover onto transformers -- run the Max-3 saliency
criterion on BERT-tiny's FFN (intermediate.dense) layers via
NetworkSaliencyScanner (the same scanner used on VGG16 in stage 2, just
handed a different layer set), then flag the layer with the highest
score variance as the best pruning candidate.
"""
from transformers import AutoModel

from pruning_transformer import discover_layers, Max3SaliencyScorer, NetworkSaliencyScanner

if __name__ == "__main__":
    print("Loading BERT-tiny...")
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    layers = discover_layers(
        bert_model, include_conv=True, include_linear=True, linear_name_filter="intermediate"
    )
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer())
    report = scanner.scan()
    print(report.to_string())

    best_layer = report.loc[report["StdDev"].idxmax()]
    print(f"\nRecommendation: start pruning layer '{best_layer['Layer']}'")
    print(f"Reason: highest score variance ({best_layer['StdDev']:.4f})")
