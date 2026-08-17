"""Stage 2: run NetworkSaliencyScanner across every conv layer of VGG16
to get a full "health report", then drill into one deep layer's weakest
filters.
"""
import torchvision.models as models

from pruning_transformer import discover_layers, Max3SaliencyScorer, NetworkSaliencyScanner

if __name__ == "__main__":
    model = models.vgg16(pretrained=True)
    layers = discover_layers(model, include_conv=True, include_linear=False)
    scanner = NetworkSaliencyScanner(layers, Max3SaliencyScorer())

    report = scanner.scan()
    print("\n--- Network Saliency Diagnosis ---")
    print(report.to_string())

    target_layer = "features.24"
    print(f"\n--- Detailed Analysis for {target_layer} ---")
    weakest = scanner.weakest_units(target_layer, n=5)
    for rank, (idx, score) in enumerate(weakest):
        print(f"Rank {rank + 1}: Filter {idx} | Score: {score:.4f}")
