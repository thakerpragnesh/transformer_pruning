"""Stage 1: sanity-check the vectorized Max-3 saliency score on a single
VGG16 conv layer, using CIFAR-10 only to build a realistic dataloader
(no training happens here).
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from pruning_transformer import Max3SaliencyScorer, lowest_scoring

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_environment():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    model = models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, 10)
    model.to(device)
    return model, dataloader


if __name__ == "__main__":
    print(f"Running on: {device}")
    model, dataloader = get_environment()

    target_layer_index = 0
    layer = model.features[target_layer_index]
    print(f"\nAnalyzing Layer {target_layer_index}: {layer}")
    print(f"Original Weight Shape: {layer.weight.shape}")

    prune_amount = 5
    candidates = lowest_scoring(Max3SaliencyScorer(), layer.weight.data, prune_amount)

    print(f"\n--- Top {prune_amount} Candidates for Pruning (Max-3 Saliency) ---")
    for rank, (filter_idx, score) in enumerate(candidates):
        print(f"Rank {rank + 1}: Filter Index {filter_idx} | Saliency Score: {score:.4f}")
