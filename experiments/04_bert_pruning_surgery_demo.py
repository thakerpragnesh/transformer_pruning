"""Stage 4: verify the surgery itself works -- prune 30% of BERT-tiny's
layer 0 FFN neurons and confirm the model still runs a forward pass
without shape errors.
"""
import torch
from transformers import AutoModel

from pruning_transformer import Max3SaliencyScorer, SaliencySelector, prune_ffn_layer

if __name__ == "__main__":
    print("Loading fresh BERT-tiny...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    print(f"Original Params: {sum(p.numel() for p in model.parameters())}")

    selector = SaliencySelector(Max3SaliencyScorer(), prune_percent=30)
    prune_ffn_layer(model, layer_index=0, selector=selector)

    print(f"\nPruned Params: {sum(p.numel() for p in model.parameters())}")
    print("\nStructure Check:")
    print(model.encoder.layer[0].intermediate.dense)

    print("\nRunning Test Inference...")
    dummy_input = torch.tensor([[101, 2054, 2003, 102]])
    try:
        output = model(dummy_input)
        print("SUCCESS: The pruned model runs without crashing!")
        print(f"Output Shape: {output.last_hidden_state.shape}")
    except Exception as e:
        print(f"CRASH: {e}")
