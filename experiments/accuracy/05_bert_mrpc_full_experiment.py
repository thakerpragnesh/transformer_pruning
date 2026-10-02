"""Stage 5: the full accuracy experiment on MRPC.

Fine-tune BERT-tiny, prune 40% of layer 0's FFN neurons, measure the
accuracy drop, then fine-tune again ("heal") and measure recovery.

Rather than reporting that arc for a single criterion, this runs the same
arc for five, from the same trained baseline and the same seed, so the
numbers are directly comparable:

  1. Max-3 saliency                  -- the thesis criterion, as-is
  2. Max-3 + bias compensation       -- same cut, mean output restored
  3. In/out norm + compensation      -- also weighs the output projection
  4. Activation-aware + compensation -- also weighs how hard neurons fire
  5. Weight-cluster + merge + comp.   -- K-Means on W_in directions, one
                                         survivor per cluster, the rest folded in

The interesting column is *pruned* accuracy, before healing. Healing can
paper over a bad cut given enough epochs; the pre-heal number is what
actually measures how much the criterion knew.
"""
import sys
from pathlib import Path

# `_mrpc.py` sits one level up, in experiments/, shared by both groups.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mrpc import MrpcHarness, compare

from pruning_transformer import (
    ActivationAwareSelector,
    FFNCalibrator,
    InOutNormSelector,
    Max3SaliencyScorer,
    SaliencySelector,
    WeightClusterRedundancySelector,
    prune_ffn_layer,
)

LAYER = 0
PRUNE_PERCENT = 40


def main():
    harness = MrpcHarness(num_epochs=3)
    baseline = harness.train_baseline()

    print("\n--- Phase 2: calibrating layer 0 activations ---")
    stats = FFNCalibrator(baseline.model).collect(harness.calibration_batches(), layer_idx=LAYER)
    print(f"Calibrated on {stats.tokens} tokens across {stats.num_neurons} neurons.")

    def cut(selector, compensate):
        return lambda model: prune_ffn_layer(
            model, LAYER, selector, stats=stats, compensate_bias=compensate
        )

    compare(harness, baseline, [
        ("Max-3 saliency", cut(SaliencySelector(Max3SaliencyScorer(), PRUNE_PERCENT), False)),
        ("Max-3 + bias comp.", cut(SaliencySelector(Max3SaliencyScorer(), PRUNE_PERCENT), True)),
        ("In/out norm + comp.", cut(InOutNormSelector(PRUNE_PERCENT), True)),
        ("Activation-aware + comp.", cut(ActivationAwareSelector(PRUNE_PERCENT), True)),
        ("Weight-cluster + merge + comp.",
         cut(WeightClusterRedundancySelector(PRUNE_PERCENT, merge=True), True)),
    ])
    print(f"\nCompression: layer {LAYER} FFN reduced by {PRUNE_PERCENT}%.")


if __name__ == "__main__":
    main()
