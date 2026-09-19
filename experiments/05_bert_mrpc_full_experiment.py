"""Stage 5: the full accuracy experiment on MRPC.

Fine-tune BERT-tiny, prune 40% of layer 0's FFN neurons, measure the
accuracy drop, then fine-tune again ("heal") and measure recovery.

Rather than reporting that arc for a single criterion, this runs the same
arc for four, from the same trained baseline and the same seed, so the
numbers are directly comparable:

  1. Max-3 saliency                  -- the thesis criterion, as-is
  2. Max-3 + bias compensation       -- same cut, mean output restored
  3. In/out norm + compensation      -- also weighs the output projection
  4. Activation-aware + compensation -- also weighs how hard neurons fire

The interesting column is *pruned* accuracy, before healing. Healing can
paper over a bad cut given enough epochs; the pre-heal number is what
actually measures how much the criterion knew.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mrpc import MrpcHarness, accuracy, report

from pruning_transformer import (
    ActivationAwareSelector,
    FFNCalibrator,
    InOutNormSelector,
    Max3SaliencyScorer,
    SaliencySelector,
    prune_ffn_layer,
)

LAYER = 0
PRUNE_PERCENT = 40


def main():
    harness = MrpcHarness(num_epochs=3)

    print("--- Phase 1: training the shared baseline ---")
    model = harness.new_model()
    trainer = harness.new_trainer(model)
    trainer.train()
    baseline = trainer.evaluate()
    baseline_state = copy.deepcopy(model.state_dict())
    print(f"Baseline accuracy: {accuracy(baseline):.4f}")

    print("\n--- Phase 2: calibrating layer 0 activations ---")
    stats = FFNCalibrator(model.bert).collect(harness.calibration_batches(), layer_idx=LAYER)
    print(f"Calibrated on {stats.tokens} tokens across {stats.num_neurons} neurons.")

    variants = [
        ("Max-3 saliency", SaliencySelector(Max3SaliencyScorer(), PRUNE_PERCENT), False),
        ("Max-3 + bias comp.", SaliencySelector(Max3SaliencyScorer(), PRUNE_PERCENT), True),
        ("In/out norm + comp.", InOutNormSelector(PRUNE_PERCENT), True),
        ("Activation-aware + comp.", ActivationAwareSelector(PRUNE_PERCENT), True),
    ]

    rows = [("Baseline (unpruned)", f"{accuracy(baseline):.2%}")]
    for label, selector, compensate in variants:
        print(f"\n--- {label} ---")
        # Every variant starts from the identical trained baseline, so the
        # only difference between runs is the criterion under test.
        variant_model = harness.new_model()
        variant_model.load_state_dict(baseline_state)

        kept = prune_ffn_layer(
            variant_model.bert, LAYER, selector, stats=stats, compensate_bias=compensate
        )
        print(f"Kept {kept} neurons.")

        # A fresh Trainer: the old one's optimizer still holds the
        # pre-surgery parameters. See MrpcHarness.new_trainer.
        variant_trainer = harness.new_trainer(variant_model)
        pruned = variant_trainer.evaluate()
        variant_trainer.train()
        healed = variant_trainer.evaluate()

        rows.append((
            label,
            f"pruned {accuracy(pruned):.2%}  ->  healed {accuracy(healed):.2%}"
            f"   (drop {accuracy(baseline) - accuracy(pruned):+.2%})",
        ))

    report(rows)
    print(f"\nCompression: layer {LAYER} FFN reduced by {PRUNE_PERCENT}%.")


if __name__ == "__main__":
    main()
