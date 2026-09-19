"""Stage 10: the co-activation counterpart to stage 5.

Fine-tune on MRPC, find co-activation "twin" neurons, remove one of each
pair, measure the drop, then heal.

The comparison here is drop-vs-merge. Both remove exactly the same
neurons, so compression is identical; the only difference is whether the
removed twin's output column is folded into its survivor or thrown away.
If the twins are genuinely twins, merging should cost noticeably less
accuracy for free -- and if it doesn't, that is itself evidence the
Jaccard threshold is admitting pairs that aren't really interchangeable.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mrpc import MrpcHarness, accuracy, report

from pruning_transformer import (
    ActivationRecorder,
    FFNCalibrator,
    JaccardTwinFinder,
    TwinRedundancySelector,
    prune_ffn_layer,
)

LAYER = 0
THRESHOLD = 0.95


def main():
    harness = MrpcHarness(num_epochs=5)

    print("--- Phase 1: training the shared baseline ---")
    model = harness.new_model()
    trainer = harness.new_trainer(model)
    trainer.train()
    baseline = trainer.evaluate()
    baseline_state = copy.deepcopy(model.state_dict())
    print(f"Pre-surgery accuracy: {accuracy(baseline):.4f}")

    print("\n--- Phase 2: scanning for co-activation twins ---")
    batches = harness.calibration_batches()
    fires = ActivationRecorder(model.bert, harness.tokenizer).record_batches(batches, layer_idx=LAYER)
    twins, dead = JaccardTwinFinder().find(fires, threshold=THRESHOLD)
    print(f"Recorded {fires.shape[0]} tokens x {fires.shape[1]} neurons.")
    print(f"Found {len(twins)} twin pairs and {len(dead)} dead neurons.")
    if not twins:
        print("No twins at this threshold -- nothing to prune. Try lowering THRESHOLD.")
        return

    stats = FFNCalibrator(model.bert).collect(batches, layer_idx=LAYER)

    rows = [("Baseline (unpruned)", f"{accuracy(baseline):.2%}")]
    for label, merge in [("Twins dropped", False), ("Twins merged", True)]:
        print(f"\n--- {label} ---")
        variant_model = harness.new_model()
        variant_model.load_state_dict(baseline_state)

        kept = prune_ffn_layer(
            variant_model.bert, LAYER,
            TwinRedundancySelector(twins, merge=merge),
            stats=stats, compensate_bias=True,
        )
        print(f"Kept {kept} neurons.")

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


if __name__ == "__main__":
    main()
