"""Stage 10: the co-activation counterpart to stage 5.

Fine-tune on MRPC, find co-activation "twin" neurons, remove one of each
pair, measure the drop, then heal.

The comparison here is drop-vs-merge. All three variants remove exactly
the same neurons, so compression is identical; the only difference is
whether the removed twin's output column is folded into its survivor or
thrown away, and at what scale:

  - dropped                -- discarded (bias compensation still restores the mean)
  - merged, mean-matching  -- folded in at s = E[a_j] / E[a_i]
  - merged, least-squares  -- folded in at s* = E[a_i a_j] / E[a_i^2], the scale
                              minimizing E[(a_j - s a_i)^2]; needs one extra
                              calibration pass for the cross moments

If the twins are genuinely twins, merging should cost noticeably less
accuracy for free -- and if it doesn't, that is itself evidence the
Jaccard threshold is admitting pairs that aren't really interchangeable.
Mean output is preserved exactly under either scale (compensation fixes
the residual), so any gap between the two merged rows is pointwise error.
"""
import sys
from pathlib import Path

# `_mrpc.py` sits one level up, in experiments/, shared by both groups.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mrpc import MrpcHarness, compare

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
    baseline = harness.train_baseline()

    print("\n--- Phase 2: scanning for co-activation twins ---")
    batches = harness.calibration_batches()
    fires = ActivationRecorder(baseline.model, harness.tokenizer).record_batches(batches, layer_idx=LAYER)
    twins, dead = JaccardTwinFinder().find(fires, threshold=THRESHOLD)
    print(f"Recorded {fires.shape[0]} tokens x {fires.shape[1]} neurons.")
    print(f"Found {len(twins)} twin pairs and {len(dead)} dead neurons.")
    if not twins:
        print("No twins at this threshold -- nothing to prune. Try lowering THRESHOLD.")
        return

    calibrator = FFNCalibrator(baseline.model)
    stats = calibrator.collect(batches, layer_idx=LAYER)
    # Cross moments for exactly the (dropped, survivor) folds the merge will
    # make -- chains mean these are not simply the twin pairs.
    merge_pairs = TwinRedundancySelector(twins, merge=True).merge_pairs
    cross_moments = calibrator.collect_cross_moments(batches, LAYER, merge_pairs)

    def cut(merge, cross):
        return lambda model: prune_ffn_layer(
            model, LAYER, TwinRedundancySelector(twins, merge=merge),
            stats=stats, compensate_bias=True, cross_moments=cross,
        )

    compare(harness, baseline, [
        ("Twins dropped", cut(False, None)),
        ("Twins merged (mean-matching)", cut(True, None)),
        ("Twins merged (least-squares)", cut(True, cross_moments)),
    ])


if __name__ == "__main__":
    main()
