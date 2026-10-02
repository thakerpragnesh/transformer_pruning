"""Stage 12: the attention-head counterpart to stage 05.

Fine-tune once, remove attention heads by each criterion from that shared
baseline, measure the drop, then heal. Stages 06-08 only ever *diagnosed*
head redundancy from query weights; this is the first stage that cuts
heads and measures what each criterion's cut costs.

BERT-small (4 layers x 8 heads), not BERT-tiny: tiny has 2 heads per
layer, so every criterion at any budget collapses to the same handful of
choices and the comparison says nothing.

Every variant removes the same number of heads (PRUNE_PERCENT of each
layer, or of the model for the global rows):

  1. Query-cosine redundancy  -- the stage-06 signal, as the baseline
  2. OV-circuit redundancy    -- compares what heads write, not where they look
  3. OV norm                  -- how much a head *can* write
  4. Activation-aware (rms)   -- how much it *does* write on MRPC
  5. Activation-aware (std) + bias compensation
                              -- what is left after mean-ablation, and the cut made that way
  6. Gradient importance      -- Michel et al. (2019): how much the loss cares
  7. Gradient, global + comp. -- same signal, budget spent model-wide

As in stage 05, read the *pruned* column first: it measures what the
criterion knew; healing can paper over a bad cut.
"""
import sys
from pathlib import Path

# `_mrpc.py` sits one level up, in experiments/, shared by both groups.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mrpc import MrpcHarness, compare

from pruning_transformer import (
    ActivationAwareHeadSelector,
    GradientHeadSelector,
    HeadCalibrator,
    OVNormHeadSelector,
    RedundantHeadSelector,
    prune_model_attention,
)

MODEL_NAME = "prajjwal1/bert-small"
PRUNE_PERCENT = 25


def main():
    harness = MrpcHarness(model_name=MODEL_NAME, num_epochs=3)
    baseline = harness.train_baseline()

    print("\n--- Phase 2: measuring every head ---")
    # Both signals are measured once, on the trained baseline every variant
    # starts from, so no variant is judged against a different model. The
    # task model is passed as-is: the calibrator hooks its encoder's layers
    # while the batches -- labels included -- run through the classifier, so
    # gradient importance sees the real task loss.
    calibrator = HeadCalibrator(baseline.model)
    head_stats = calibrator.collect_many(harness.calibration_batches())
    gradient_importance = calibrator.collect_gradient_importance(
        harness.calibration_batches(with_labels=True)
    )
    for layer, scores in gradient_importance.items():
        print(f"  layer {layer} gradient importance: "
              + " ".join(f"{s:.3g}" for s in scores.tolist()))

    def cut(selector, allocation="uniform", compensate=False):
        return lambda model: prune_model_attention(
            model, selector, allocation=allocation, normalize="l2",
            stats_by_layer=head_stats, gradient_importance_by_layer=gradient_importance,
            compensate_bias=compensate,
        )

    compare(harness, baseline, [
        ("Query-cosine redundancy", cut(RedundantHeadSelector(PRUNE_PERCENT, similarity="query"))),
        ("OV-circuit redundancy", cut(RedundantHeadSelector(PRUNE_PERCENT, similarity="ov"))),
        ("OV norm", cut(OVNormHeadSelector(PRUNE_PERCENT))),
        ("Activation-aware (rms)", cut(ActivationAwareHeadSelector(PRUNE_PERCENT))),
        ("Activation-aware (std) + comp.",
         cut(ActivationAwareHeadSelector(PRUNE_PERCENT, statistic="std"), compensate=True)),
        ("Gradient importance", cut(GradientHeadSelector(PRUNE_PERCENT))),
        ("Gradient, global + comp.",
         cut(GradientHeadSelector(PRUNE_PERCENT), allocation="global", compensate=True)),
    ])


if __name__ == "__main__":
    main()
