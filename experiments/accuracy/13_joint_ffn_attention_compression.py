"""Stage 13: compress the whole model -- attention heads and FFN neurons
together, each with its strongest calibrated criterion.

Stages 05-12 each cut one kind of structure. A compression result has to
answer the combined question: for a given parameter saving, is it cheaper
to take it from the FFNs, from the heads, or from both?

Three variants from one shared trained baseline:

  - FFN only    -- global activation-aware, FFN_PERCENT of all FFN neurons
  - heads only  -- global gradient importance (Michel et al.), HEAD_PERCENT of all heads
  - both        -- the heads cut, then the FFN cut, on the same model

Order matters in "both": heads go first, and the FFN is re-calibrated on
the head-pruned model before its cut. Removing heads changes every
downstream FFN's input distribution, so FFN statistics measured on the
unpruned model would describe activations the pruned FFN never sees --
and bias compensation would then restore the wrong mean. Every cut uses
bias compensation, and every row reports its parameter saving next to
its accuracy, since the variants do not save the same amount.
"""
import sys
from pathlib import Path

# `_mrpc.py` sits one level up, in experiments/, shared by both groups.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mrpc import MrpcHarness, compare

from pruning_transformer import (
    ActivationAwareSelector,
    FFNCalibrator,
    GradientHeadSelector,
    HeadCalibrator,
    prune_model_attention,
    prune_model_ffn,
)

MODEL_NAME = "prajjwal1/bert-small"
FFN_PERCENT = 40
HEAD_PERCENT = 25


def prune_heads(model, harness):
    calibrator = HeadCalibrator(model)
    stats = calibrator.collect_many(harness.calibration_batches())
    importance = calibrator.collect_gradient_importance(harness.calibration_batches(with_labels=True))
    return prune_model_attention(
        model, GradientHeadSelector(HEAD_PERCENT), allocation="global", normalize="l2",
        stats_by_layer=stats, gradient_importance_by_layer=importance, compensate_bias=True,
    )


def prune_ffn(model, harness):
    # Calibrated on the model as it is *now* -- after any head cut.
    stats = FFNCalibrator(model).collect_many(harness.calibration_batches())
    return prune_model_ffn(
        model, ActivationAwareSelector(FFN_PERCENT), allocation="global",
        stats_by_layer=stats, compensate_bias=True,
    )


def main():
    harness = MrpcHarness(model_name=MODEL_NAME, num_epochs=3)
    baseline = harness.train_baseline()

    def steps(*cuts):
        return lambda model: {cut.__name__: cut(model, harness) for cut in cuts}

    compare(harness, baseline, [
        (f"FFN only ({FFN_PERCENT}%)", steps(prune_ffn)),
        (f"heads only ({HEAD_PERCENT}%)", steps(prune_heads)),
        (f"heads {HEAD_PERCENT}% then FFN {FFN_PERCENT}%", steps(prune_heads, prune_ffn)),
    ])


if __name__ == "__main__":
    main()
