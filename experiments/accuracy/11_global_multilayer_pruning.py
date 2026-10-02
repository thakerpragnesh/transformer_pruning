"""Stage 11: prune the whole FFN stack instead of one layer.

Stages 05 and 10 both cut a single layer, which keeps the comparison
clean but sidesteps the question a real compression result has to answer:
given a model-wide budget, *where* should the cuts go?

This contrasts the two answers at identical total compression:

  - uniform -- every layer loses the same fraction
  - global  -- every neuron in the model competes against every other,
               so layers that turn out to be redundant give up more

FFN redundancy is generally not spread evenly across depth, so uniform
over-cuts the layers carrying the model and under-cuts the ones that
aren't. The per-layer survivor counts printed below show how unevenly
the global ranking actually chooses to spend the budget.
"""
import sys
from pathlib import Path

# `_mrpc.py` sits one level up, in experiments/, shared by both groups.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mrpc import MrpcHarness, compare

from pruning_transformer import ActivationAwareSelector, FFNCalibrator, prune_model_ffn

PRUNE_PERCENT = 40


def main():
    harness = MrpcHarness(num_epochs=3)
    baseline = harness.train_baseline()

    print("\n--- Phase 2: calibrating every layer ---")
    # One pass over the corpus for every layer, not one pass per layer.
    stats_by_layer = FFNCalibrator(baseline.model).collect_many(harness.calibration_batches())
    print(f"Calibrated {len(stats_by_layer)} layers on {stats_by_layer[0].tokens} tokens.")

    def cut(allocation):
        return lambda model: prune_model_ffn(
            model, ActivationAwareSelector(PRUNE_PERCENT), allocation=allocation,
            stats_by_layer=stats_by_layer, compensate_bias=True,
        )

    compare(harness, baseline, [
        (f"{allocation} @ {PRUNE_PERCENT}%", cut(allocation)) for allocation in ("uniform", "global")
    ])


if __name__ == "__main__":
    main()
