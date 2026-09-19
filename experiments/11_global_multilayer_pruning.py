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
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mrpc import MrpcHarness, accuracy, report

from pruning_transformer import ActivationAwareSelector, FFNCalibrator, prune_model_ffn

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

    print("\n--- Phase 2: calibrating every layer ---")
    batches = harness.calibration_batches()
    calibrator = FFNCalibrator(model.bert)
    num_layers = len(model.bert.encoder.layer)
    stats_by_layer = {i: calibrator.collect(batches, layer_idx=i) for i in range(num_layers)}
    print(f"Calibrated {num_layers} layers on {stats_by_layer[0].tokens} tokens.")

    rows = [("Baseline (unpruned)", f"{accuracy(baseline):.2%}")]
    for allocation in ("uniform", "global"):
        print(f"\n--- {allocation} allocation ---")
        variant_model = harness.new_model()
        variant_model.load_state_dict(baseline_state)

        kept = prune_model_ffn(
            variant_model.bert,
            ActivationAwareSelector(PRUNE_PERCENT),
            allocation=allocation,
            stats_by_layer=stats_by_layer,
            compensate_bias=True,
        )
        print(f"Neurons kept per layer: {kept}  (total {sum(kept.values())})")

        variant_trainer = harness.new_trainer(variant_model)
        pruned = variant_trainer.evaluate()
        variant_trainer.train()
        healed = variant_trainer.evaluate()
        rows.append((
            f"{allocation} @ {PRUNE_PERCENT}%",
            f"pruned {accuracy(pruned):.2%}  ->  healed {accuracy(healed):.2%}"
            f"   kept {sum(kept.values())} neurons",
        ))

    report(rows)


if __name__ == "__main__":
    main()
