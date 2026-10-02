"""Shared MRPC setup for the full accuracy experiments (stages 05, 10-13).

Stages 05 and 10-13 differ only in what they prune, but each used to
carry its own copy of the dataset/trainer boilerplate, and then of the
train-baseline / clone / prune / evaluate / heal / evaluate loop. Two copies
of a training setup is two chances for them to drift, and a baseline that
drifts makes the comparison between criteria meaningless -- which is the
entire point of running them side by side.

This module sits at the top of `experiments/`, above `diagnostics/` and
`accuracy/`, because both groups import it: every `accuracy/` script uses
the harness, and the diagnostic stage 09 uses `set_seeds`.

So an experiment script is now only its variants: each is a label and a
`prune(model)` function, and `MrpcHarness.train_baseline` /
`run_variant` own everything else (Single Responsibility). The piece of
behaviour worth reading carefully is `run_variant`, which is the one
place a fresh Trainer is built after surgery -- see `new_trainer`.
"""
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG the experiments touch.

    Without this, "pruning cost us 1.2% accuracy" is indistinguishable
    from ordinary run-to-run variance on a dataset as small as MRPC --
    and a thesis number nobody can reproduce is not a result.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class Baseline:
    """The trained, unpruned model every variant starts from."""

    model: object
    metrics: dict
    state: dict
    params: int


@dataclass
class VariantResult:
    """One variant's arc: accuracy straight after the cut, then after healing."""

    model: object
    pruned: dict
    healed: dict
    params: int


class MrpcHarness:
    """Dataset, metric, Trainer factory, and the baseline -> variant
    protocol for the MRPC experiments."""

    def __init__(self, model_name="prajjwal1/bert-tiny", num_epochs=3, batch_size=32,
                 learning_rate=2e-5, output_dir="./results", seed=SEED):
        import evaluate
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments

        set_seeds(seed)
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = load_dataset("glue", "mrpc")
        self.metric = evaluate.load("glue", "mrpc")

        def preprocess(examples):
            return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

        self.tokenized = self.dataset.map(preprocess, batched=True)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            seed=seed,
            report_to=[],
        )

    def new_model(self):
        from transformers import BertForSequenceClassification

        return BertForSequenceClassification.from_pretrained(self.model_name)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        return self.metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    def new_trainer(self, model):
        """Build a *fresh* Trainer around `model`.

        This must be called again after any surgery. A Trainer caches the
        optimizer (and the accelerate-prepared model) after its first
        `train()`, and those hold references to the exact `nn.Parameter`
        objects that existed then. Pruning replaces the FFN Linears with
        new modules, so reusing the old Trainer to "heal" would step an
        optimizer over the *detached* pre-surgery parameters -- the
        pruned model would sit untrained while the reported accuracy
        drifted for reasons having nothing to do with the criterion under
        test.
        """
        from transformers import Trainer

        return Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["validation"],
            data_collator=self.collator,
            compute_metrics=self.compute_metrics,
        )

    def train_baseline(self) -> Baseline:
        """Fine-tune one model to serve as every variant's starting point."""
        print("--- Phase 1: training the shared baseline ---")
        model = self.new_model()
        trainer = self.new_trainer(model)
        trainer.train()
        metrics = trainer.evaluate()
        baseline = Baseline(model, metrics, copy.deepcopy(model.state_dict()), count_params(model))
        print(f"Baseline accuracy: {accuracy(metrics):.4f}  ({baseline.params:,} params)")
        return baseline

    def run_variant(self, baseline: Baseline, label: str, prune) -> VariantResult:
        """Clone the baseline, apply `prune(model)`, then measure the cut and
        the heal.

        Every variant starts from the identical trained weights, so the only
        difference between rows is what `prune` did. Whatever `prune`
        returns (typically the per-layer kept counts) is printed. The fresh
        Trainer is built *here*, after surgery, never by the caller -- see
        `new_trainer` for why reusing one silently measures nothing.
        """
        print(f"\n--- {label} ---")
        model = self.new_model()
        model.load_state_dict(baseline.state)
        outcome = prune(model)
        if outcome is not None:
            print(f"Kept: {outcome}")
        trainer = self.new_trainer(model)
        pruned = trainer.evaluate()
        trainer.train()
        healed = trainer.evaluate()
        return VariantResult(model, pruned, healed, count_params(model))

    def calibration_batches(self, split="train", limit=512, batch_size=32, with_labels=False):
        """Collated batches for `FFNCalibrator` / `HeadCalibrator` /
        `ActivationRecorder`.

        Reuses the same tokenization and collator as evaluation, so the
        activation statistics describe the distribution the model is
        actually scored on. `with_labels=True` keeps the label (the
        collator renames it `labels`), which
        `HeadCalibrator.collect_gradient_importance` needs for the loss;
        every other calibrator ignores it.
        """
        from torch.utils.data import DataLoader

        keep = ("input_ids", "attention_mask", "token_type_ids") + (("label",) if with_labels else ())
        ds = self.tokenized[split].select(range(min(limit, len(self.tokenized[split]))))
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return list(DataLoader(ds, batch_size=batch_size, collate_fn=self.collator))


def describe(baseline: Baseline, result: VariantResult) -> str:
    """The standard results-table cell: pre-heal accuracy first, because it
    is what measures how much the criterion knew."""
    drop = accuracy(baseline.metrics) - accuracy(result.pruned)
    saved = 1 - result.params / baseline.params
    return (f"pruned {accuracy(result.pruned):.2%}  ->  healed {accuracy(result.healed):.2%}"
            f"   (drop {drop:+.2%}, params -{saved:.1%})")


def compare(harness: MrpcHarness, baseline: Baseline, variants) -> None:
    """Run `(label, prune)` variants from one baseline and print the table."""
    rows = [("Baseline (unpruned)", f"{accuracy(baseline.metrics):.2%}")]
    for label, prune in variants:
        rows.append((label, describe(baseline, harness.run_variant(baseline, label, prune))))
    report(rows)


def report(rows):
    print("\n=== RESULTS ===")
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def accuracy(result) -> float:
    return result["eval_accuracy"]
