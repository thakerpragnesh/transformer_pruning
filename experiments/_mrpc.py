"""Shared MRPC setup for the two full accuracy experiments.

Stages 05 and 10 differ only in which pruning criterion they apply, but
each used to carry its own ~50-line copy of the dataset/trainer
boilerplate. Two copies of a training setup is two chances for them to
drift, and a baseline that drifts makes the comparison between the two
criteria meaningless -- which is the entire point of running both.

The one piece of behaviour worth reading carefully is `new_trainer`.
"""
import random

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


class MrpcHarness:
    """Dataset, metric, and Trainer factory for the MRPC experiments."""

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

    def calibration_batches(self, split="train", limit=512, batch_size=32):
        """Collated batches for `FFNCalibrator` / `ActivationRecorder`.

        Reuses the same tokenization and collator as evaluation, so the
        activation statistics describe the distribution the model is
        actually scored on.
        """
        from torch.utils.data import DataLoader

        keep = ("input_ids", "attention_mask", "token_type_ids")
        ds = self.tokenized[split].select(range(min(limit, len(self.tokenized[split]))))
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return list(DataLoader(ds, batch_size=batch_size, collate_fn=self.collator))


def report(rows):
    print("\n=== RESULTS ===")
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def accuracy(result) -> float:
    return result["eval_accuracy"]
