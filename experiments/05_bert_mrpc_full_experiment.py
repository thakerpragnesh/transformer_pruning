"""Stage 5: the full accuracy experiment -- fine-tune BERT-tiny on MRPC,
prune 40% of layer 0's FFN neurons with Max-3 saliency, measure the
accuracy drop, then fine-tune again ("heal") and measure recovery.
"""
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pruning_transformer import Max3SaliencyScorer, SaliencySelector, prune_ffn_layer

MODEL_NAME = "prajjwal1/bert-tiny"


def main():
    print("Loading MRPC dataset (paraphrase detection)...")
    dataset = load_dataset("glue", "mrpc")
    metric = evaluate.load("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print("\n--- Phase 1: Training Baseline Model ---")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    baseline_result = trainer.evaluate()
    print(f"Baseline Accuracy: {baseline_result['eval_accuracy']:.4f}")

    print("\n--- Phase 2: Applying Max-3 Pruning ---")
    selector = SaliencySelector(Max3SaliencyScorer(), prune_percent=40)
    prune_ffn_layer(model.bert, layer_index=0, selector=selector)

    print("\n--- Phase 3: Measuring Damage ---")
    damage_result = trainer.evaluate()
    print(f"Post-Pruning Accuracy: {damage_result['eval_accuracy']:.4f}")
    drop = baseline_result["eval_accuracy"] - damage_result["eval_accuracy"]
    print(f"Accuracy Drop: {drop:.4f}")

    print("\n--- Phase 4: Healing (Fine-Tuning) ---")
    trainer.train()
    healed_result = trainer.evaluate()

    print("\n=== FINAL RESULTS ===")
    print(f"1. Original Accuracy: {baseline_result['eval_accuracy']:.2%}")
    print(f"2. Pruned Accuracy:   {damage_result['eval_accuracy']:.2%}")
    print(f"3. Healed Accuracy:   {healed_result['eval_accuracy']:.2%}")
    print("4. Compression:       Layer 0 FFN reduced by 40%")


if __name__ == "__main__":
    main()
