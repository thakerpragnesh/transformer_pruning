"""Stage 10: the co-activation counterpart to stage 5's Max-3 experiment
-- fine-tune on MRPC, find co-activation "twin" neurons, drop one of each
pair, measure the accuracy drop, then heal with further fine-tuning.
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

from pruning_transformer import ActivationRecorder, JaccardTwinFinder, TwinRedundancySelector, prune_ffn_layer

MODEL_NAME = "prajjwal1/bert-tiny"


def main():
    print("Re-initializing model and evaluation environment...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("glue", "mrpc")

    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("glue", "mrpc")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
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

    print("\n--- Stage 1: Fine-Tuning Baseline ---")
    trainer.train()
    baseline_result = trainer.evaluate()
    print(f"Pre-Surgery Accuracy: {baseline_result['eval_accuracy']:.4f}")

    print("\n--- Stage 2: Scanning for Co-Activation Twins ---")
    recorder = ActivationRecorder(model=model.bert, tokenizer=tokenizer)
    texts = dataset["train"]["sentence1"][:500]
    recorded_fires = recorder.record(texts, layer_idx=0)
    twins, _dead = JaccardTwinFinder().find(recorded_fires, threshold=0.95)
    print(f"Found {len(twins)} twin pairs.")

    print("\n--- Stage 3: Twin Surgery ---")
    prune_ffn_layer(model.bert, layer_index=0, selector=TwinRedundancySelector(twins))

    print("\n--- Measuring 'Twin' Brain Damage ---")
    twin_damage_result = trainer.evaluate()
    print(f"Accuracy after dropping twins: {twin_damage_result['eval_accuracy']:.4f}")

    print("\n--- Final Healing (Fine-Tuning) ---")
    trainer.train()
    final_healed_result = trainer.evaluate()

    print("\n=== RESULTS: TWIN-NEURON PRUNING ===")
    print(f"1. Pre-Surgery Accuracy:  {baseline_result['eval_accuracy']:.2%}")
    print(f"2. Post-Twin Accuracy:    {twin_damage_result['eval_accuracy']:.2%}")
    print(f"3. Final Healed Accuracy: {final_healed_result['eval_accuracy']:.2%}")


if __name__ == "__main__":
    main()
