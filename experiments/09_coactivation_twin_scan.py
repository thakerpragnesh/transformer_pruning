"""Stage 9: instead of scoring neurons by weight (Max-3, distance), score
them by *behavior* -- run real MRPC sentences through a trained BERT-tiny
and record which FFN neurons fire together. Neuron pairs that fire
together >95% of the time are functionally redundant regardless of what
their raw weights look like.

Assumes stage 5 has already produced a trained `model`/`tokenizer` in the
same process; here we retrain a small baseline first so this script can
run standalone.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

from pruning_transformer import ActivationRecorder, JaccardTwinFinder

MODEL_NAME = "prajjwal1/bert-tiny"

if __name__ == "__main__":
    print("Loading a BERT-tiny classifier (untrained head, pretrained encoder)...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    recorder = ActivationRecorder(model=model.bert, tokenizer=tokenizer)

    print("Loading data...")
    dataset = load_dataset("glue", "mrpc", split="train[:500]")
    texts = dataset["sentence1"]

    layer_to_scan = 0
    recorded_fires = recorder.record(texts, layer_idx=layer_to_scan)

    twins, dead = JaccardTwinFinder().find(recorded_fires, threshold=0.95)

    print("\n=== DIAGNOSIS ===")
    print(f"Found {len(dead)} completely dead neurons (never fired once).")
    print(f"Found {len(twins)} pairs of twin neurons (fire together >=95% of the time).")
    for r, c, sim in twins[:10]:
        print(f"   Neuron {r} & Neuron {c} -> Overlap: {sim:.1%}")
