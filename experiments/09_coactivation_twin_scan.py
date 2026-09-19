"""Stage 9: instead of scoring neurons by weight (Max-3, distance), score
them by *behaviour* -- run real MRPC sentences through BERT-tiny and
record which FFN neurons fire together. Neuron pairs that fire together
almost always are functionally redundant regardless of what their raw
weights look like.

Diagnostic only; stage 10 turns the same signal into actual surgery.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mrpc import set_seeds

from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

from pruning_transformer import ActivationRecorder, JaccardTwinFinder

MODEL_NAME = "prajjwal1/bert-tiny"
LAYER = 0

if __name__ == "__main__":
    set_seeds()
    print("Loading a BERT-tiny classifier (untrained head, pretrained encoder)...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading data...")
    dataset = load_dataset("glue", "mrpc", split="train[:500]")

    recorder = ActivationRecorder(model.bert, tokenizer, batch_size=32)
    fires = recorder.record(dataset["sentence1"], layer_idx=LAYER)

    twins, dead = JaccardTwinFinder().find(fires, threshold=0.95)

    print("\n=== DIAGNOSIS ===")
    print(f"Recorded {fires.shape[0]} real (non-padding) tokens x {fires.shape[1]} neurons.")
    print(f"Found {len(dead)} completely dead neurons (never fired once): {dead[:20]}")
    print(f"Found {len(twins)} twin pairs (fire together >=95% of the time).")
    for r, c, overlap in twins[:10]:
        print(f"   Neuron {r} & Neuron {c} -> overlap {overlap:.1%}")
