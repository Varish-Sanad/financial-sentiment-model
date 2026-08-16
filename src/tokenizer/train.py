import json
import random
import time
from pathlib import Path

from src.tokenizer.bpe import Tokenizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "tokenizer" / "tokenizer.json"

VOCAB_SIZE = 8000  # deliberately small, matched to a small model + ~25M-char corpus
SEED = 1337


def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def round_trip_check(tokenizer, texts, n_samples=200):
    rng = random.Random(SEED)
    sample = rng.sample(texts, min(n_samples, len(texts)))
    failures = 0
    for text in sample:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        if decoded != text:
            failures += 1
    return failures, len(sample)


def compression_ratio(tokenizer, texts, n_samples=500):
    rng = random.Random(SEED)
    sample = rng.sample(texts, min(n_samples, len(texts)))
    total_chars = sum(len(t) for t in sample)
    total_tokens = sum(len(tokenizer.encode(t)) for t in sample)
    return total_chars / total_tokens


def main():
    print("loading pretraining corpus...")
    train_texts = load_texts(DATA_DIR / "pretrain" / "train.jsonl")
    val_texts = load_texts(DATA_DIR / "pretrain" / "val.jsonl")
    print(f"  {len(train_texts)} training documents")

    print(f"training BPE tokenizer, target vocab size {VOCAB_SIZE}...")
    start = time.time()
    tokenizer = Tokenizer.train(train_texts, vocab_size=VOCAB_SIZE)
    elapsed = time.time() - start
    print(f"  trained in {elapsed:.1f}s, final vocab size {tokenizer.vocab_size}")

    tokenizer.save(TOKENIZER_PATH)
    print(f"  saved to {TOKENIZER_PATH}")

    print("round-trip check on held-out validation text (never used in training)...")
    failures, n = round_trip_check(tokenizer, val_texts)
    print(f"  {n - failures}/{n} exact round-trips" + (" -- ALL PASSED" if failures == 0 else " -- FAILURES FOUND"))

    ratio = compression_ratio(tokenizer, val_texts)
    print(f"  compression ratio on val set: {ratio:.2f} chars/token")


if __name__ == "__main__":
    main()
