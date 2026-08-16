"""Corpus collection: pulls headlines + earnings-call transcripts (pretraining,
unlabeled) and Financial PhraseBank (fine-tuning, labeled) from Hugging Face,
and writes them out as train/val JSONL files.

Pretraining and fine-tuning sources are kept physically separate end to end —
the fine-tuning set never contributes text to the pretraining split, and we
explicitly check for accidental overlap before writing anything to disk.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEED = 1337

MAX_HEADLINES = 60_000     # of 306,242 available
MAX_TRANSCRIPTS = 400      # of 33,362 available; transcripts average ~53k chars each
PRETRAIN_VAL_FRACTION = 0.05


def collect_headlines(max_examples=MAX_HEADLINES):
    ds = load_dataset("ashraq/financial-news-articles", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(max_examples, len(ds))))
    return [{"text": row["title"], "source": "headline"} for row in ds if row["title"]]


def collect_transcripts(max_examples=MAX_TRANSCRIPTS):
    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(max_examples, len(ds))))
    return [
        {
            "text": row["content"],
            "source": "transcript",
            "symbol": row["symbol"],
            "date": row["date"],
        }
        for row in ds
        if row["content"]
    ]


def collect_labeled_sentiment():
    train = load_dataset("FinanceMTEB/financial_phrasebank", split="train")
    test = load_dataset("FinanceMTEB/financial_phrasebank", split="test")
    to_records = lambda split: [
        {"text": row["text"], "label": row["label_text"]} for row in split
    ]
    return to_records(train), to_records(test)


def split_train_val(records, val_fraction, seed=SEED):
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def check_no_leakage(pretrain_records, finetune_records):
    pretrain_texts = {r["text"] for r in pretrain_records}
    finetune_texts = {r["text"] for r in finetune_records}
    overlap = pretrain_texts & finetune_texts
    if overlap:
        print(f"  leakage check: removing {len(overlap)} overlapping text(s) from pretraining set")
        pretrain_records = [r for r in pretrain_records if r["text"] not in overlap]
    else:
        print("  leakage check: no overlap found")
    return pretrain_records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    print("collecting headlines...")
    headlines = collect_headlines()
    print(f"  {len(headlines)} headlines")

    print("collecting earnings call transcripts...")
    transcripts = collect_transcripts()
    print(f"  {len(transcripts)} transcripts")

    print("collecting labeled sentiment (Financial PhraseBank)...")
    finetune_train, finetune_val = collect_labeled_sentiment()
    print(f"  {len(finetune_train)} train / {len(finetune_val)} val labeled examples")

    pretrain_all = headlines + transcripts
    pretrain_all = check_no_leakage(pretrain_all, finetune_train + finetune_val)
    pretrain_train, pretrain_val = split_train_val(pretrain_all, PRETRAIN_VAL_FRACTION)

    write_jsonl(pretrain_train, DATA_DIR / "pretrain" / "train.jsonl")
    write_jsonl(pretrain_val, DATA_DIR / "pretrain" / "val.jsonl")
    write_jsonl(finetune_train, DATA_DIR / "finetune" / "train.jsonl")
    write_jsonl(finetune_val, DATA_DIR / "finetune" / "val.jsonl")

    def char_count(records):
        return sum(len(r["text"]) for r in records)

    print("\nfinal corpus sizes:")
    print(f"  pretrain/train.jsonl: {len(pretrain_train)} docs, {char_count(pretrain_train):,} chars")
    print(f"  pretrain/val.jsonl:   {len(pretrain_val)} docs, {char_count(pretrain_val):,} chars")
    print(f"  finetune/train.jsonl: {len(finetune_train)} docs, {char_count(finetune_train):,} chars")
    print(f"  finetune/val.jsonl:   {len(finetune_val)} docs, {char_count(finetune_val):,} chars")


if __name__ == "__main__":
    main()
