import json

import torch

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def encode_examples(examples, tokenizer, pad_id, max_len):
    input_ids, lengths, labels = [], [], []
    for ex in examples:
        ids = tokenizer.encode(ex["text"])[:max_len]
        length = len(ids)
        ids = ids + [pad_id] * (max_len - length)  # right-pad; causal masking keeps this safe to read from
        input_ids.append(ids)
        lengths.append(length)
        labels.append(LABEL2ID[ex["label"]])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def iterate_batches(input_ids, lengths, labels, batch_size, device, shuffle=True):
    n = input_ids.size(0)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield (
            input_ids[idx].to(device),
            lengths[idx].to(device),
            labels[idx].to(device),
        )
