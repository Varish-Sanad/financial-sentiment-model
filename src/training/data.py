"""Concatenates all documents into one long token stream, separated by an
EOS token so the model can learn document boundaries, then samples random
fixed-length windows from that stream for training -- standard approach for
pretraining on many short/variable-length documents rather than padding each
one individually.
"""

import json

import torch


def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def build_token_stream(texts, tokenizer, eos_id):
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(stream, batch_size, seq_len, device):
    max_start = len(stream) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([stream[i : i + seq_len] for i in starts])
    y = torch.stack([stream[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)
