"""Byte-level BPE tokenizer, built from scratch (no off-the-shelf tokenizer library).

Operates on UTF-8 bytes rather than characters, so the base vocabulary is fixed
at the 256 possible byte values and there is never an out-of-vocabulary input --
anything unmergeable just falls back to raw bytes, the same guarantee GPT-2's
tokenizer relies on.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Simplified GPT-2-style pre-tokenizer pattern: keeps contractions, letter runs,
# digit runs, and punctuation runs as separate chunks (each with an optional
# leading space folded in) so merges never cross word/space boundaries.
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s"""
)


def get_pretoken_counts(texts):
    counts = Counter()
    for text in texts:
        for m in PRETOKEN_PATTERN.finditer(text):
            counts[m.group()] += 1
    return counts


def merge_pair(word, pair, new_id):
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def train_bpe(pretoken_counts, vocab_size, min_pair_count=2):
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    word_freqs = {}
    for token, count in pretoken_counts.items():
        b = tuple(token.encode("utf-8"))
        word_freqs[b] = word_freqs.get(b, 0) + count

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    num_merges = vocab_size - 256
    for i in range(num_merges):
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < min_pair_count:
            break

        new_id = 256 + i
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append(best_pair)

        for word in list(pair_to_words[best_pair]):
            freq = word_freqs.pop(word, None)
            if freq is None:
                continue  # already replaced by an earlier merge this same step

            for pair in zip(word, word[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word)

            new_word = merge_pair(word, best_pair, new_id)
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_word)

        del pair_to_words[best_pair]

    return merges, vocab


class Tokenizer:
    def __init__(self, merges, vocab):
        self.merges = merges
        self.vocab = vocab
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def train(cls, texts, vocab_size, min_pair_count=2):
        pretoken_counts = get_pretoken_counts(texts)
        merges, vocab = train_bpe(pretoken_counts, vocab_size, min_pair_count)
        return cls(merges, vocab)

    def _encode_pretoken(self, token_str):
        word = tuple(token_str.encode("utf-8"))
        while len(word) >= 2:
            pairs = list(zip(word, word[1:]))
            ranked = [p for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            best_pair = min(ranked, key=lambda p: self.merge_ranks[p])
            new_id = 256 + self.merge_ranks[best_pair]
            word = merge_pair(word, best_pair, new_id)
        return list(word)

    def encode(self, text):
        ids = []
        for m in PRETOKEN_PATTERN.finditer(text):
            ids.extend(self._encode_pretoken(m.group()))
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merges": [list(p) for p in self.merges],
            "vocab": {str(i): b.hex() for i, b in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        merges = [tuple(p) for p in data["merges"]]
        vocab = {int(i): bytes.fromhex(h) for i, h in data["vocab"].items()}
        return cls(merges, vocab)
