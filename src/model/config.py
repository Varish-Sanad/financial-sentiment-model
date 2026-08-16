from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000  # must match tokenizer/tokenizer.json
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
