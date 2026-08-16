import json
import time
from pathlib import Path

import torch

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.tokenizer.bpe import Tokenizer
from src.training.data import build_token_stream, get_batch, load_texts

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "pretrain"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "pretrained.pt"
HISTORY_PATH = ROOT / "checkpoints" / "pretrain_history.json"

BATCH_SIZE = 32
SEQ_LEN = 256
MAX_STEPS = 3000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20  # averaged per eval, to smooth out single-batch noise
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
SEED = 1337


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model, stream, device):
    model.eval()
    losses = torch.zeros(EVAL_BATCHES)
    for i in range(EVAL_BATCHES):
        x, y = get_batch(stream, BATCH_SIZE, SEQ_LEN, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device}")

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    eos_id = tokenizer.vocab_size  # one past the trained vocab, reserved as a document separator
    vocab_size = tokenizer.vocab_size + 1

    print("tokenizing corpus into a single stream (train/val kept separate)...")
    train_texts = load_texts(DATA_DIR / "train.jsonl")
    val_texts = load_texts(DATA_DIR / "val.jsonl")
    train_stream = build_token_stream(train_texts, tokenizer, eos_id)
    val_stream = build_token_stream(val_texts, tokenizer, eos_id)
    print(f"  train stream: {len(train_stream):,} tokens")
    print(f"  val stream:   {len(val_stream):,} tokens")

    config = GPTConfig(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=4, max_seq_len=SEQ_LEN)
    model = GPT(config).to(device)
    print(f"model: {model.num_params():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = {"step": [], "train_loss": [], "val_loss": []}
    start = time.time()

    for step in range(MAX_STEPS):
        x, y = get_batch(train_stream, BATCH_SIZE, SEQ_LEN, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            train_loss = estimate_loss(model, train_stream, device)
            val_loss = estimate_loss(model, val_stream, device)
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            elapsed = time.time() - start
            print(f"step {step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {elapsed:.0f}s elapsed")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, CHECKPOINT_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print(f"\nsaved checkpoint to {CHECKPOINT_PATH}")
    print(f"saved loss history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
