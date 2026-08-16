import json
from pathlib import Path

import torch

from src.model.classifier import GPTForClassification
from src.tokenizer.bpe import Tokenizer
from src.training.finetune_data import LABELS, encode_examples, iterate_batches, load_examples
from src.training.pretrain import get_device

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "finetune"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
PRETRAINED_CHECKPOINT = ROOT / "checkpoints" / "pretrained.pt"
CHECKPOINT_PATH = ROOT / "checkpoints" / "finetuned.pt"
HISTORY_PATH = ROOT / "checkpoints" / "finetune_history.json"

MAX_LEN = 200
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4  # smaller than pretraining's 3e-4 so we don't wreck the pretrained backbone
SEED = 1337


@torch.no_grad()
def evaluate(model, input_ids, lengths, labels, batch_size, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, l, y in iterate_batches(input_ids, lengths, labels, batch_size, device, shuffle=False):
        logits, loss = model(x, l, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_n += x.size(0)
    model.train()
    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device}")

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    pad_id = tokenizer.vocab_size  # same reserved id used as EOS during pretraining

    checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device, weights_only=False)
    config = checkpoint["config"]

    train_examples = load_examples(DATA_DIR / "train.jsonl")
    val_examples = load_examples(DATA_DIR / "val.jsonl")
    train_ids, train_lens, train_labels = encode_examples(train_examples, tokenizer, pad_id, MAX_LEN)
    val_ids, val_lens, val_labels = encode_examples(val_examples, tokenizer, pad_id, MAX_LEN)
    print(f"train: {len(train_examples)} examples, val: {len(val_examples)} examples")

    model = GPTForClassification(config, num_classes=len(LABELS)).to(device)
    model.load_pretrained_backbone(checkpoint["model_state_dict"])
    print(f"loaded pretrained backbone from {PRETRAINED_CHECKPOINT}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        for x, l, y in iterate_batches(train_ids, train_lens, train_labels, BATCH_SIZE, device):
            _, loss = model(x, l, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_ids, train_lens, train_labels, BATCH_SIZE, device)
        val_loss, val_acc = evaluate(model, val_ids, val_lens, val_labels, BATCH_SIZE, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # deep-copy to CPU so later epochs (which keep training the live model) can't mutate this
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state_dict, "config": config}, CHECKPOINT_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    print(f"\nsaved best checkpoint (epoch {best_epoch}, val acc {best_val_acc:.3f}) to {CHECKPOINT_PATH}")
    print(f"saved history to {HISTORY_PATH}")
    print(f"final epoch val accuracy: {history['val_acc'][-1]:.3f} (for comparison -- not what's saved)")


if __name__ == "__main__":
    main()
