"""Rigor checks for the architecture, not a demo: confirms the two structural
claims the design depends on actually hold in the implementation, rather than
just trusting the code looks right.
"""

import torch

from src.model.config import GPTConfig
from src.model.rope import apply_rope, precompute_rope
from src.model.transformer import GPT


def check_forward_backward():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=8000, d_model=256, n_heads=4, n_layers=4, max_seq_len=256)
    model = GPT(config)

    B, T = 4, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, config.vocab_size), f"bad logits shape: {logits.shape}"
    assert loss.isfinite(), f"loss is not finite: {loss.item()}"

    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())

    print(f"forward/backward: logits {tuple(logits.shape)}, loss {loss.item():.3f}, "
          f"gradients on {n_grad}/{n_total} parameter tensors, model has {model.num_params():,} params")
    return n_grad == n_total


def check_causality():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=8000, d_model=256, n_heads=4, n_layers=4, max_seq_len=256)
    model = GPT(config)
    model.eval()

    T = 10
    idx = torch.randint(0, config.vocab_size, (1, T))
    with torch.no_grad():
        logits_a, _ = model(idx)

    perturbed = idx.clone()
    perturbed[0, 7] = (perturbed[0, 7] + 1) % config.vocab_size  # change a future token
    with torch.no_grad():
        logits_b, _ = model(perturbed)

    early_positions_match = torch.allclose(logits_a[0, :7], logits_b[0, :7], atol=1e-5)
    late_positions_differ = not torch.allclose(logits_a[0, 7:], logits_b[0, 7:], atol=1e-5)

    print(f"causality: positions 0-6 unaffected by change at position 7: {early_positions_match}")
    print(f"causality: positions 7-9 (>= the change) do shift: {late_positions_differ}")
    return early_positions_match and late_positions_differ


def check_rope_relative_position():
    head_dim, max_seq_len = 16, 64
    cos, sin = precompute_rope(head_dim, max_seq_len)

    torch.manual_seed(0)
    q0 = torch.randn(1, head_dim)
    k0 = torch.randn(1, head_dim)

    def rotated_dot(pos_q, pos_k):
        qr = apply_rope(q0, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        kr = apply_rope(k0, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (qr * kr).sum().item()

    same_offset_a = rotated_dot(pos_q=5, pos_k=3)    # relative offset 2
    same_offset_b = rotated_dot(pos_q=40, pos_k=38)  # relative offset 2, different absolute positions
    diff_offset = rotated_dot(pos_q=5, pos_k=2)      # relative offset 3

    matches_at_same_offset = abs(same_offset_a - same_offset_b) < 1e-4
    differs_at_different_offset = abs(same_offset_a - diff_offset) > 1e-3

    print(f"RoPE: offset-2 score at (5,3)={same_offset_a:.4f} vs (40,38)={same_offset_b:.4f} "
          f"-- match: {matches_at_same_offset}")
    print(f"RoPE: offset-3 score at (5,2)={diff_offset:.4f} differs from offset-2: {differs_at_different_offset}")
    return matches_at_same_offset and differs_at_different_offset


def main():
    ok = True
    ok &= check_forward_backward()
    ok &= check_causality()
    ok &= check_rope_relative_position()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
