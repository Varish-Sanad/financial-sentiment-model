import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer import GPT


class GPTForClassification(nn.Module):
    """Reuses a pretrained GPT's token embedding + attention blocks, swaps the
    next-token lm_head for a small classification head. Reads out the
    representation at each sequence's true last token -- under causal masking
    that's the only position that has attended to the entire input, and (with
    right-padding) it's guaranteed to be unaffected by whatever padding tokens
    follow it, since causal attention never looks forward.
    """

    def __init__(self, gpt_config, num_classes):
        super().__init__()
        self.gpt = GPT(gpt_config)
        self.classifier = nn.Linear(gpt_config.d_model, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def load_pretrained_backbone(self, state_dict):
        self.gpt.load_state_dict(state_dict)

    def forward(self, idx, lengths, labels=None):
        x = self.gpt.token_emb(idx)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)

        batch_idx = torch.arange(idx.size(0), device=idx.device)
        last_hidden = x[batch_idx, lengths - 1]  # true last real token per example
        logits = self.classifier(last_hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss
