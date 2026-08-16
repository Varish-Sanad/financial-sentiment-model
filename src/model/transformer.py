import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm + residual
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        # no positional embedding table -- position is injected entirely via
        # RoPE inside each attention layer instead
        self.blocks = nn.ModuleList(
            [
                Block(config.d_model, config.n_heads, config.max_seq_len, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
