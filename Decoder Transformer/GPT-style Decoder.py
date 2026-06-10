"""
GPT-style Decoder-Only Transformer — Full PyTorch Implementation.
Includes: DecoderLayer, GPTDecoder, text generation with
temperature / top-k / top-p sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# Reuse from encoder: ScaledDotProductAttention, MultiHeadAttention,
# PositionwiseFFN (copy or import from encoder_full.py)
# ─────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k, self.d_model = h, d_model // h, d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention(dropout)

    def _split(self, x, B):
        return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q, K, V = self._split(self.W_q(Q), B), self._split(self.W_k(K), B), self._split(self.W_v(V), B)
        x, _ = self.attn(Q, K, V, mask)
        return self.W_o(x.transpose(1, 2).contiguous().view(B, -1, self.d_model))


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# DECODER LAYER  (GPT style — masked self-attention + FFN only)
# ─────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Sub-layer 1: Masked multi-head self-attention + Add & Norm
    Sub-layer 2: FFN + Add & Norm

    Causal mask is built inside forward() so it adapts to sequence length.
    """

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask: (1, 1, T, T). 1=keep, 0=mask."""
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, device = x.size(1), x.device
        mask = self._causal_mask(T, device)

        # Masked self-attention
        _attn = self.masked_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(_attn))

        # FFN
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


# ─────────────────────────────────────────────────────────────────
# FULL GPT-STYLE DECODER MODEL
# ─────────────────────────────────────────────────────────────────

class GPTDecoder(nn.Module):
    """
    GPT-style decoder-only Transformer.
    Uses learned positional embeddings (not sinusoidal).
    Weight tying: lm_head shares weights with token embedding.

    Input:  token indices (B, T)
    Output: logits over vocabulary (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)   # LEARNED (GPT style)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)

        # LM head: projects d_model → vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and output projection share weights
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape

        # Token embeddings (scaled) + positional embeddings
        tok_emb = self.token_embed(tokens) * math.sqrt(self.d_model)
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)   # (1, T)
        pos_emb = self.pos_embed(pos)                               # (1, T, d_model)

        x = self.dropout(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)   # (B, T, vocab_size)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Next-token prediction loss.
        tokens: (B, T+1) — includes BOS and EOS
        """
        src = tokens[:, :-1]    # input: first T tokens
        target = tokens[:, 1:]  # target: last T tokens (shifted by 1)

        logits = self.forward(src)   # (B, T, vocab_size)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
            target.reshape(-1),                  # (B*T,)
        )


# ─────────────────────────────────────────────────────────────────
# TEXT GENERATION
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: GPTDecoder,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> torch.Tensor:
    """
    Auto-regressive text generation.

    Args:
        prompt_ids:     (1, T) — seed token IDs
        temperature:    > 1 = more diverse, < 1 = more focused
        top_k:          if set, sample from top-k tokens only
        top_p:          if set, nucleus sampling (keep tokens summing to p)

    Returns: (1, T + max_new_tokens)
    """
    model.eval()
    ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]   # (1, vocab_size) — only last position

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, top_k)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, -1e9)

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens where cumulative prob already exceeds p
            remove_mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits[remove_mask] = -1e9
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample from filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

        ids = torch.cat([ids, next_id], dim=1)

    return ids


# ─────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB = 50257   # GPT-2 vocabulary size
    model = GPTDecoder(vocab_size=VOCAB, d_model=512, N=6, h=8, d_ff=2048, max_len=1024)

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.1f}M")

    # Forward pass
    tokens = torch.randint(0, VOCAB, (2, 32))   # batch=2, seq=32
    logits = model(tokens)
    print(f"Logits shape: {logits.shape}")       # (2, 32, 50257)

    # Loss computation
    tokens_with_eos = torch.randint(0, VOCAB, (2, 33))
    loss = model.compute_loss(tokens_with_eos)
    print(f"Loss: {loss.item():.4f}")

    # Generation
    prompt = torch.randint(0, VOCAB, (1, 5))
    generated = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_p=0.9)
    print(f"Generated sequence length: {generated.shape[1]}")
