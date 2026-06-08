"""
Standalone attention mechanism implementations.
Covers: scaled dot-product, multi-head, cross-attention,
        bahdanau (additive), and efficient variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. SCALED DOT-PRODUCT ATTENTION (standalone)
# ─────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0):
    """
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

    Args:
        Q: (*, seq_q, d_k)
        K: (*, seq_k, d_k)
        V: (*, seq_k, d_v)
        mask: (*, seq_q, seq_k) — 0 = mask out, 1 = keep
        dropout_p: dropout probability on attention weights

    Returns:
        output: (*, seq_q, d_v)
        weights: (*, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    output = torch.matmul(weights, V)
    return output, weights


# ─────────────────────────────────────────────────────────────────
# 2. BAHDANAU (ADDITIVE) ATTENTION — historical reference
# ─────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive attention from Bahdanau et al. (2015).
    Used in RNN encoder-decoder models before Transformers.

    e_{t,i} = v^T · tanh(W_s · s_{t-1} + W_h · h_i)
    α_{t,i} = softmax(e_{t,i})
    context_t = Σ_i α_{t,i} · h_i
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.W_s = nn.Linear(decoder_dim, attn_dim, bias=False)   # decoder state
        self.W_h = nn.Linear(encoder_dim, attn_dim, bias=False)   # encoder hidden
        self.v   = nn.Linear(attn_dim, 1, bias=False)             # energy scalar

    def forward(self, decoder_state, encoder_outputs):
        """
        decoder_state:   (B, decoder_dim)
        encoder_outputs: (B, src_len, encoder_dim)

        Returns:
            context: (B, encoder_dim)
            weights: (B, src_len)
        """
        # Expand decoder state to match encoder sequence length
        s = self.W_s(decoder_state).unsqueeze(1)   # (B, 1, attn_dim)
        h = self.W_h(encoder_outputs)               # (B, src_len, attn_dim)

        energy = self.v(torch.tanh(s + h)).squeeze(-1)   # (B, src_len)
        weights = F.softmax(energy, dim=-1)               # (B, src_len)

        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)   # (B, enc_dim)
        return context, weights


# ─────────────────────────────────────────────────────────────────
# 3. NUMPY DEMO — attention from scratch (educational)
# ─────────────────────────────────────────────────────────────────

def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation. For understanding only — not efficient.
    Q, K: (seq, d_k)
    V:    (seq, d_v)
    """
    d_k = Q.shape[-1]

    # Raw attention scores
    scores = Q @ K.T / np.sqrt(d_k)   # (seq, seq)

    # Softmax (numerically stable)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # Weighted sum
    output = weights @ V   # (seq, d_v)

    return output, weights


# ─────────────────────────────────────────────────────────────────
# 4. CAUSAL MASK UTILITIES
# ─────────────────────────────────────────────────────────────────

def causal_mask(T: int, device=None) -> torch.Tensor:
    """
    Lower-triangular mask for decoder self-attention.
    Shape: (1, 1, T, T) — broadcasts over (B, h, T, T).
    1 = keep, 0 = mask.
    """
    return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)


def padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Mask for padding tokens.
    Shape: (B, 1, 1, seq_len) — broadcasts over (B, h, seq_q, seq_k).
    1 = real token, 0 = padding.
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def combined_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Combine causal mask + padding mask for decoder self-attention.
    Result is 1 where both conditions are met.
    """
    T = src.size(1)
    c_mask = causal_mask(T, device=src.device)           # (1, 1, T, T)
    p_mask = padding_mask(src, pad_idx)                   # (B, 1, 1, T)
    return c_mask & p_mask                                # (B, 1, T, T)


# ─────────────────────────────────────────────────────────────────
# 5. ATTENTION VISUALIZATION HELPER
# ─────────────────────────────────────────────────────────────────

def attention_heatmap_text(tokens_q, tokens_k, weights, head=0):
    """
    Print a text-based heatmap of attention weights.
    weights: (h, seq_q, seq_k) — e.g. from a single example
    """
    attn = weights[head].detach().cpu().numpy()   # (seq_q, seq_k)

    col_width = max(len(t) for t in tokens_k) + 2
    header = " " * col_width + "".join(t.center(col_width) for t in tokens_k)
    print(f"\nHead {head} attention weights:")
    print(header)

    for i, row_token in enumerate(tokens_q):
        row = row_token.ljust(col_width)
        for j in range(len(tokens_k)):
            val = attn[i, j]
            # Use block characters for intensity
            if val < 0.05:
                row += " ░ ".center(col_width)
            elif val < 0.20:
                row += " ▒ ".center(col_width)
            elif val < 0.50:
                row += " ▓ ".center(col_width)
            else:
                row += " █ ".center(col_width)
        print(f"{row}  (row sum: {attn[i].sum():.3f})")


# ─────────────────────────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # NumPy demo
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    out, w = attention_numpy(Q, K, V)
    print("=== NumPy Attention Demo ===")
    print(f"Input Q shape: {Q.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Weights row 0 (sum to 1): {w[0].sum():.6f}")
    print(f"Weights:\n{w.round(3)}")

    # PyTorch causal mask demo
    print("\n=== Causal Mask (T=4) ===")
    mask = causal_mask(4)
    print(mask.squeeze())

    # Bahdanau attention demo
    print("\n=== Bahdanau Attention Demo ===")
    bah_attn = BahdanauAttention(encoder_dim=32, decoder_dim=32, attn_dim=16)
    enc_out = torch.randn(2, 10, 32)   # batch=2, src_len=10, enc_dim=32
    dec_state = torch.randn(2, 32)     # batch=2, dec_dim=32
    ctx, attn_w = bah_attn(dec_state, enc_out)
    print(f"Context shape:  {ctx.shape}")      # (2, 32)
    print(f"Weights shape:  {attn_w.shape}")   # (2, 10)
    print(f"Weights sum:    {attn_w.sum(dim=-1)}")  # should be [1, 1]
