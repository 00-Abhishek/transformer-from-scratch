# Encoder Code — Line-by-Line Explanation

This document walks through `encoder_full.py` explaining every design decision.

---

## 1. ScaledDotProductAttention

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

`Q` shape: (B, heads, seq_q, d_k)  
`K.transpose(-2, -1)` shape: (B, heads, d_k, seq_k)  
Result: (B, heads, seq_q, seq_k) — one score for every (query position, key position) pair.

Dividing by √d_k: without this, scores grow with d_k, pushing softmax into near-zero gradient regions (the softmax saturates).

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

Sets masked positions to a large negative number. After softmax, e^(-1e9) ≈ 0, so masked positions receive zero attention weight.

---

## 2. MultiHeadAttention — _split_heads

```python
x.view(B, -1, self.h, self.d_k).transpose(1, 2)
```

Reshapes (B, seq, d_model) → (B, seq, h, d_k) → (B, h, seq, d_k).  
This allows running h independent attention computations with a single batched `torch.matmul`.

The final W_o projection merges the h heads back together:

```python
x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
return self.W_o(x)
```

`.contiguous()` is needed because `.transpose()` returns a view with non-contiguous memory, and `.view()` requires contiguous memory.

---

## 3. PositionwiseFFN

```python
self.linear2(self.dropout(F.relu(self.linear1(x))))
```

d_model → d_ff (expand): captures interactions between features  
ReLU: introduces non-linearity (attention is linear; FFN is where non-linear transformation happens)  
Dropout: regularization  
d_ff → d_model (contract): back to model dimension

This is applied **identically to every position** — the same weights, applied independently. The FFN has no cross-position communication (that is attention's job).

---

## 4. EncoderLayer — Residual + LayerNorm

```python
x = self.norm1(x + self.dropout(_attn))
```

This is **Pre-LN** if LayerNorm is inside Sublayer, or **Post-LN** if applied after the residual. The original paper uses Post-LN. Modern practice often uses Pre-LN (applies norm before the sublayer) for more stable training in very deep models:

```python
# Pre-LN variant (more stable for deep models):
_attn = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
x = x + self.dropout(_attn)
```

---

## 5. Encoder._build_pe — Sinusoidal Table

```python
div = torch.exp(
    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
)
```

This is numerically equivalent to `1 / 10000^(2i/d_model)` but computed in log-space for stability.

`pe[:, 0::2]` = even dimensions → sin  
`pe[:, 1::2]` = odd dimensions → cos

The table is registered as a buffer (not a parameter), so it moves to the right device with `.to(device)` but is not updated during training.

---

## 6. make_padding_mask Shape

```python
return (src != pad_idx).unsqueeze(1).unsqueeze(2)
```

`src`: (B, seq)  
After unsqueeze: (B, 1, 1, seq)

This broadcasts correctly against attention scores of shape (B, h, seq_q, seq_k):
- dim 0: batch — matches
- dim 1: 1 broadcasts to h heads
- dim 2: 1 broadcasts to seq_q positions
- dim 3: seq — marks which key positions are padding

---

## 7. EncoderForClassification — CLS Token

```python
cls_repr = enc_out[:, 0, :]
```

This assumes token ID at position 0 is a special [CLS] token (as in BERT). After the full encoder stack, this token's representation is a function of every other token in the sequence (via self-attention), making it a good sequence-level representation.

Alternative: mean-pool all token representations:

```python
cls_repr = enc_out.mean(dim=1)    # (B, d_model)
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forget mask | Model attends to [PAD] | Pass `make_padding_mask(src)` |
| Wrong mask sign | All attention scores -inf | Mask = 0 means PAD, 1 means real |
| No √d_k scaling | Gradient vanishes in attention | Divide scores by `math.sqrt(d_k)` |
| No residual connections | Gradient vanishes in deep stack | Add `x + Sublayer(x)` |
| Wrong embedding scale | PE dominates embeddings | Multiply embeds by `math.sqrt(d_model)` |
