# Encoder-Decoder Code — Explanation

## FullDecoderLayer: Why Three Sub-layers?

The GPT decoder has 2 sub-layers (self-attn + FFN). The full encoder-decoder decoder has 3:

```python
def forward(self, tgt, memory, src_mask=None):
    # 1. Causal self-attention: track what was already generated
    tgt = self.norm1(tgt + self.dropout(self.self_attn(tgt, tgt, tgt, tgt_mask)))

    # 2. Cross-attention: consult the encoder's understanding
    tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt, memory, memory, src_mask)))

    # 3. FFN: transform each position's representation
    tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
```

Sub-layer 2 is the essential addition. Without it, the decoder can only see what it has generated so far — it has no access to the source.

---

## Cross-Attention: What Flows Where

```python
self.cross_attn(tgt, memory, memory, src_mask)
```

In `MultiHeadAttention.forward(Q, K, V, mask)`:
- `Q` = `tgt` projected through `W_q` → "what is the decoder currently thinking?"
- `K` = `memory` projected through `W_k` → "what does each source token offer?"
- `V` = `memory` projected through `W_v` → "what is the actual content of each source token?"

The attention weight `softmax(QKᵀ / √d_k)` answers: "given what the decoder is currently thinking (Q), how much should it attend to each source token (K)?"

The output is a weighted sum of source token contents (V), where weights come from Q-K similarity.

**The src_mask here** is the source **padding** mask (not a causal mask). The decoder can freely attend to any non-padding source position.

---

## encode() Called Once, decode() Called T Times

```python
memory = model.encode(src, src_mask)   # ← once per source sequence
for step in range(max_len):
    dec_out = model.decode(tgt, memory, src_mask)   # ← once per generated token
```

The encoder's computation is amortized: O(src_len² · d) computed once, then reused for every decoding step. This is more efficient than repeating the encoder computation at each step.

---

## Teacher Forcing Mechanics

```python
tgt_in  = tgt[:, :-1]   # [BOS, w1, w2, w3]      ← decoder sees these
tgt_out = tgt[:, 1:]    # [w1,  w2, w3, EOS]      ← model must predict these
```

The causal mask ensures position 0 only sees [BOS], position 1 sees [BOS, w1], etc. So even though all positions are computed in parallel, each position only has access to tokens that would have been generated before it during inference.

The loss is cross-entropy at every position simultaneously — no sequential loop needed during training.

---

## LabelSmoothingLoss

```python
smooth = torch.full_like(log_probs, self.epsilon / (self.vocab_size - 1))
smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
smooth[targets == self.pad_idx] = 0.0
```

Step by step:
1. Fill every cell with `ε / (V-1)` → uniform distribution scaled to ε
2. Set the true class to `1 - ε` → most of probability mass on correct token
3. Zero out padding positions → no gradient from [PAD] tokens

Result: instead of a hard 0/1 target, each training example has a "soft" target. The model is penalized for being too confident, even about the correct answer.

```
Hard target: [0, 0, 1, 0, 0]         (ε=0)
Soft target: [0.025, 0.025, 0.9, 0.025, 0.025]  (ε=0.1, V=5)
```

---

## Beam Search: Why Length Penalty?

Without length penalty, beam scores are sums of log-probabilities. A sequence of length 3 has 3 terms; length 10 has 10 terms. Log-probs are negative, so longer sequences systematically score lower.

```python
key=lambda x: x[0] / (len(x[1]) ** length_penalty)
```

Dividing by length^α normalizes for length. α=0 → no penalty (favor length). α=1 → full normalization. α≈0.6–0.7 gives a sweet spot: slightly favors complete, longer translations without massively penalizing short ones.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Using causal mask in cross-attention | Decoder can't see future source tokens | No mask in cross-attn |
| Forgetting src_mask in cross-attention | Attends to [PAD] source tokens | Pass src_mask to cross_attn |
| No teacher forcing shift | Position i predicts itself, not next | `tgt_in = tgt[:, :-1]`, `tgt_out = tgt[:, 1:]` |
| Recomputing encoder per decode step | Very slow inference | Call `encode()` once, cache memory |
| Beam search without length penalty | Prefers short translations | Apply `/ len^α` when scoring |
