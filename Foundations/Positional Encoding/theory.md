# Positional Encoding

## The Problem

Self-attention is **permutation-invariant**: it computes the same output regardless of the order of tokens. The sentence "cat sat on mat" and "mat on sat cat" would produce identical attention scores if position information is not injected.

RNNs got position for free (they process tokens in order). Transformers need to add it explicitly.

---

## Sinusoidal Positional Encoding (Original Paper)

For each position `pos` (0-indexed) and each dimension `i` of the d_model-dimensional embedding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

The resulting PE matrix has shape (max_seq_len, d_model). The PE vector for each position is added element-wise to the token embedding vector.

### Why sinusoidal?

1. **Bounded values**: sin/cos always in [-1, 1], so PEs don't dominate embeddings
2. **Relative positions learnable**: PE(pos+k) is a linear function of PE(pos) for any fixed offset k — the model can learn to compute relative distances
3. **Generalizes to unseen lengths**: unlike learned embeddings, sinusoidal PE can extend to sequences longer than those seen in training
4. **Different frequencies per dimension**: low-frequency dimensions change slowly (coarse position), high-frequency dimensions change rapidly (fine position) — like a binary clock

### Frequency intuition:

```
Dimension 0-1:    period ≈ 2π         (changes every ~6 tokens)
Dimension 256-257: period ≈ 10000     (changes every ~63000 tokens)
Dimension 510-511: period ≈ 10000²   (almost constant over normal sequences)
```

---

## Learned Positional Embeddings (GPT style)

Instead of a fixed formula, maintain a trainable embedding table:

```
pos_embed = nn.Embedding(max_len, d_model)
x = token_embed(tokens) + pos_embed(positions)
```

**Advantage:** Can capture more complex position-dependent patterns  
**Disadvantage:** Cannot generalize beyond max_len seen during training

---

## Rotary Position Embedding (RoPE) — Modern Standard

Used by LLaMA, GPT-NeoX, Mistral, etc. Instead of adding positional information to embeddings, RoPE *rotates* the Q and K vectors before computing attention:

```
q_m = R(m) · q     (rotate by position m)
k_n = R(n) · k     (rotate by position n)
q_m · k_n = q · R(n-m) · k    (dot product depends only on relative position m-n)
```

This gives the model true relative position information in every attention layer, not just at the input.

---

## ALiBi (Attention with Linear Biases)

Add a position-dependent bias directly to attention scores before softmax:

```
Attention score(i, j) = (q_i · k_j / √d_k)  -  m · |i - j|
```

Where m is a head-specific slope. Closer tokens get smaller penalty, farther tokens get larger penalty. ALiBi generalizes well to sequences longer than training length.

---

## Summary

| Method | Generalizes beyond max_len? | Relative position | Used by |
|--------|----------------------------|-------------------|---------|
| Sinusoidal (fixed) | Yes | Linear functions | Original Transformer |
| Learned absolute | No | None | GPT-2, BERT |
| RoPE | Yes (with tuning) | Native | LLaMA, Mistral, GPT-NeoX |
| ALiBi | Yes | Linear bias | MPT, BLOOM |
