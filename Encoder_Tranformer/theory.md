# Encoder Transformer — Theory and Mathematics

## Role of the Encoder

The encoder reads the full input sequence and produces a matrix of contextual representations — one d_model-dimensional vector per input token. These representations capture not just the meaning of each token in isolation, but its meaning in context (the word "bank" near "river" vs. near "loan" gets different representations).

Encoder-only models (BERT, RoBERTa, DistilBERT) use this architecture for tasks requiring deep bidirectional understanding: classification, NER, question answering, semantic similarity.

---

## Architecture Overview

```
Input tokens
     │
     ▼
[Token Embedding × √d_model]  +  [Sinusoidal Positional Encoding]
     │
     ▼
┌────────────────────────────────────┐
│  Encoder Layer × N (N=6)           │
│                                    │
│  ┌──────────────────────────────┐  │
│  │  Multi-Head Self-Attention   │  │
│  └──────────────┬───────────────┘  │
│                 │                  │
│  ┌──────────────▼───────────────┐  │
│  │  Add & LayerNorm             │  │
│  │  out = LayerNorm(x + Attn(x))│  │
│  └──────────────┬───────────────┘  │
│                 │                  │
│  ┌──────────────▼───────────────┐  │
│  │  Position-wise FFN           │  │
│  └──────────────┬───────────────┘  │
│                 │                  │
│  ┌──────────────▼───────────────┐  │
│  │  Add & LayerNorm             │  │
│  └──────────────┬───────────────┘  │
└─────────────────┼──────────────────┘
                  │ (repeat N times)
                  ▼
          Encoder Output
    (batch, seq_len, d_model)
```

---

## Sub-layer 1: Multi-Head Self-Attention

In the encoder, Q = K = V = the same input sequence. Every token attends to every other token (including itself) simultaneously.

```
Q = X · W^Q       W^Q ∈ ℝ^(d_model × d_model)
K = X · W^K       W^K ∈ ℝ^(d_model × d_model)
V = X · W^V       W^V ∈ ℝ^(d_model × d_model)

head_i = Attention(X·W_i^Q, X·W_i^K, X·W_i^V)
       = softmax( (X·W_i^Q)(X·W_i^K)ᵀ / √d_k ) · X·W_i^V

MultiHead(X) = Concat(head_1,...,head_h) · W^O
```

With h=8, d_model=512: each head has d_k = d_v = 64. Total computation ≈ same as one 512-dimensional attention head, but the model attends from 8 different learned perspectives simultaneously.

**No mask** — encoder sees the full sequence bidirectionally. This is the critical difference from the decoder.

---

## Residual Connections + Layer Normalization

After each sub-layer:

```
output = LayerNorm(x + Sublayer(x))
```

**Residual connections** (He et al., 2016) prevent the vanishing gradient problem in deep networks. The gradient can flow directly through the skip connection without passing through the sublayer's parameters.

**Layer Normalization** normalizes across the feature dimension (not the batch dimension). For each token position independently:

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

Where μ and σ are the mean and std of the d_model values for that position. γ and β are learned scale and shift parameters.

---

## Sub-layer 2: Position-wise Feed-Forward Network

Applied identically and independently to each token position:

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
```

Dimensions:
- Input / output: d_model = 512
- Inner dimension: d_ff = 2048 (4× expansion)

The FFN provides additional model capacity and introduces non-linearity that attention lacks. Attention aggregates information across positions; FFN transforms each position's representation.

Different encoder layers specialize in different types of features:
- **Lower layers (1-2):** Token identity, character composition, local context
- **Middle layers (3-4):** Syntactic structure (POS tags, dependency relations, named entities)
- **Higher layers (5-6):** Semantics and pragmatics (coreference, semantic roles, entailment)

---

## Padding Mask

When batches contain sequences of different lengths, shorter sequences are padded with [PAD] tokens. The attention mechanism must ignore these:

```
padding_mask[b, i] = { -∞   if token i is [PAD]
                      {  0   otherwise

Masked scores = (QKᵀ / √d_k) + padding_mask
Attention weights = softmax(masked_scores)
```

After softmax, [PAD] positions receive exactly 0 attention weight.

---

## Hyperparameters

| Parameter | Base Model | Large Model |
|-----------|-----------|-------------|
| N (encoder layers) | 6 | 6 |
| d_model | 512 | 1024 |
| h (attention heads) | 8 | 16 |
| d_k = d_v | 64 | 64 |
| d_ff | 2048 | 4096 |
| Dropout | 0.1 | 0.3 |

---

## Complexity Analysis

| Operation | Time | Memory |
|-----------|------|--------|
| Self-attention (per layer) | O(n² · d) | O(n²) |
| FFN (per layer) | O(n · d²) | O(n · d) |
| N encoder layers | O(N · n² · d + N · n · d²) | O(n²) |

For typical tasks (n < 512), the n² term is manageable. For long documents, it becomes the bottleneck.
