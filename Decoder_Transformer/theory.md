# Decoder Transformer — Theory and Mathematics

## Role of the Decoder

The decoder **generates sequences one token at a time**. Unlike the encoder (which processes the full input simultaneously), the decoder is auto-regressive: each new token is conditioned on all previously generated tokens.

Decoder-only models (GPT-2, GPT-3, GPT-4, LLaMA, Claude) use this architecture for text generation, code generation, dialogue, and reasoning.

---

## Auto-Regressive Generation

The decoder models the joint probability of a sequence using the chain rule:

```
P(t_1, t_2, ..., t_n) = ∏_{i=1}^{n} P(t_i | t_1, t_2, ..., t_{i-1})
```

At inference time:
1. Start with a prompt (seed tokens)
2. Compute P(next token | all previous tokens)
3. Sample or select the next token
4. Append it to the sequence
5. Repeat until [EOS] or max length

At training time, all positions are predicted in parallel using **teacher forcing**: the model receives the true previous tokens (not its own predictions). The causal mask prevents cheating.

---

## The Causal Mask

Without a mask, position i could attend to position j > i — seeing the answer before predicting it. The causal (look-ahead) mask prevents this:

```
M[i, j] = { 0    if j ≤ i    (allowed: same position or earlier)
           { -∞   if j > i    (blocked: future positions)
```

For a sequence of length 4:

```
Mask =  1 0 0 0     ← token 0 sees only itself
        1 1 0 0     ← token 1 sees 0,1
        1 1 1 0     ← token 2 sees 0,1,2
        1 1 1 1     ← token 3 sees everything
```

This lower-triangular binary matrix is the causal mask. After adding -∞ to blocked positions and applying softmax, future tokens receive exactly 0 attention weight.

---

## GPT-Style Decoder Architecture

```
Input tokens (shifted right — starts with [BOS])
     │
     ▼
[Token Embedding] + [Learned Positional Embedding]
     │
     ▼
┌──────────────────────────────────────┐
│  Decoder Layer × N                  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │  Masked Multi-Head Self-Attn   │  │
│  │  (causal mask applied)         │  │
│  └────────────────┬───────────────┘  │
│                   │                  │
│  ┌────────────────▼───────────────┐  │
│  │  Add & LayerNorm               │  │
│  └────────────────┬───────────────┘  │
│                   │                  │
│  ┌────────────────▼───────────────┐  │
│  │  Position-wise FFN             │  │
│  └────────────────┬───────────────┘  │
│                   │                  │
│  ┌────────────────▼───────────────┐  │
│  │  Add & LayerNorm               │  │
│  └────────────────┬───────────────┘  │
└───────────────────┼──────────────────┘
                    │ (repeat N times)
                    ▼
             Final LayerNorm
                    │
                    ▼
         Linear (d_model → vocab_size)  ← language model head
                    │
                    ▼
              Softmax / logits
```

Note: GPT-style decoder has **2 sub-layers** per layer (no cross-attention). The full encoder-decoder Transformer has 3 sub-layers in the decoder.

---

## Learned vs. Sinusoidal Positional Encoding

GPT uses **learned** positional embeddings:

```python
pos_embed = nn.Embedding(max_len, d_model)   # trainable lookup table
x = token_embed + pos_embed(positions)
```

**Advantage:** Can learn complex position-dependent patterns  
**Disadvantage:** Cannot generalize beyond max_len trained on

The original Transformer uses sinusoidal PE, which in principle generalizes to arbitrary lengths.

---

## Temperature and Sampling Strategies

After the LM head produces logits, a token must be selected:

### Temperature

```
P(token | context) = softmax(logits / T)
```

- T → 0: argmax (greedy, deterministic)
- T = 1: original distribution
- T > 1: flatter, more uniform (more diverse but less coherent)

### Top-k Sampling

Keep only the k highest-probability tokens, re-normalize, sample:

```
k = 50:  keep top 50 vocab entries, zero the rest, sample
```

### Top-p (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability exceeds p:

```
p = 0.9: if top 2 tokens have cumulative prob 0.92, keep those 2
         if top 2 have cumulative prob 0.7, extend until ≥ 0.9
```

Top-p adapts to the model's confidence: when confident (peaked distribution), nucleus is small; when uncertain, nucleus is large.

### Beam Search

Maintain B "beams" (partial sequences) at each step. Extend each beam with all vocabulary tokens, keep the B best by cumulative log-probability. Better than greedy for structured tasks (translation), but not ideal for open-ended generation (tends to be repetitive).

---

## Output Head and Weight Tying

```
logits = LayerNorm(dec_output) · W_lm     W_lm ∈ ℝ^(d_model × vocab_size)
```

Weight tying: set `W_lm = E^T` (transpose of token embedding matrix). The embedding space and prediction space are forced to be consistent. Reduces parameter count by vocab_size × d_model ≈ 25M+ parameters.

---

## Exposure Bias

During training (teacher forcing): model always sees correct previous tokens  
During inference: model sees its own (possibly wrong) previous predictions

This mismatch is called **exposure bias**. Errors can compound: one bad prediction makes the next prediction worse. Techniques to address it:
- **Scheduled sampling**: gradually replace teacher tokens with model predictions during training
- **REINFORCE**: train directly on sampled sequences with reward signal
- **Minimum Bayes Risk decoding**: generate many candidates, pick the one most similar to the rest

---

## Hyperparameters

| Parameter | GPT-1 | GPT-2 (med) | GPT-3 (175B) |
|-----------|-------|-------------|--------------|
| Layers N | 12 | 24 | 96 |
| d_model | 768 | 1024 | 12288 |
| Heads h | 12 | 16 | 96 |
| d_ff | 3072 | 4096 | 49152 |
| Max context | 512 | 1024 | 2048 |
