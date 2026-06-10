# Encoder-Decoder Transformer — Theory and Mathematics

## What It Is

The full encoder-decoder Transformer, from Vaswani et al. (2017) "Attention Is All You Need", combines both stacks into a single system for **sequence-to-sequence** tasks: the input and output are different sequences (possibly different length, different language, different modality).

Classic applications: machine translation, text summarization, question answering, data-to-text generation.

Models built on this architecture: **T5, BART, mBart, MarianMT**.

---

## Why Encoder-Decoder?

A decoder-only model can handle translation by treating the source as part of the context — and modern LLMs do this well. But for specialized seq2seq tasks with limited compute:

- **Encoder builds a deep, bidirectional understanding** of the source (no generation distraction)
- **Decoder focuses entirely on generation**, consulting the encoder's understanding via cross-attention at every step
- The separation is a useful inductive bias for translation-like tasks
- Cross-attention gives every decoder position direct access to every source position (no bottleneck)

---

## Architecture Overview

```
SOURCE SEQUENCE
      │
      ▼
[Source Embedding + Sinusoidal PE]
      │
      ▼
┌────────────────────────────────────────┐
│           ENCODER STACK (N=6)          │
│  [Multi-Head Self-Attention]           │
│  [Add & Norm]                          │
│  [FFN]                                 │
│  [Add & Norm]        × N               │
└────────────────────────────────────────┘
      │
      │ encoder memory (B, src_len, d_model)
      │                    ↓
TARGET SEQUENCE            │ (fixed; reused by every decoder layer)
      │                    │
      ▼                    │
[Target Embedding + PE]    │
      │                    │
      ▼                    │
┌─────────────────────────────────────────────────────┐
│                DECODER STACK (N=6)                  │
│                                                     │
│  [Masked Multi-Head Self-Attention]                 │
│  [Add & Norm]                                       │
│                                                     │
│  [Cross-Attention: Q=decoder, K=V=encoder memory]   │
│  [Add & Norm]                                       │
│                                                     │
│  [FFN]                                              │
│  [Add & Norm]                   × N                 │
└─────────────────────────────────────────────────────┘
      │
      ▼
[Linear → vocab_size]
[Softmax]
      │
      ▼
OUTPUT PROBABILITIES
```

---

## Sub-layer 1: Masked Multi-Head Self-Attention

Same as the GPT decoder: causal mask prevents position i from seeing positions j > i. Maintains the auto-regressive property during training. Allows the decoder to track what it has already generated.

```
h1 = LayerNorm(tgt + MaskedSelfAttn(tgt, tgt, tgt, causal_mask))
```

---

## Sub-layer 2: Cross-Attention (Encoder-Decoder Attention)

The critical linking mechanism. Queries come from the decoder; Keys and Values come from the encoder output:

```
Q = decoder_hidden · W^Q       (from current decoder state)
K = encoder_output · W^K       (from encoder — fixed throughout decoding)
V = encoder_output · W^V       (from encoder — fixed throughout decoding)

CrossAttn = softmax( QKᵀ / √d_k ) · V
```

**No causal mask** — the decoder can freely attend to any position in the source sequence, both before and after the currently focused position. In translation, the decoder generating the end of the French sentence should still see the beginning of the English sentence.

```
h2 = LayerNorm(h1 + CrossAttn(Q=h1, K=enc_mem, V=enc_mem))
```

This direct connection between every decoder step and every source token avoids the "bottleneck" problem of RNN seq2seq (where all source meaning had to be compressed into one vector).

---

## Sub-layer 3: Feed-Forward Network

Identical to encoder and GPT decoder FFN:

```
out = LayerNorm(h2 + FFN(h2))
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

---

## Full Decoder Layer Data Flow

```
Input: tgt (B, T_tgt, d_model), enc_mem (B, T_src, d_model)

Step 1 (Masked Self-Attn):
  h1 = LayerNorm(tgt + MaskedSelfAttn(tgt, tgt, tgt, causal_mask))

Step 2 (Cross-Attn):
  h2 = LayerNorm(h1  + CrossAttn(h1, enc_mem, enc_mem, src_padding_mask))

Step 3 (FFN):
  out = LayerNorm(h2  + FFN(h2))
```

---

## Encoder Memory Reuse

The encoder runs **once** on the source sequence and produces `enc_mem`. This matrix is passed to every decoder layer, at every generation step. The encoder's computation is not repeated during autoregressive decoding — this is architecturally efficient.

---

## Teacher Forcing During Training

During training, the decoder receives the **correct** previous target tokens at every position (not its own predictions). This allows parallel training across all positions via the causal mask.

Downside: **exposure bias** — training always sees correct inputs; inference sees potentially wrong ones. Errors compound during generation.

---

## Beam Search Decoding

Used for structured tasks (translation, summarization). Maintains B candidate sequences ("beams") at each step, extending each with every vocabulary token and keeping the top B by cumulative log-probability.

```
score(Y) = (1 / |Y|^α) · Σ_i log P(y_i | y_{<i}, X)
```

α ≈ 0.6–0.7 is a length penalty. Without it, beam search favors short sequences (they have fewer terms to multiply, so higher per-token probability is easier to maintain).

Typical beam size: B = 4 or 5. Larger beams give better quality but more compute.

---

## When to Use Each Architecture

| Task | Best architecture | Why |
|------|------------------|-----|
| Text classification / NER | Encoder only (BERT) | Bidirectional context; no need to generate |
| Open-ended text generation | Decoder only (GPT) | Auto-regressive; scales well |
| Translation | Encoder-decoder (T5, MarianMT) | Cross-attention; efficient for domain-specific data |
| Summarization | Encoder-decoder | Clear source/target separation |
| Code generation | Decoder only | Generation is the primary task |
| Question answering from passage | Either; BERT for extractive, T5 for generative | |
