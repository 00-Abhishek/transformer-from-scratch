<div align="center">

<h1>🔮 Transformer from Scratch</h1>

<p><em>The complete architectural journey — from broken RNNs to GPT-style generation.<br>Theory, mathematics, and working code. All in one place.</em></p>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-Educational-8A2BE2?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Ayush-2703/transformer-from-scratch?style=for-the-badge&color=FFD700)](https://github.com/Ayush-2703/transformer-from-scratch/stargazers)

<br/>
---

## 🧭 The Story This Repo Tells

Every modern LLM — GPT-4, Claude, Gemini — runs on one architecture. But the Transformer didn't appear from nowhere.
This repo walks the full intellectual journey:

```
  1990                    1997                    2017
    │                       │                       │
┌───────┐             ┌───────────┐            ┌─────────────┐
│  RNN  │──vanishing──▶   LSTM    │──slow ────▶ TRANSFORMER │
│       │  gradients  │           │  sequential│             │
└───────┘             └───────────┘            └─────────────┘
 Hidden                Cell state +             Self-Attention
  state                   Gates               processes ALL
 only                  (forget/input/          tokens at once
                          output)
```

Each module in this repo is a chapter in that story — with the theory explained, the math derived, and the code written from scratch.

---

## 📂 Repository Structure

```
transformer-from-scratch/
│
├── 📁 Encoder Transformer/        ← BERT-style · bidirectional · understanding tasks
│   ├── Theory & Math              
│   ├── PyTorch Implementation     ← ScaledDotProductAttention → MultiHeadAttention
│   └── Code Walkthrough               → PositionwiseFFN → EncoderLayer → Encoder
│
├── 📁 Decoder Transformer/        ← GPT-style · causal · generation tasks  
│   ├── Theory & Math              
│   ├── PyTorch Implementation     ← Causal mask → DecoderLayer → GPTDecoder
│   ├── Pure Python Implementation ← Zero dependencies · built on basic math only
│   └── Code Walkthrough               
│
└── 📁 Encoder-Decoder Transformer/  ← T5/BART-style · seq2seq · translation tasks
    ├── Theory & Math              
    ├── PyTorch Implementation     ← FullDecoderLayer (cross-attn) → EncoderDecoder
    └── Code Walkthrough               
```

---

## 🏗️ Architecture Deep Dive

### Module 1 — Encoder Transformer
*For tasks that require understanding: classification, NER, semantic search*

```
Input Tokens
    │
    ▼
[Embedding × √d_model]  +  [Sinusoidal Positional Encoding]
    │
    ▼  ┌─────────────────────────────────── × N=6 ──┐
    │  │                                            │
    │  │  ┌──────────────────────────────────┐      │
    │  │  │   Multi-Head Self-Attention      │      │
    │  │  │   Q=K=V=x  |  h=8  |  d_k=64     │      │
    │  │  └──────────────────────────────────┘      │
    │  │          │  Add & LayerNorm                │
    │  │  ┌──────────────────────────────────┐      │
    │  │  │   Position-wise FFN              │      │
    │  │  │   d_model=512 → 2048 → 512       │      │
    │  │  └──────────────────────────────────┘      │
    │  │          │  Add & LayerNorm                │
    │  └──────────┼─────────────────────────────────┘
    ▼
Contextual Representations  (B, seq_len, 512)
```

---

### Module 2 — Decoder Transformer (GPT-style)
*For generation: text completion, chat, code synthesis*

```
Token IDs  +  Position IDs
    │               │
    ▼               ▼
[Token Embed]  [Pos Embed]  ← both LEARNED (not sinusoidal)
    └──────┬────────┘
           ▼
  ┌──────────────────────────── × N=6 ─┐
  │                                    │
  │  ┌─────────────────────────────┐   │
  │  │  Masked Multi-Head Attn     │   │
  │  │  Causal mask: can only      │   │
  │  │  attend to past positions   │   │
  │  └─────────────────────────────┘   │
  │          │  Add & LayerNorm        │
  │  ┌─────────────────────────────┐   │
  │  │  Position-wise FFN          │   │
  │  └─────────────────────────────┘   │
  │          │  Add & LayerNorm        │
  └──────────┼─────────────────────────┘
             ▼
     [LM Head: d_model → vocab]
             ▼
    Logits / Temperature / Top-p
             ▼
        Next Token  →  Repeat
```

> Also includes a **pure Python implementation** — no PyTorch, no NumPy. Just floats, lists, and calculus.
> The same architecture. 100 lines. Everything visible.

---

### Module 3 — Encoder-Decoder Transformer
*For seq2seq: translation, summarisation, data-to-text*

```
Source Sequence                  Target Sequence (teacher-forced)
      │                                   │
      ▼                                   ▼
 ENCODER STACK                      DECODER STACK
 (bidirectional)                    (causal, 3 sub-layers)
      │                                   │
      │  encoder_memory         ┌─────────┴───────────┐
      └─────────────────────────▶  Cross-Attention   │
                                 │  Q ← decoder       │
                                 │  K,V ← enc memory  │
                                 │  no causal mask    │
                                 └────────────────────┘
                                          │
                                   [Output Projection]
                                          │
                                   Token Probabilities
```

**The key insight:** encoder runs *once*, decoder consults it at *every generation step*.
This is why translation is not just a decoder task.

---

## 📐 Core Mathematics

#### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```
> Scaling by `√d_k` prevents softmax saturation when `d_k` is large

#### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · Wᴼ
      head_i     = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)
```
> h=8 heads each see a different 64-dim projection — different aspects of meaning

#### Sinusoidal Positional Encoding
```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```
> Deterministic, not learned — generalises to unseen sequence lengths

#### Feed-Forward Network
```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
         d_model=512 → d_ff=2048 → d_model=512
```

#### Warmup Learning Rate Schedule
```
lr = d_model⁻⁰·⁵ · min( step⁻⁰·⁵,  step · warmup_steps⁻¹·⁵ )
```
> Increases linearly for 4000 steps, then decays — stabilises early training

---

## 📊 Comparison Tables

### RNN vs LSTM vs Transformer

| Feature | RNN | LSTM | Transformer |
|---|---|---|---|
| **Year** | 1990 | 1997 | 2017 |
| **Core op** | `hₜ = f(hₜ₋₁, xₜ)` | 3-gate cell state | Self-attention |
| **Parallelisable** | ✗ Sequential | ✗ Sequential | ✓ Fully parallel |
| **Long-range deps** | ✗ Vanishes | ⚠ Partial | ✓ Direct O(1) path |
| **Training cost** | O(n·d²) | O(n·d²) | O(n²·d) |
| **Gradient stability** | ✗ Explodes/vanishes | ✓ Gated | ✓ LayerNorm + residuals |
| **Parameters** | Low | Medium | Huge (65M–175B+) |
| **Key models** | Elman RNN | Deep Speech, seq2seq | GPT, BERT, T5, Claude |

---

### Encoder vs Decoder vs Encoder-Decoder

| Property | Encoder Only | Decoder Only | Encoder-Decoder |
|---|---|---|---|
| **Attention direction** | Bidirectional | Causal (left-only) | Enc: bi · Dec: causal |
| **Cross-attention** | ✗ | ✗ | ✓ |
| **Sub-layers / layer** | 2 | 2 | 2 (enc) · 3 (dec) |
| **Inference mode** | Single forward pass | Autoregressive | Encode once, decode autoregressively |
| **Pre-training obj** | Masked LM (MLM) | Causal LM (CLM) | Span corruption / denoising |
| **Key models** | BERT, RoBERTa | GPT-2/3/4, LLaMA, **Claude** | T5, BART, mBART |
| **Best for** | Classification, NER, QA | Generation, chat, code | Translation, summarisation |

---

## ⚙️ Original Paper Hyperparameters

| Hyperparameter | Symbol | Base Model | Large Model |
|---|---|---|---|
| Model dimension | `d_model` | 512 | 1024 |
| Feed-forward dim | `d_ff` | 2048 | 4096 |
| Attention heads | `h` | 8 | 16 |
| Dim per head | `d_k = d_v` | 64 | 64 |
| Encoder/Decoder layers | `N` | 6 | 6 |
| Dropout rate | `p` | 0.1 | 0.3 |
| Label smoothing | `ε` | 0.1 | 0.1 |
| Warmup steps | — | 4,000 | 4,000 |
| Parameters | — | ~65M | ~213M |
| Training time (8×P100) | — | 12 hours | 3.5 days |

---

## 🗺️ Reading Guide

> Different goals → different paths through the repo.

| If you want to... | Start here |
|---|---|
| Understand *why* Transformers replaced RNNs | `Encoder Transformer/Theory` → RNN/LSTM section |
| Implement self-attention from scratch | `Encoder Transformer/Code` |
| Build a GPT-style model | `Decoder Transformer/Code` |
| See the architecture in pure Python (no libraries) | `Decoder Transformer/Pure Python` |
| Understand cross-attention for translation | `Encoder-Decoder Transformer/Theory` |
| Know how to train a Transformer properly | Adam + warmup schedule section in `Encoder-Decoder` |
| Compare all three architectures | Comparison tables above |
| Understand current limitations (hallucination, O(n²)) | `Encoder-Decoder Transformer/Theory` — Limitations |

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/Ayush-2703/transformer-from-scratch.git
cd transformer-from-scratch

# Install dependencies
pip install torch numpy

# Run encoder (BERT-style)
cd "Encoder Transformer"
python encoder.py

# Run GPT-style decoder
cd "../Decoder Transformer"
python decoder.py          # PyTorch version
python gpt_pure_python.py  # Pure Python — zero dependencies

# Run full encoder-decoder (translation-style)
cd "../Encoder-Decoder Transformer"
python encoder_decoder.py
```

---

## 📚 References

| Paper | Authors | Year |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 |
| [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) | Hochreiter & Schmidhuber | 1997 |
| [BERT](https://arxiv.org/abs/1810.04805) | Devlin et al. | 2019 |
| [GPT-3](https://arxiv.org/abs/2005.14165) | Brown et al. | 2020 |
| [RoBERTa](https://arxiv.org/abs/1907.11692) | Liu et al. | 2019 |
| [T5](https://arxiv.org/abs/1910.10683) | Raffel et al. | 2020 |

---


<div align="left">

## 🙏 Acknowledgements

- **Dr. Shikha Singh** — guidance and supervision
- **Vaswani et al.** — the paper that changed everything
- **Andrej Karpathy** — educational philosophy that inspired the pure-Python implementation
- Open-source community — PyTorch, Hugging Face, NumPy

---

<div align="center">

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703) & [Abhishek Pal](https://github.com/00-Abhishek)**

Amity University Uttar Pradesh · B.Tech Artificial Intelligence · 2026

---

⭐ **Star this repo if it helped you understand the architecture that powers modern AI.**

</div>
