# Transformer Architecture — Complete Reference

> **Authors:** Ayush Kumar Singh & Abhishek Pal  

This repo is the one-stop reference for everything Transformer — theory, maths, and runnable Python. No other source needed.

---

## What's Inside

```
transformer-complete-guide/
├── Foundations/                   ← Pre-Transformer history (RNN, LSTM, embeddings, attention basics)
│   ├── RNN/
│   ├── LSTM/
│   ├── Input_Embedding/
│   ├── Positional_Encoding/
│   └── Attention_Mechanism/
│
├── Encoder_Transformer/           ← BERT-style encoder-only models
│   ├── Theory_and_Math/
│   ├── Code/                      ← Full PyTorch implementation
│   └── Code_Explanation/
│
├── Decoder_Transformer/           ← GPT-style decoder-only models
│   ├── Theory_and_Math/
│   ├── Code/                      ← GPT from scratch (PyTorch + pure Python)
│   └── Code_Explanation/
│
├── Encoder_Decoder_Transformer/   ← T5/BART-style seq2seq models
│   ├── Theory_and_Math/
│   ├── Code/                      ← Full enc-dec with cross-attention
│   └── Code_Explanation/
│
├── Training/                      ← Adam, warmup schedule, dropout, label smoothing
│   ├── Theory/
│   ├── Code/
│   └── Code_Explanation/
│
├── Comparison_Tables/             ← RNN vs LSTM vs Transformer; enc vs dec vs enc-dec
├── Architecture_Diagrams/         ← ASCII + Mermaid diagrams of every component
├── Glossary/                      ← Every term defined
├── Limitations/                   ← Hallucination, quadratic cost, bias
├── Applications/                  ← NLP, vision, code gen, science
└── References/                    ← Full bibliography
```

---

## Quick-Start Reading Order

| Goal | Path |
|------|------|
| Understand why Transformers exist | `Foundations/RNN` → `Foundations/LSTM` |
| Core mechanism | `Foundations/Attention_Mechanism` → `Foundations/Positional_Encoding` |
| Build an encoder | `Encoder_Transformer/Theory_and_Math` → `Encoder_Transformer/Code` |
| Build a GPT | `Decoder_Transformer/Theory_and_Math` → `Decoder_Transformer/Code` |
| Build seq2seq | `Encoder_Decoder_Transformer/Theory_and_Math` → `Encoder_Decoder_Transformer/Code` |
| Train properly | `Training/Theory` → `Training/Code` |
| Compare architectures | `Comparison_Tables/` |
| Know the limits | `Limitations/` |

---

## Architecture at a Glance

```
INPUT
  │
  ▼
[Token Embedding]  +  [Positional Encoding]
  │
  ▼
┌─────────────────────────────────────┐
│         ENCODER STACK (N=6)         │
│  ┌─────────────────────────────┐    │
│  │  Multi-Head Self-Attention  │    │
│  │  Add & Norm (residual)      │    │
│  │  Feed-Forward Network       │    │
│  │  Add & Norm (residual)      │    │
│  └─────────────────────────────┘    │
│         × N layers                  │
└─────────────────────────────────────┘
  │
  ▼  (encoder memory)
┌─────────────────────────────────────┐
│         DECODER STACK (N=6)         │
│  ┌─────────────────────────────┐    │
│  │  Masked Multi-Head Attn     │    │
│  │  Add & Norm                 │    │
│  │  Cross-Attention (enc mem)  │    │
│  │  Add & Norm                 │    │
│  │  Feed-Forward Network       │    │
│  │  Add & Norm                 │    │
│  └─────────────────────────────┘    │
│         × N layers                  │
└─────────────────────────────────────┘
  │
  ▼
[Linear Projection → Vocab Size]
[Softmax]
  │
  ▼
OUTPUT TOKEN PROBABILITIES
```

---

## Key Hyperparameters (Original Paper)

| Parameter | Base | Large |
|-----------|------|-------|
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| Heads (h) | 8 | 16 |
| d_k = d_v | 64 | 64 |
| Layers N | 6 | 6 |
| Dropout | 0.1 | 0.3 |
| Warmup steps | 4000 | 4000 |
| Parameters | ~65M | ~213M |

---

## Core Equations

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Feed-Forward Network:**
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Learning Rate Schedule:**
```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

---

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997
- Devlin et al., "BERT", NAACL 2019
- Brown et al., "GPT-3", NeurIPS 2020
- Raffel et al., "T5", JMLR 2020

See `References/bibliography.md` for full list.


## 🙏 Acknowledgements

- **Dr. Shikha Singh** — guidance and supervision throughouts
- **Vaswani et al. (2017)** — foundational *"Attention Is All You Need"* paper
- **Andrej Karpathy** — from-scratch Transformer implementation inspiration
- Open-source community — PyTorch, Hugging Face, NumPy

---

<div align="center">

**Made with ❤️ by Ayush Kumar Singh & Abhishek Pal**  
Amity University Uttar Pradesh · B.Tech Artificial Intelligence · 2026  
⭐ *Star this repo if it helped you!*

</div>
