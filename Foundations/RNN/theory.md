# Recurrent Neural Networks (RNN)

## What RNNs Are

RNNs process sequential data by maintaining an internal "hidden state" — a running summary of everything seen so far. At each time step t, the model takes the current input x_t and the previous hidden state h_{t-1}, combines them, and produces a new hidden state h_t.

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

This means the model's "memory" of past inputs is always compressed into a fixed-size vector h_t — regardless of how long the sequence is.

---

## Why RNNs Existed

Before RNNs, standard feedforward networks could not handle sequences of variable length. Each input had to be a fixed-size vector. RNNs broke that constraint by introducing the concept of state: the same network weights are applied repeatedly across time, with the hidden state carrying information forward.

This made RNNs the dominant architecture for:
- Language modeling
- Machine translation (with encoder-decoder RNNs)
- Speech recognition
- Time-series forecasting

---

## The Vanishing Gradient Problem

RNNs are trained via **Backpropagation Through Time (BPTT)** — unrolling the network across all time steps and applying standard backprop. The gradient of the loss with respect to an early hidden state h_t requires multiplying gradients through all later time steps:

```
∂L/∂h_t = ∂L/∂h_T · (∂h_T/∂h_{T-1}) · ... · (∂h_{t+1}/∂h_t)
```

Each term is a product of the weight matrix W_hh and the derivative of tanh. If the singular values of W_hh are < 1, this product shrinks exponentially → **vanishing gradient**: the model cannot update weights for early inputs, so it forgets long-range dependencies.

If singular values > 1 → **exploding gradient**: updates diverge.

---

## The Parallelism Problem

RNNs process token t only after token t-1. The hidden state h_t depends on h_{t-1}, which depends on h_{t-2}, etc. This **sequential dependency** means:

- Training cannot be parallelized across time steps
- GPU/TPU utilization is poor for long sequences
- Training scales linearly with sequence length

This was a hard architectural limit, not a hyperparameter to tune.

---

## RNN Variants

| Variant | Fix | Limitation |
|---------|-----|-----------|
| Vanilla RNN | Baseline | Vanishing gradient |
| LSTM | Gating mechanism | Still sequential |
| GRU | Simpler gating (2 gates) | Still sequential |
| Bidirectional RNN | Reads sequence both ways | Still sequential; no future tokens at decode time |

---

## The Transition to Attention

Bahdanau et al. (2015) first introduced **attention as an add-on to RNN decoders**. Instead of compressing the entire source into a single vector, the decoder at each step could look back at all encoder hidden states and compute a weighted average. This solved the bottleneck problem but the underlying RNN computation remained sequential.

The Transformer (2017) took the next step: remove recurrence entirely, and make attention the primary computation.

---

## RNN Equation Summary

```
Forward pass at time t:
  h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
  y_t = softmax(W_hy · h_t + b_y)

BPTT gradient (simplified):
  ∂L/∂h_t ∝ Π_{k=t}^{T} W_hh^T · diag(1 - h_k²)
                          ↑ this product vanishes or explodes
```

---

## Summary Table

| Property | Vanilla RNN |
|----------|-------------|
| Core operation | Sequential recurrence |
| Memory type | Single hidden state h_t |
| Gradient stability | Unstable (vanishing/exploding) |
| Parallelizable | No |
| Long-range dependency | Fails beyond ~10–20 steps |
| Complexity | O(n · d²) time, O(n) memory |
