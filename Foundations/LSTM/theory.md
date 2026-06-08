# Long Short-Term Memory (LSTM)

## Origin

Introduced by Hochreiter & Schmidhuber (1997) specifically to fix the vanishing gradient problem. The key insight: standard RNNs overwrite their hidden state at every step. If instead there is a separate "cell state" c_t that flows forward with minimal modification, gradients can propagate backward without shrinking.

---

## Architecture: Two States

LSTMs maintain two vectors per time step:
- **h_t** — hidden state (short-term, used as output)
- **c_t** — cell state (long-term memory conveyor belt)

The cell state is modified only via gated addition and multiplication, which gives gradients a nearly direct path backward through time.

---

## The Three Gates

Every gate uses a sigmoid activation σ, which outputs values in [0, 1]. Think of each gate as a learned "valve."

### 1. Forget Gate
Decides what fraction of the old cell state to keep.

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

f_t ≈ 0 → forget everything  
f_t ≈ 1 → keep everything

### 2. Input Gate + Candidate Memory
Decides what new information to write into the cell state.

```
i_t   = σ(W_i · [h_{t-1}, x_t] + b_i)       # how much to write
C̃_t  = tanh(W_C · [h_{t-1}, x_t] + b_C)    # what to write
```

### 3. Cell State Update

```
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ C̃_t
        ↑ keep old      ↑ add new
```

This additive update is what prevents vanishing gradients — the gradient of c_t with respect to c_{t-1} is simply f_t (not a product of weight matrices and saturating nonlinearities).

### 4. Output Gate
Determines how much of the cell state to expose as the hidden state.

```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

---

## Full LSTM Equations

```
f_t  = σ(W_f  · [h_{t-1}, x_t] + b_f)    # forget gate
i_t  = σ(W_i  · [h_{t-1}, x_t] + b_i)    # input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)   # candidate values
o_t  = σ(W_o  · [h_{t-1}, x_t] + b_o)    # output gate

c_t  = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t        # new cell state
h_t  = o_t ⊙ tanh(c_t)                    # new hidden state
```

---

## GRU: Simplified Variant

Cho et al. (2014) proposed the Gated Recurrent Unit — two gates instead of three, no separate cell state:

```
z_t = σ(W_z · [h_{t-1}, x_t])       # update gate (replaces forget+input)
r_t = σ(W_r · [h_{t-1}, x_t])       # reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

GRUs use fewer parameters and often match LSTM performance on shorter sequences.

---

## LSTM Applications (Pre-2017)

- **Machine translation**: encoder LSTM + decoder LSTM + Bahdanau attention
- **Speech recognition**: bidirectional LSTM (BiLSTM) reads audio context from both directions (DeepSpeech)
- **Time-series**: anomaly detection, stock price modeling, sensor data
- **Language modeling**: character-level and word-level LMs for text generation
- **Music generation**: sequential note prediction

---

## Why LSTMs Still Matter

Even after Transformers, LSTMs remain useful when:
- Sequence length is moderate (up to a few hundred steps)
- Compute budget is limited (LSTMs are cheaper per step than Transformers)
- Streaming / online prediction is needed (Transformers need the full sequence upfront)
- Embedded or edge deployment (small model size matters)

---

## LSTM vs Vanilla RNN vs Transformer

| Property | RNN | LSTM | Transformer |
|----------|-----|------|-------------|
| Long-range dependency | Fails | Good | Excellent |
| Gradient stability | Unstable | Stable | Stable |
| Parallelizable | No | No | Yes |
| Parameters | Low | ~4× RNN | Very high |
| Memory cost | O(d) per step | O(d) per step | O(n²) attention matrix |
