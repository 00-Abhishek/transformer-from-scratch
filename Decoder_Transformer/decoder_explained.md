# Decoder Code — Line-by-Line Explanation

## gpt_decoder.py (PyTorch)

### DecoderLayer._causal_mask

```python
torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
```

`torch.tril` returns a lower-triangular matrix:
```
T=4:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

`unsqueeze(0).unsqueeze(0)` → shape (1, 1, T, T).  
Broadcasts over (B, h, T, T) attention scores automatically.

In the attention function, positions with mask=0 get -1e9 → softmax → 0.  
Position i cannot receive any gradient signal from position j > i.

---

### Learned Positional Embeddings

```python
self.pos_embed = nn.Embedding(max_len, d_model)
pos = torch.arange(T, device=tokens.device).unsqueeze(0)
pos_emb = self.pos_embed(pos)
```

`torch.arange(T)` = [0, 1, 2, ..., T-1] — the position indices.  
Each position gets its own learned d_model-dimensional vector.  
These are **not** fixed (unlike sinusoidal PE) — they are updated by gradient descent.

---

### Weight Tying

```python
self.lm_head.weight = self.token_embed.weight
```

This sets the LM head's weight matrix to literally be the same object as the token embedding matrix. When the optimizer updates one, it updates both. This enforces that the prediction space and embedding space are consistent — and saves ~25M parameters for a 50k vocabulary with 512-dim model.

---

### compute_loss — Teacher Forcing

```python
src    = tokens[:, :-1]   # [BOS, w1, w2, ..., w_{n-1}]
target = tokens[:, 1:]    # [w1,  w2, ..., w_n,  EOS]
```

The model takes the first T tokens and predicts the next T tokens at every position simultaneously. Position i predicts token i+1. The causal mask ensures position i only sees positions 0..i during this parallel computation.

This is far more efficient than running the model T times per sequence.

---

### generate — Top-p (Nucleus) Filtering

```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
remove_mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
sorted_logits[remove_mask] = -1e9
logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
```

Step by step:
1. Sort tokens by probability (highest first)
2. Compute cumulative probabilities along the sorted order
3. Find the first position where cumulative prob exceeds p
4. Zero out all tokens after that position
5. `.scatter` maps back to original (unsorted) token ordering

The key subtlety: `cum_probs - F.softmax(sorted_logits)` computes cumulative prob **excluding the current token**. This ensures the nucleus always includes at least the top token.

---

## gpt_pure_python.py (No Libraries)

### Value.backward() — Autograd Engine

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children: build_topo(child)
        topo.append(v)
```

This is a depth-first topological sort of the computation graph. After appending all children, the current node is appended — so `topo` is in reverse topological order.

```python
self.grad = 1.0
for v in reversed(topo):
    for child, lg in zip(v._children, v._local_grads):
        child.grad += lg * v.grad
```

Starting from the loss (grad=1.0), gradient flows backward using the chain rule. `lg` is the local gradient (∂v/∂child), stored during the forward pass. `v.grad` is the upstream gradient. Their product = child's gradient contribution from this path. Multiple paths accumulate with `+=`.

### KV Cache

```python
keys[li].append(k)
values[li].append(v)
```

This is a token-by-token KV cache. At inference, each call to `gpt()` appends the new token's K and V to the cache. The attention over all previous tokens is computed without recomputing their K/V. This is the same technique used in production inference for GPT models.

### rmsnorm vs LayerNorm

RMSNorm: `x = x * (1 / sqrt(mean(x²) + ε))`  
LayerNorm: `x = (x - mean(x)) / (std(x) + ε) * γ + β`

RMSNorm omits the mean subtraction and the learned scale/shift (γ, β). Simpler and faster. Used in LLaMA, Mistral, and other modern models.

### Softmax with Temperature

```python
scaled = [l * Value(1.0 / temperature) for l in logits]
```

Dividing logits by temperature T is equivalent to raising probabilities to the power 1/T before normalizing. Low T → sharper, more deterministic. High T → flatter, more random.
