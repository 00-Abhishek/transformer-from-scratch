# Training Code — Explanation

## WarmupInvSqrtScheduler

```python
lr = (self.d_model ** -0.5) * min(
    self.step_num ** -0.5,
    self.step_num * (self.warmup_steps ** -1.5),
)
```

The `min` picks the smaller of two terms:

**During warmup** (`step ≤ warmup_steps`): the second term `step * warmup_steps^(-1.5)` is smaller — it grows linearly from 0.

**After warmup** (`step > warmup_steps`): the first term `step^(-0.5)` is smaller — it decays as inverse square root.

The crossover happens exactly at `step = warmup_steps`:
```
step^(-0.5) = step * warmup_steps^(-1.5)
step^(-0.5) = step * warmup_steps^(-1.5)
1 = step^(3/2) / warmup_steps^(3/2)
step = warmup_steps   ✓
```

---

## LabelSmoothingLoss — scatter_ Explained

```python
smooth_dist = torch.full_like(log_probs, self.epsilon / (self.vocab_size - 1))
smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
```

`torch.full_like` fills the entire matrix with `ε/(V-1)`.

`scatter_(1, idx, value)` writes `value` at positions specified by `idx` along dimension 1.

`targets.unsqueeze(1)` has shape (N, 1) — one target class per row.

After scatter_: each row has `1-ε` at the true class index, `ε/(V-1)` everywhere else.

Sum check: `(1-ε) + (V-1) * ε/(V-1) = (1-ε) + ε = 1.0` ✓

---

## Gradient Accumulation — Why Divide Loss?

```python
loss = criterion(...) / accumulation_steps
scaler.scale(loss).backward()
```

Each `backward()` adds gradients to the `param.grad` buffer (it doesn't zero them). After `accumulation_steps` backward calls, `param.grad` holds the sum of gradients from `accumulation_steps` batches.

Without the `/ accumulation_steps`, this sum would be `accumulation_steps` times larger than a single large batch. Dividing each loss by `accumulation_steps` makes the accumulated gradient equivalent to averaging over all batches — same as computing one large batch.

---

## Mixed Precision (AMP) — Why FP16 for Forward, FP32 for Optimizer?

FP16 has range ~[6×10⁻⁵, 65504] and only 10 bits of mantissa (3 decimal digits). This is enough for activations and forward/backward pass computations.

FP32 has range ~[10⁻³⁸, 10³⁸] and 23 bits of mantissa (7 decimal digits). This is needed for optimizer state (momentum, variance) which accumulates small incremental updates over many steps.

```python
scaler.unscale_(optimizer)   # convert gradients from FP16 → FP32 before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
scaler.step(optimizer)       # optimizer step in FP32
scaler.update()              # adjust loss scale factor for next step
```

The `GradScaler` multiplies the loss by a large scale factor before backward, keeping FP16 gradients from underflowing to zero. It then divides by the scale factor before the optimizer step.

---

## Perplexity

```python
perplexity = math.exp(avg_loss)
```

Perplexity = 2^H, where H is cross-entropy in bits. In nats (natural log): PP = e^H.

Interpretation: a perplexity of 100 means the model is as uncertain as if it were choosing uniformly among 100 options at each step. Lower is better.

Typical values:
- Untrained model on V=30K vocabulary: PP ≈ 30000
- Decent language model: PP ≈ 50–200
- State-of-the-art LLM on easy text: PP ≈ 10–30

---

## Checkpoint Save/Load — Why Save Scheduler Step?

```python
scheduler.step_num = ckpt["scheduler_step"]
```

If you resume training from a checkpoint without restoring the scheduler step, the LR would restart from the warmup phase. This means training would be unstable (large LR at a later step) or the warmup would be unnecessarily repeated. Restoring `step_num` ensures the LR schedule continues exactly where it left off.
