"""
Training utilities for Transformer models.
Covers: WarmupInvSqrtScheduler, LabelSmoothingLoss,
        full training loop, gradient accumulation, mixed precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import GradScaler, autocast


# ─────────────────────────────────────────────────────────────────
# 1.  WARMUP + INVERSE SQRT LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────

class WarmupInvSqrtScheduler:
    """
    Learning rate schedule from Vaswani et al. (2017).

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Phase 1 (step ≤ warmup_steps): linear increase from 0 to peak lr
    Phase 2 (step > warmup_steps): decreases as 1/√step
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0

    def step(self) -> float:
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5),
        )


# ─────────────────────────────────────────────────────────────────
# 2.  LABEL SMOOTHING LOSS
# ─────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    ε=0.1: true class gets 0.9, remaining 0.1 spread uniformly.
    Padding positions receive 0 gradient.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, epsilon: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.epsilon    = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, vocab_size)
        targets: (N,)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.epsilon / (self.vocab_size - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
            smooth_dist[targets == self.pad_idx] = 0.0   # mask padding

        non_pad = (targets != self.pad_idx).sum().float()
        loss = -(smooth_dist * log_probs).sum() / non_pad
        return loss


# ─────────────────────────────────────────────────────────────────
# 3.  BASIC TRAINING STEP
# ─────────────────────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int = 0,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Single training step for an encoder-decoder Transformer.

    Args:
        src: source token IDs (B, src_len)
        tgt: target token IDs (B, tgt_len + 1)  includes BOS and EOS
    Returns:
        loss value (float)
    """
    model.train()

    tgt_in  = tgt[:, :-1]   # decoder input: BOS + words
    tgt_out = tgt[:, 1:]    # labels: words + EOS

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)   # (B, 1, 1, src_len)

    logits = model(src, tgt_in, src_mask)   # (B, T, vocab_size)

    loss = criterion(
        logits.view(-1, logits.size(-1)),
        tgt_out.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP WITH GRADIENT ACCUMULATION + MIXED PRECISION
# ─────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
    accumulation_steps: int = 4,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Full training epoch.

    Gradient accumulation: gradients are accumulated over `accumulation_steps`
    batches before performing an optimizer step. This simulates a batch size
    accumulation_steps× larger without requiring more GPU memory.

    Mixed precision (AMP): forward pass and loss computed in FP16/BF16,
    optimizer step in FP32. ~2× speedup on modern GPUs with minimal accuracy loss.
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)
    total_loss = 0.0
    step = 0

    optimizer.zero_grad()

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            logits = model(src, tgt_in, src_mask)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            ) / accumulation_steps   # normalize for accumulation

        scaler.scale(loss).backward()

        # Optimizer step every `accumulation_steps` batches
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)   # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────────────────────────
# 5.  VALIDATION / PERPLEXITY EVALUATION
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int = 0,
) -> dict:
    """
    Evaluate model on validation set.
    Returns dict with loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        logits = model(src, tgt_in, src_mask)

        # Use standard cross-entropy for evaluation (not label-smoothed)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
            reduction="sum",
        )

        non_pad = (tgt_out != pad_idx).sum().item()
        total_loss   += loss.item()
        total_tokens += non_pad

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {"loss": avg_loss, "perplexity": perplexity}


# ─────────────────────────────────────────────────────────────────
# 6.  CHECKPOINT SAVE / LOAD
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, path):
    torch.save({
        "step":           step,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_step": scheduler.step_num,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.step_num = ckpt["scheduler_step"]
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────
# 7.  USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal demo (no real dataloader)
    import sys
    sys.path.insert(0, "..")

    from Encoder_Decoder_Transformer.Code.enc_dec_transformer import EncoderDecoderTransformer

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_VOCAB  = 8000
    TGT_VOCAB  = 6000
    D_MODEL    = 256
    PAD_IDX    = 0

    model = EncoderDecoderTransformer(
        SRC_VOCAB, TGT_VOCAB, d_model=D_MODEL, N=3, h=8, d_ff=512
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = WarmupInvSqrtScheduler(optimizer, d_model=D_MODEL, warmup_steps=4000)
    criterion = LabelSmoothingLoss(TGT_VOCAB, pad_idx=PAD_IDX, epsilon=0.1)

    # Fake batch
    src = torch.randint(1, SRC_VOCAB, (4, 20)).to(DEVICE)
    tgt = torch.randint(1, TGT_VOCAB, (4, 16)).to(DEVICE)

    loss = train_step(model, optimizer, scheduler, criterion, src, tgt, PAD_IDX)
    print(f"Training loss: {loss:.4f}")
    print(f"LR after step: {scheduler.get_lr():.8f}")

    # LR schedule visualization
    print("\nLR schedule (first 10000 steps):")
    dummy_opt = torch.optim.Adam([torch.tensor(0.0)], betas=(0.9, 0.98), eps=1e-9)
    sched = WarmupInvSqrtScheduler(dummy_opt, d_model=512, warmup_steps=4000)
    for step in [100, 500, 1000, 2000, 4000, 8000, 10000]:
        while sched.step_num < step:
            sched.step()
        print(f"  Step {step:6d}: lr = {sched.get_lr():.6f}")
