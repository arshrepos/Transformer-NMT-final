"""Training utilities: label smoothing, LR scheduler, checkpoint save/load."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: (batch * seq_len, vocab_size) or (batch, seq_len, vocab_size)
        # target: (batch * seq_len) or (batch, seq_len)
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.sum(dim=-1) / self.vocab_size

        # Mask padding
        non_pad = target != self.pad_idx
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.masked_select(non_pad).mean()
        return loss


class WarmupDecayScheduler:
    """Linear warmup then inverse-sqrt decay, from 'Attention Is All You Need'."""

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.d_model ** (-0.5) * min(
            self._step ** (-0.5),
            self._step * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, epoch, train_config, path, scheduler=None):
    """Save checkpoint with full model_config for reproducible loading."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "model_config": model.get_config(),
        "train_config": train_config,
    }
    if scheduler is not None:
        checkpoint["scheduler_step"] = scheduler._step
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, device="cpu"):
    """Load checkpoint and return all saved data."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint
