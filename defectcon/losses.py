"""Training objectives (Sections 4.4 and 4.5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss, L_out form (Khosla et al., 2020).

    Single scalar temperature; self-comparisons removed from numerator and
    denominator; row maximum subtracted for numerical stability; anchors
    without a positive are excluded from the mean.
    """

    def __init__(self, temperature: float, base_temperature: float | None = None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        features = F.normalize(features, dim=1)
        n = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        pos_per_anchor = mask.sum(1)
        valid = pos_per_anchor > 0
        if valid.sum() == 0:
            return features.sum() * 0.0
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / pos_per_anchor[valid]
        return (-(self.temperature / self.base_temperature) * mean_log_prob_pos).mean()


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): cross-entropy scaled by ``(1 - p_t)^gamma``."""

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()
