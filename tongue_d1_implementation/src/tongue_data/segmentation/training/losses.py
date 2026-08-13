"""BCE + soft Dice loss（logits 输入，不做 threshold）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """DiceLoss = 1 - soft Dice；使用 continuous probability，禁止 threshold。"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits/target shape mismatch: logits={tuple(logits.shape)} "
                f"target={tuple(target.shape)}"
            )
        probability = torch.sigmoid(logits)
        probability_flat = probability.reshape(probability.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1).float()
        intersection = (probability_flat * target_flat).sum(dim=1)
        denominator = probability_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """total = bce_weight * BCEWithLogits + dice_weight * SoftDiceLoss。"""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits/target shape mismatch: logits={tuple(logits.shape)} "
                f"target={tuple(target.shape)}"
            )
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"expected logits [B,1,H,W], got {tuple(logits.shape)}")
        bce_term = self.bce(logits, target.float())
        dice_term = self.dice(logits, target)
        return self.bce_weight * bce_term + self.dice_weight * dice_term


def build_loss(loss_cfg: dict) -> BCEDiceLoss:
    return BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        smooth=float(loss_cfg.get("smooth", 1e-6)),
    )
