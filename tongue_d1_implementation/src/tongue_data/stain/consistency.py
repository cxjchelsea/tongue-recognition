"""D4-C.1-B：source/external consistency loss（禁止 pseudo-label / entropy min）。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_bce_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """BCEWithLogits；labels 为 gold stain。"""
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def probability_consistency_loss(
    logit_student: torch.Tensor,
    logit_teacher: torch.Tensor,
    *,
    stop_gradient_teacher: bool = True,
) -> torch.Tensor:
    """
    MSE(sigmoid(student), target_prob)。
    teacher 可 detach；不是 pseudo stain label。
    """
    if logit_student.ndim == 2 and logit_student.shape[1] == 1:
        logit_student = logit_student[:, 0]
    if logit_teacher.ndim == 2 and logit_teacher.shape[1] == 1:
        logit_teacher = logit_teacher[:, 0]
    teacher = logit_teacher.detach() if stop_gradient_teacher else logit_teacher
    p_student = torch.sigmoid(logit_student.float())
    p_teacher = torch.sigmoid(teacher.float())
    return F.mse_loss(p_student, p_teacher)


def source_supervised_two_view_loss(
    logit_weak: torch.Tensor,
    logit_style: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """两个 view 共享同一 gold label。"""
    loss_weak = supervised_bce_from_logits(logit_weak, labels)
    loss_style = supervised_bce_from_logits(logit_style, labels)
    return 0.5 * (loss_weak + loss_style)


def consistency_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """线性 warmup：epoch 从 1 开始。"""
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch / float(warmup_epochs))))


def decompose_total_loss(
    *,
    supervised: torch.Tensor,
    source_consistency: torch.Tensor,
    external_consistency: torch.Tensor,
    supervised_weight: float,
    source_consistency_weight: float,
    external_consistency_weight: float,
    warmup: float,
) -> dict[str, torch.Tensor]:
    """总损失分解；无 entropy term。"""
    total = (
        float(supervised_weight) * supervised
        + float(source_consistency_weight) * warmup * source_consistency
        + float(external_consistency_weight) * warmup * external_consistency
    )
    return {
        "total": total,
        "supervised": supervised,
        "source_consistency": source_consistency,
        "external_consistency": external_consistency,
        "warmup": torch.tensor(warmup),
    }
