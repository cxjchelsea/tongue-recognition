"""Validation metrics：overall + per-domain。"""
from __future__ import annotations

from collections import defaultdict

import torch


def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """sigmoid + threshold；仅用于 metric，不用于 loss。"""
    probability = torch.sigmoid(logits)
    return (probability >= float(threshold)).to(dtype=logits.dtype)


def batch_dice_iou_precision_recall(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """逐图指标，返回 shape [B] 的 tensor。"""
    if logits.shape != target.shape:
        raise ValueError(
            f"logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    # 强制 float32：AMP float16 在 H*W 求和时可能溢出 → Inf/NaN
    logits = logits.float()
    target = target.float()
    prediction = logits_to_binary(logits, threshold)
    target_binary = (target > 0.5).to(dtype=prediction.dtype)

    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target_binary.reshape(target_binary.shape[0], -1)
    true_positive = (prediction_flat * target_flat).sum(dim=1)
    false_positive = (prediction_flat * (1.0 - target_flat)).sum(dim=1)
    false_negative = ((1.0 - prediction_flat) * target_flat).sum(dim=1)

    dice = (2 * true_positive + eps) / (2 * true_positive + false_positive + false_negative + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    # 无预测正例时 precision=0；无 GT 正例时 recall=0
    precision = torch.where(
        (true_positive + false_positive) > 0,
        true_positive / (true_positive + false_positive + eps),
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        (true_positive + false_negative) > 0,
        true_positive / (true_positive + false_negative + eps),
        torch.zeros_like(true_positive),
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


class MetricAggregator:
    """按 overall / dataset 聚合 per-image 指标。"""

    def __init__(self):
        self._scores = defaultdict(lambda: defaultdict(list))
        self._losses = []

    def update(
        self,
        metrics_per_image: dict[str, torch.Tensor],
        datasets: list[str] | tuple[str, ...],
        loss_value: float | None = None,
    ):
        batch_size = int(metrics_per_image["dice"].shape[0])
        if len(datasets) != batch_size:
            raise ValueError("datasets length must match batch size")
        for index in range(batch_size):
            dataset_name = str(datasets[index])
            for metric_name, values in metrics_per_image.items():
                score = float(values[index].detach().cpu().item())
                self._scores["overall"][metric_name].append(score)
                self._scores[dataset_name][metric_name].append(score)
        if loss_value is not None:
            self._losses.append(float(loss_value))

    def summarize(self) -> dict:
        summary = {}
        for domain, metric_map in self._scores.items():
            summary[domain] = {}
            for metric_name, values in metric_map.items():
                summary[domain][metric_name] = (
                    float(sum(values) / len(values)) if values else 0.0
                )
                summary[domain][f"{metric_name}_count"] = int(len(values))
        summary["loss"] = float(sum(self._losses) / len(self._losses)) if self._losses else None
        return summary
