"""Segmentation metrics：Dice / IoU / Precision / Recall。"""
from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(array_like: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def binarize_prediction(prediction: Any, threshold: float = 0.5) -> np.ndarray:
    """probability/logit≥threshold → 1；支持概率或已二值 mask。"""
    array = _to_numpy(prediction).astype(np.float32)
    return (array >= float(threshold)).astype(np.float32)


def _confusion(prediction_binary: np.ndarray, target_binary: np.ndarray) -> tuple[float, float, float]:
    prediction_flat = prediction_binary.reshape(-1) > 0.5
    target_flat = target_binary.reshape(-1) > 0.5
    true_positive = float(np.logical_and(prediction_flat, target_flat).sum())
    false_positive = float(np.logical_and(prediction_flat, ~target_flat).sum())
    false_negative = float(np.logical_and(~prediction_flat, target_flat).sum())
    return true_positive, false_positive, false_negative


def dice_coefficient(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Dice = 2TP / (2TP + FP + FN)。空预测→0（若 GT 非空）。"""
    prediction_binary = binarize_prediction(prediction, threshold)
    target_binary = (_to_numpy(target) > 0.5).astype(np.float32)
    if target_binary.sum() <= 0:
        raise ValueError("GT mask is empty; dataset validation should reject empty GT")
    true_positive, false_positive, false_negative = _confusion(prediction_binary, target_binary)
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator <= 0:
        return 0.0
    return float((2 * true_positive + eps) / (denominator + eps))


def iou_score(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """IoU = TP / (TP + FP + FN)。"""
    prediction_binary = binarize_prediction(prediction, threshold)
    target_binary = (_to_numpy(target) > 0.5).astype(np.float32)
    if target_binary.sum() <= 0:
        raise ValueError("GT mask is empty; dataset validation should reject empty GT")
    true_positive, false_positive, false_negative = _confusion(prediction_binary, target_binary)
    denominator = true_positive + false_positive + false_negative
    if denominator <= 0:
        return 0.0
    return float((true_positive + eps) / (denominator + eps))


def precision_score(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    prediction_binary = binarize_prediction(prediction, threshold)
    target_binary = (_to_numpy(target) > 0.5).astype(np.float32)
    true_positive, false_positive, _false_negative = _confusion(prediction_binary, target_binary)
    denominator = true_positive + false_positive
    if denominator <= 0:
        return 0.0
    return float((true_positive + eps) / (denominator + eps))


def recall_score(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    prediction_binary = binarize_prediction(prediction, threshold)
    target_binary = (_to_numpy(target) > 0.5).astype(np.float32)
    true_positive, _false_positive, false_negative = _confusion(prediction_binary, target_binary)
    denominator = true_positive + false_negative
    if denominator <= 0:
        return 0.0
    return float((true_positive + eps) / (denominator + eps))


def compute_segmentation_metrics(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    return {
        "dice": dice_coefficient(prediction, target, threshold),
        "iou": iou_score(prediction, target, threshold),
        "precision": precision_score(prediction, target, threshold),
        "recall": recall_score(prediction, target, threshold),
    }


def summarize_per_image_scores(scores: list[float]) -> dict[str, float]:
    """per-image mean 及分布统计。"""
    if not scores:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "count": 0,
        }
    array = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "count": int(len(array)),
    }
