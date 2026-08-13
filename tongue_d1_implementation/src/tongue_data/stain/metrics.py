"""Stain 二分类与 3-state runtime 指标。"""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def binary_metrics_at_threshold(
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(y_score)
    preds = (scores >= float(threshold)).astype(int)
    true_positive = int(((preds == 1) & (labels == 1)).sum())
    true_negative = int(((preds == 0) & (labels == 0)).sum())
    false_positive = int(((preds == 1) & (labels == 0)).sum())
    false_negative = int(((preds == 0) & (labels == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive
    ) else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative
    ) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (
        true_negative + false_positive
    ) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (true_positive + true_negative) / len(labels) if len(labels) else 0.0
    balanced = 0.5 * (recall + specificity)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def roc_auc_score(y_true, y_score) -> float | None:
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(y_score)
    if len(np.unique(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score as _roc_auc

        return float(_roc_auc(labels, scores))
    except Exception:
        # 无 sklearn 时用 Wilcoxon-Mann-Whitney 估计
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        correct = 0.0
        for positive_score in pos:
            correct += float((neg < positive_score).sum())
            correct += 0.5 * float((neg == positive_score).sum())
        return float(correct / (len(pos) * len(neg)))


def pr_auc_score(y_true, y_score) -> float | None:
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(y_score)
    if len(np.unique(labels)) < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(labels, scores))
    except Exception:
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        tp = 0
        fp = 0
        positives = int((labels == 1).sum())
        if positives == 0:
            return None
        precision_sum = 0.0
        prev_recall = 0.0
        for index, label in enumerate(labels_sorted, start=1):
            if label == 1:
                tp += 1
            else:
                fp += 1
            precision = tp / (tp + fp)
            recall = tp / positives
            precision_sum += precision * (recall - prev_recall)
            prev_recall = recall
        return float(precision_sum)


def brier_score(y_true, y_score) -> float:
    labels = _as_numpy(y_true)
    scores = _as_numpy(y_score)
    return float(np.mean((scores - labels) ** 2))


def expected_calibration_error(y_true, y_score, *, n_bins: int = 10) -> float:
    labels = _as_numpy(y_true)
    scores = np.clip(_as_numpy(y_score), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = len(labels)
    if total == 0:
        return 0.0
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (scores >= left) & (scores < right if right < 1.0 else scores <= right)
        if not mask.any():
            continue
        bin_confidence = float(scores[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        ece += (mask.sum() / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def map_probability_to_finding(
    probability: float,
    t_clear: float,
    t_retake: float,
) -> str:
    """false / uncertain / true。"""
    if t_clear >= t_retake:
        raise ValueError(f"require t_clear < t_retake, got {t_clear} / {t_retake}")
    score = float(probability)
    if score <= float(t_clear):
        return "false"
    if score >= float(t_retake):
        return "true"
    return "uncertain"


def three_state_metrics(
    y_true,
    y_score,
    *,
    t_clear: float,
    t_retake: float,
) -> dict[str, Any]:
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(y_score)
    findings = np.array(
        [map_probability_to_finding(score, t_clear, t_retake) for score in scores]
    )
    clear_mask = findings == "false"
    stain_mask = findings == "true"
    uncertain_mask = findings == "uncertain"

    clear_count = int(clear_mask.sum())
    stain_count = int(stain_mask.sum())
    uncertain_count = int(uncertain_mask.sum())
    total = int(len(labels))

    clean_purity = (
        float((labels[clear_mask] == 0).mean()) if clear_count else None
    )
    stain_precision = (
        float((labels[stain_mask] == 1).mean()) if stain_count else None
    )
    # stain recall：真实 stained 中被判 true 的比例（uncertain 不算命中）
    positives = int((labels == 1).sum())
    stain_recall = (
        float(((labels == 1) & stain_mask).sum() / positives) if positives else None
    )
    confident_coverage = (
        float((clear_count + stain_count) / total) if total else 0.0
    )
    return {
        "t_clear": float(t_clear),
        "t_retake": float(t_retake),
        "clear_count": clear_count,
        "uncertain_count": uncertain_count,
        "stain_count": stain_count,
        "uncertain_rate": float(uncertain_count / total) if total else 0.0,
        "confident_coverage": confident_coverage,
        "confident_clean_purity": clean_purity,
        "confident_stain_precision": stain_precision,
        "stain_recall": stain_recall,
        "findings": findings.tolist(),
    }


def summarize_ranking_metrics(y_true, y_score) -> dict[str, Any]:
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "pr_auc": pr_auc_score(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
        **{f"at_0.5_{key}": value for key, value in binary_metrics_at_threshold(
            y_true, y_score, threshold=0.5
        ).items() if key != "threshold"},
    }
