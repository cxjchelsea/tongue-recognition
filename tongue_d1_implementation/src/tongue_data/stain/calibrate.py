"""Validation-only 双阈值校准：t_clear / t_retake。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import map_probability_to_finding, three_state_metrics


def calibrate_dual_thresholds(
    y_true,
    y_score,
    *,
    target_confident_precision: float = 0.90,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Deterministic:
    - t_clear: 满足 clean purity target 的最大阈值
    - t_retake: 满足 stain precision target 的最小阈值
    要求最终 t_clear < t_retake；否则 constraint_not_met。
    """
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("y_true/y_score length mismatch")
    if grid is None:
        # 固定网格，保证可复现
        grid = np.round(np.linspace(0.01, 0.99, 99), 4)

    target = float(target_confident_precision)
    clear_candidates: list[float] = []
    for threshold in grid:
        mask = scores <= threshold
        if mask.sum() == 0:
            continue
        purity = float((labels[mask] == 0).mean())
        if purity + 1e-12 >= target:
            clear_candidates.append(float(threshold))
    retake_candidates: list[float] = []
    for threshold in grid:
        mask = scores >= threshold
        if mask.sum() == 0:
            continue
        precision = float((labels[mask] == 1).mean())
        if precision + 1e-12 >= target:
            retake_candidates.append(float(threshold))

    constraint_not_met = False
    reasons: list[str] = []
    if not clear_candidates:
        constraint_not_met = True
        reasons.append("no_t_clear_meets_clean_purity_target")
    if not retake_candidates:
        constraint_not_met = True
        reasons.append("no_t_retake_meets_stain_precision_target")

    # 强分离时独立的 max(clear)/min(retake) 可能交叉。
    # 合法对 (tc<tr) 中按以下字典序选择（deterministic）：
    # 1) 最大 confident coverage
    # 2) 最大 t_clear
    # 3) 最小 t_retake
    t_clear_out: float
    t_retake_out: float
    if clear_candidates and retake_candidates:
        valid_pairs = [
            (clear_threshold, retake_threshold)
            for clear_threshold in clear_candidates
            for retake_threshold in retake_candidates
            if clear_threshold < retake_threshold
        ]
        if valid_pairs:
            def _pair_key(pair: tuple[float, float]) -> tuple[float, float, float]:
                clear_threshold, retake_threshold = pair
                coverage = float(
                    ((scores <= clear_threshold) | (scores >= retake_threshold)).mean()
                )
                return (coverage, clear_threshold, -retake_threshold)

            t_clear_out, t_retake_out = max(valid_pairs, key=_pair_key)
        else:
            constraint_not_met = True
            reasons.append("no_valid_pair_with_t_clear_lt_t_retake")
            # 诊断 fallback：不降低 purity target，只保证顺序
            t_clear_out = min(clear_candidates)
            t_retake_out = max(retake_candidates)
            if t_clear_out >= t_retake_out:
                t_clear_out = 0.49
                t_retake_out = 0.50
    else:
        t_clear_out = max(clear_candidates) if clear_candidates else 0.2
        t_retake_out = min(retake_candidates) if retake_candidates else 0.8
        if t_clear_out >= t_retake_out:
            t_clear_out = 0.49
            t_retake_out = 0.50

    metrics = three_state_metrics(
        labels, scores, t_clear=t_clear_out, t_retake=t_retake_out
    )
    return {
        "target_confident_precision": target,
        "t_clear": t_clear_out,
        "t_retake": t_retake_out,
        "constraint_not_met": bool(constraint_not_met),
        "constraint_reasons": reasons,
        "clear_candidates_count": len(clear_candidates),
        "retake_candidates_count": len(retake_candidates),
        "source_split": "val",
        **metrics,
    }


def freeze_thresholds(
    calibration: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t_clear": float(calibration["t_clear"]),
        "t_retake": float(calibration["t_retake"]),
        "target_confident_precision": float(
            calibration["target_confident_precision"]
        ),
        "constraint_not_met": bool(calibration["constraint_not_met"]),
        "constraint_reasons": list(calibration.get("constraint_reasons", [])),
        "source_split": "val",
        "val_clear_purity": calibration.get("confident_clean_purity"),
        "val_stain_precision": calibration.get("confident_stain_precision"),
        "val_stain_recall": calibration.get("stain_recall"),
        "val_uncertain_rate": calibration.get("uncertain_rate"),
        "val_confident_coverage": calibration.get("confident_coverage"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_frozen_thresholds(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    t_clear = float(doc["t_clear"])
    t_retake = float(doc["t_retake"])
    if not (t_clear < t_retake):
        raise ValueError(f"frozen thresholds invalid: {t_clear} >= {t_retake}")
    return doc


def apply_frozen_thresholds_to_frame(
    frame: pd.DataFrame,
    *,
    probability_column: str = "p_stain",
    t_clear: float,
    t_retake: float,
) -> pd.DataFrame:
    """对已有预测套用 frozen thresholds；禁止重算。"""
    result = frame.copy()
    result["finding"] = [
        map_probability_to_finding(float(score), t_clear, t_retake)
        for score in result[probability_column].tolist()
    ]
    return result
