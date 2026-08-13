"""D4-D：conservative color_cast（neutral-reference，禁止舌色捷径）。"""
from __future__ import annotations

from typing import Any

import numpy as np

from .ontology import (
    CheckId,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    Severity,
)
from .policy import InputGuardPolicy
from .schema import CheckResult, make_not_evaluated_check


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 RGB → Lab（L[0,100], a/b 约[-128,127]）。OpenCV 优先并重标度。"""
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)
    try:
        import cv2

        lab_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
        # OpenCV: L∈[0,255], a/b∈[0,255]（128=中性）→ 标准 Lab
        lab = np.empty_like(lab_cv)
        lab[..., 0] = lab_cv[..., 0] * (100.0 / 255.0)
        lab[..., 1] = lab_cv[..., 1] - 128.0
        lab[..., 2] = lab_cv[..., 2] - 128.0
        return lab
    except Exception:
        # 简化 sRGB→Lab（足够工程 baseline）
        linear = rgb.astype(np.float64) / 255.0
        mask = linear <= 0.04045
        linear = np.where(mask, linear / 12.92, ((linear + 0.055) / 1.055) ** 2.4)
        matrix = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        xyz = linear @ matrix.T
        xyz[..., 0] /= 0.95047
        xyz[..., 2] /= 1.08883
        epsilon = 216 / 24389
        kappa = 24389 / 27

        def _f(channel: np.ndarray) -> np.ndarray:
            return np.where(
                channel > epsilon,
                np.cbrt(channel),
                (kappa * channel + 16.0) / 116.0,
            )

        fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
        lab = np.stack(
            [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1
        )
        return lab


def rgb_to_hsv(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_uint8, dtype=np.float64) / 255.0
    max_channel = rgb.max(axis=-1)
    min_channel = rgb.min(axis=-1)
    delta = max_channel - min_channel
    saturation = np.zeros_like(max_channel)
    valid = max_channel > 1e-8
    saturation[valid] = delta[valid] / max_channel[valid]
    value = max_channel
    # hue 不用于 neutral 选择主判据
    hue = np.zeros_like(value)
    return np.stack([hue, saturation, value], axis=-1)


def compute_color_cast_features(
    original_rgb: np.ndarray,
    tongue_mask: np.ndarray | None,
    *,
    neutral_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    仅使用 tongue mask 外区域估计 neutral support / cast。
    禁止使用 tongue ROI mean RGB 作为 cast 判据。
    """
    rgb = np.asarray(original_rgb)
    if rgb.dtype != np.uint8:
        raise ValueError("color_cast requires original uint8 RGB")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("color_cast requires HxWx3 RGB")

    height, width = rgb.shape[:2]
    if tongue_mask is None:
        outside = np.ones((height, width), dtype=bool)
    else:
        mask = np.asarray(tongue_mask) > 0
        if mask.shape != (height, width):
            raise ValueError("tongue_mask shape mismatch")
        outside = ~mask

    hsv = rgb_to_hsv(rgb)
    lab = rgb_to_lab(rgb)
    luminance = hsv[..., 2]
    saturation = hsv[..., 1]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    clip_ratio = (
        (rgb.max(axis=-1) >= 250).astype(np.float64)
        + (rgb.min(axis=-1) <= 5).astype(np.float64)
    ) / 2.0

    min_lum = float(neutral_cfg.get("min_luminance", 0.25))
    max_lum = float(neutral_cfg.get("max_luminance", 0.92))
    max_sat = float(neutral_cfg.get("max_hsv_saturation", 0.18))
    max_chroma = float(neutral_cfg.get("max_lab_chroma", 12.0))
    max_clip = float(neutral_cfg.get("max_channel_clip_ratio", 0.02))

    luminance_gate = (
        outside
        & (luminance >= min_lum)
        & (luminance <= max_lum)
        & (clip_ratio <= max_clip)
    )
    # 主路径：低饱和/低色度 neutral；强偏色时它们会消失
    primary = (
        luminance_gate & (saturation <= max_sat) & (chroma <= max_chroma)
    )
    # 回退：tongue 外宽亮度采样（允许轻微 clipping），用于估计偏色幅度
    fallback = outside & (luminance >= max(0.05, min_lum * 0.5))
    use_fallback = False
    candidate = primary
    if int(primary.sum()) < int(neutral_cfg.get("min_candidate_count", 200)):
        candidate = fallback
        use_fallback = True

    candidate_count = int(candidate.sum())
    outside_count = int(outside.sum())
    candidate_ratio = (
        float(candidate_count / outside_count) if outside_count > 0 else 0.0
    )

    def _spatial_coverage(mask_bool: np.ndarray) -> float:
        grid_rows, grid_cols = 8, 8
        cell_h = max(1, height // grid_rows)
        cell_w = max(1, width // grid_cols)
        covered = 0
        total_cells = 0
        for row_index in range(grid_rows):
            for col_index in range(grid_cols):
                total_cells += 1
                block = mask_bool[
                    row_index * cell_h : (row_index + 1) * cell_h,
                    col_index * cell_w : (col_index + 1) * cell_w,
                ]
                if block.size and block.any():
                    covered += 1
        return float(covered / total_cells) if total_cells else 0.0

    spatial_coverage = _spatial_coverage(candidate)

    # gray-world 仅辅助（整图，不含决策主路径）
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0).astype(np.float64) + 1e-6
    gray_world_shift = float(np.std(mean_rgb / mean_rgb.mean()))

    if candidate_count == 0:
        return {
            "neutral_pixel_ratio": 0.0,
            "neutral_candidate_count": 0,
            "neutral_spatial_coverage": spatial_coverage,
            "neutral_lab_a_median": None,
            "neutral_lab_b_median": None,
            "neutral_chroma_median": None,
            "neutral_chroma_p95": None,
            "estimated_cast_magnitude": None,
            "channel_neutrality_error": None,
            "gray_world_shift": gray_world_shift,
            "support_ok": False,
            "neutral_mode": "none",
            "tongue_mean_rgb_used": False,
        }

    a_vals = lab[..., 1][candidate]
    b_vals = lab[..., 2][candidate]
    chroma_vals = chroma[candidate]
    a_med = float(np.median(a_vals))
    b_med = float(np.median(b_vals))
    cast_magnitude = float(np.sqrt(a_med**2 + b_med**2))
    channel_err = float(
        np.mean(np.abs(rgb[candidate].astype(np.float64).mean(axis=0) - 128.0))
        / 128.0
    )

    min_count = int(neutral_cfg.get("min_candidate_count", 200))
    min_ratio = float(neutral_cfg.get("min_candidate_ratio", 0.01))
    min_coverage = float(neutral_cfg.get("min_spatial_coverage", 0.05))
    support_ok = (
        candidate_count >= min_count
        and candidate_ratio >= min_ratio
        and spatial_coverage >= min_coverage
    )

    return {
        "neutral_pixel_ratio": candidate_ratio,
        "neutral_candidate_count": candidate_count,
        "neutral_spatial_coverage": spatial_coverage,
        "neutral_lab_a_median": a_med,
        "neutral_lab_b_median": b_med,
        "neutral_chroma_median": float(np.median(chroma_vals)),
        "neutral_chroma_p95": float(np.percentile(chroma_vals, 95)),
        "estimated_cast_magnitude": cast_magnitude,
        "channel_neutrality_error": channel_err,
        "gray_world_shift": gray_world_shift,
        "support_ok": bool(support_ok),
        "neutral_mode": "fallback_luminance" if use_fallback else "low_chroma",
        "tongue_mean_rgb_used": False,
    }


def evaluate_color_cast(
    original_rgb: np.ndarray,
    tongue_mask: np.ndarray | None,
    policy: InputGuardPolicy,
    *,
    d4d_cfg: dict[str, Any] | None = None,
) -> CheckResult:
    check_id = CheckId.COLOR_CAST.value
    cfg = policy.check_config(CheckId.COLOR_CAST)
    if not policy.is_check_enabled(CheckId.COLOR_CAST):
        return make_not_evaluated_check(check_id, reason="check_disabled")

    neutral_cfg = (d4d_cfg or {}).get("color_cast", {}).get("neutral", {})
    # policy 可覆盖 neutral 参数
    if isinstance(cfg.get("neutral_support"), dict):
        neutral_cfg = {**neutral_cfg, **cfg["neutral_support"]}

    features = compute_color_cast_features(
        original_rgb, tongue_mask, neutral_cfg=neutral_cfg
    )
    thresholds = dict(cfg.get("thresholds") or {})

    if not features["support_ok"]:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.UNAVAILABLE.value,
            finding=None,
            severity=Severity.NONE.value,
            decision_effect=None,
            score=None,
            thresholds=thresholds,
            evidence={
                **features,
                "fallback": "insufficient_neutral_support",
                "note": "unavailable != pass",
            },
            reason_code=None,
            source=EvidenceSource.SIGNAL_RULE.value,
        )

    magnitude = float(features["estimated_cast_magnitude"])
    warning_thr = thresholds.get("warning_cast_magnitude")
    retake_thr = thresholds.get("retake_cast_magnitude")
    if warning_thr is None or retake_thr is None:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.UNAVAILABLE.value,
            finding=None,
            severity=Severity.NONE.value,
            decision_effect=None,
            score=magnitude,
            thresholds=thresholds,
            evidence={**features, "fallback": "thresholds_not_calibrated"},
            reason_code=None,
            source=EvidenceSource.SIGNAL_RULE.value,
        )

    warning_thr = float(warning_thr)
    retake_thr = float(retake_thr)
    if magnitude >= retake_thr:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="severe",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=magnitude,
            thresholds=thresholds,
            evidence=features,
            reason_code=ReasonCode.SEVERE_COLOR_CAST.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if magnitude >= warning_thr:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="suspected",
            severity=Severity.MODERATE.value,
            decision_effect=Decision.WARNING.value,
            score=magnitude,
            thresholds=thresholds,
            evidence=features,
            reason_code=ReasonCode.COLOR_CAST_SUSPECTED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="acceptable",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=magnitude,
        thresholds=thresholds,
        evidence=features,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def apply_channel_cast(
    rgb: np.ndarray,
    *,
    direction: str,
    gain: float,
) -> np.ndarray:
    """Deterministic synthetic cast；不用于正式数据集。"""
    out = np.asarray(rgb, dtype=np.float64).copy()
    gain = float(gain)
    key = direction.lower()
    if key == "red":
        out[..., 0] *= gain
    elif key == "green":
        out[..., 1] *= gain
    elif key == "blue":
        out[..., 2] *= gain
    elif key == "yellow":
        out[..., 0] *= gain
        out[..., 1] *= gain
    else:
        raise ValueError(f"unsupported cast direction: {direction}")
    return np.clip(out, 0, 255).astype(np.uint8)
