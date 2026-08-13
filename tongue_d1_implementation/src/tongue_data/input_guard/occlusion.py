"""D4-D：conservative occlusion（多弱证据；不把裂纹/齿痕当遮挡）。"""
from __future__ import annotations

from typing import Any

import numpy as np

from .color_cast import rgb_to_hsv
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


def _distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    try:
        import cv2

        return cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float64)
    except Exception:
        # 简易近似：迭代膨胀距离
        dist = np.zeros(mask.shape, dtype=np.float64)
        if not mask.any():
            return dist
        remaining = mask.astype(bool)
        step = 1.0
        frontier = remaining.copy()
        while frontier.any():
            dist[frontier] = step
            # erode
            padded = np.pad(remaining, 1, constant_values=False)
            eroded = (
                padded[:-2, 1:-1]
                & padded[2:, 1:-1]
                & padded[1:-1, :-2]
                & padded[1:-1, 2:]
                & remaining
            )
            frontier = remaining & ~eroded
            remaining = eroded
            step += 1.0
            if step > max(mask.shape):
                break
        return dist


def compute_occlusion_features(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray | None,
    probability_map: np.ndarray | None,
    *,
    occlusion_cfg: dict[str, Any],
) -> dict[str, Any]:
    """大尺度 interior hole + bright neutral intrusion；忽略细纹理。"""
    if binary_mask is None or probability_map is None:
        return {
            "available": False,
            "reason": "missing_probability_or_mask",
            "interior_hole_ratio": None,
            "bright_intrusion_ratio": None,
            "combined_score": None,
            "evidence_count": 0,
        }

    mask = np.asarray(binary_mask) > 0
    prob = np.asarray(probability_map, dtype=np.float64)
    if mask.shape != prob.shape:
        return {
            "available": False,
            "reason": "mask_prob_shape_mismatch",
            "interior_hole_ratio": None,
            "bright_intrusion_ratio": None,
            "combined_score": None,
            "evidence_count": 0,
        }
    if not mask.any():
        return {
            "available": False,
            "reason": "empty_mask",
            "interior_hole_ratio": None,
            "bright_intrusion_ratio": None,
            "combined_score": None,
            "evidence_count": 0,
        }

    interior_cfg = occlusion_cfg.get("interior", {})
    bright_cfg = occlusion_cfg.get("bright_neutral", {})
    margin_ratio = float(interior_cfg.get("border_margin_ratio", 0.08))
    low_prob_thr = float(interior_cfg.get("low_probability_threshold", 0.35))

    dist = _distance_transform(mask)
    min_side = float(min(mask.shape))
    interior = mask & (dist >= margin_ratio * min_side)
    interior_count = int(interior.sum())
    if interior_count < 32:
        return {
            "available": False,
            "reason": "interior_too_small",
            "interior_hole_ratio": None,
            "bright_intrusion_ratio": None,
            "combined_score": None,
            "evidence_count": 0,
        }

    hole = interior & (prob < low_prob_thr)
    # 忽略极小连通噪声：用简单面积阈值（相对 interior）
    hole_ratio = float(hole.sum() / interior_count)

    rgb = np.asarray(original_rgb)
    hsv = rgb_to_hsv(rgb)
    lum = hsv[..., 2]
    sat = hsv[..., 1]
    bright = (
        mask
        & (lum >= float(bright_cfg.get("min_luminance", 0.70)))
        & (sat <= float(bright_cfg.get("max_hsv_saturation", 0.20)))
    )
    # 大块 bright：同样忽略细碎
    bright_ratio = float(bright.sum() / max(1, int(mask.sum())))

    # 组合分数：强调大尺度
    combined = float(0.65 * hole_ratio + 0.35 * bright_ratio)
    evidence_flags = {
        "interior_hole": hole_ratio >= 0.05,
        "bright_intrusion": bright_ratio >= 0.04,
    }
    evidence_count = int(sum(1 for flag in evidence_flags.values() if flag))

    return {
        "available": True,
        "reason": None,
        "interior_hole_ratio": hole_ratio,
        "bright_intrusion_ratio": bright_ratio,
        "combined_score": combined,
        "evidence_count": evidence_count,
        "evidence_flags": evidence_flags,
        "interior_pixel_count": interior_count,
    }


def evaluate_occlusion(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray | None,
    probability_map: np.ndarray | None,
    policy: InputGuardPolicy,
    *,
    d4d_cfg: dict[str, Any] | None = None,
) -> CheckResult:
    """
    finding 映射到 D4-A 已注册集合：
    none_detected → none
    suspected → possible_occlusion
    major → major
    """
    check_id = CheckId.OCCLUSION.value
    cfg = policy.check_config(CheckId.OCCLUSION)
    if not policy.is_check_enabled(CheckId.OCCLUSION):
        return make_not_evaluated_check(check_id, reason="check_disabled")

    occlusion_cfg = (d4d_cfg or {}).get("occlusion", {})
    features = compute_occlusion_features(
        original_rgb,
        binary_mask,
        probability_map,
        occlusion_cfg=occlusion_cfg,
    )
    thresholds = dict(cfg.get("thresholds") or {})

    if not features.get("available"):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.UNAVAILABLE.value,
            finding=None,
            severity=Severity.NONE.value,
            decision_effect=None,
            score=None,
            thresholds=thresholds,
            evidence={**features, "note": "unavailable != none_detected"},
            reason_code=None,
            source=EvidenceSource.SIGNAL_RULE.value,
        )

    warning_combined = thresholds.get("warning_combined_score")
    retake_combined = thresholds.get("retake_combined_score")
    if warning_combined is None or retake_combined is None:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.UNAVAILABLE.value,
            finding=None,
            severity=Severity.NONE.value,
            decision_effect=None,
            score=features.get("combined_score"),
            thresholds=thresholds,
            evidence={**features, "fallback": "thresholds_not_calibrated"},
            reason_code=None,
            source=EvidenceSource.SIGNAL_RULE.value,
        )

    score = float(features["combined_score"])
    require_multi = bool(
        thresholds.get("require_multi_evidence_for_retake", True)
    )
    evidence_count = int(features.get("evidence_count") or 0)

    if score >= float(retake_combined):
        if require_multi and evidence_count < 2:
            # 单弱证据不得 severe RETAKE → 降为 warning
            return CheckResult(
                check_id=check_id,
                evaluation_state=EvaluationState.EVALUATED.value,
                finding="possible_occlusion",
                severity=Severity.MODERATE.value,
                decision_effect=Decision.WARNING.value,
                score=score,
                thresholds=thresholds,
                evidence={
                    **features,
                    "downgraded_from_retake": "single_weak_evidence",
                },
                reason_code=ReasonCode.TONGUE_OCCLUDED.value,
                source=EvidenceSource.SIGNAL_RULE.value,
            )
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="major",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=score,
            thresholds=thresholds,
            evidence=features,
            reason_code=ReasonCode.TONGUE_OCCLUDED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if score >= float(warning_combined):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="possible_occlusion",
            severity=Severity.MODERATE.value,
            decision_effect=Decision.WARNING.value,
            score=score,
            thresholds=thresholds,
            evidence=features,
            reason_code=ReasonCode.TONGUE_OCCLUDED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="none",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=score,
        thresholds=thresholds,
        evidence=features,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def apply_synthetic_occlusion(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
    *,
    area_ratio: float,
    mode: str = "bright",
    seed: int = 20260813,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在舌体 interior 覆盖色块；同时压低对应 probability（由调用方处理）。
    返回 (rgb_out, occlusion_mask)。
    """
    rng = np.random.default_rng(int(seed))
    out = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(binary_mask) > 0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return out, np.zeros(mask.shape, dtype=bool)
    # 选 interior 中心附近
    center_y = int(np.median(ys))
    center_x = int(np.median(xs))
    tongue_area = int(mask.sum())
    target = max(16, int(tongue_area * float(area_ratio)))
    side = max(4, int(np.sqrt(target)))
    y0 = max(0, center_y - side // 2)
    x0 = max(0, center_x - side // 2)
    y1 = min(mask.shape[0], y0 + side)
    x1 = min(mask.shape[1], x0 + side)
    occ = np.zeros(mask.shape, dtype=bool)
    occ[y0:y1, x0:x1] = True
    occ &= mask
    if mode == "bright":
        color = np.array([230, 230, 225], dtype=np.uint8)
    elif mode == "dark":
        color = np.array([20, 20, 25], dtype=np.uint8)
    else:
        color = np.array(
            [int(rng.integers(40, 80))] * 3,
            dtype=np.uint8,
        )
    out[occ] = color
    return out, occ
