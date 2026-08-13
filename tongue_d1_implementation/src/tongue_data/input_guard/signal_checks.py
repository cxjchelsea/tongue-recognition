"""D4-B signal_rule 检查：统一输出 CheckResult。"""
from __future__ import annotations

from typing import Any

from .features import InputGuardFeatures
from .ontology import (
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    Severity,
)
from .policy import InputGuardPolicy
from .schema import CheckResult, make_not_evaluated_check

# D4-B 已实现 checks
IMPLEMENTED_SIGNAL_CHECKS = frozenset(
    {
        CheckId.TONGUE_PRESENCE,
        CheckId.TONGUE_SCALE,
        CheckId.TONGUE_COMPLETENESS,
        CheckId.SEGMENTATION_INTEGRITY,
        CheckId.FOCUS,
        CheckId.EXPOSURE,
        CheckId.ILLUMINATION_UNIFORMITY,
        CheckId.RESOLUTION,
    }
)

DEFERRED_CHECKS = frozenset(
    {
        CheckId.COLOR_CAST,
        CheckId.OCCLUSION,
        CheckId.STAIN_SUSPECTED,
    }
)


def _thr(cfg: dict, key: str) -> float | None:
    thresholds = cfg.get("thresholds") or {}
    value = thresholds.get(key)
    return None if value is None else float(value)


def _policy_meta(policy: InputGuardPolicy) -> dict[str, Any]:
    return {
        "policy_version": str(policy.doc.get("policy_version", policy.version)),
        "threshold_status": "engineering_heuristic",
    }


def evaluate_tongue_presence(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    check_id = CheckId.TONGUE_PRESENCE.value
    cfg = policy.check_config(CheckId.TONGUE_PRESENCE)
    if features.segmentation_status == "no_tongue_detected" or (
        features.tongue_pixel_count is not None and features.tongue_pixel_count <= 0
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="absent",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=features.foreground_ratio,
            thresholds=cfg.get("thresholds"),
            evidence={
                "segmentation_status": features.segmentation_status,
                "foreground_ratio": features.foreground_ratio,
                "tongue_pixel_count": features.tongue_pixel_count,
                **_policy_meta(policy),
            },
            reason_code=ReasonCode.NO_TONGUE_DETECTED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="present",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=features.foreground_ratio,
        thresholds=cfg.get("thresholds"),
        evidence={
            "segmentation_status": features.segmentation_status,
            "foreground_ratio": features.foreground_ratio,
            **_policy_meta(policy),
        },
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_tongue_scale(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    """多特征：foreground + tight bbox width/height；阈值：值越小越差。"""
    check_id = CheckId.TONGUE_SCALE.value
    cfg = policy.check_config(CheckId.TONGUE_SCALE)
    retake_fg = _thr(cfg, "retake_foreground_ratio")
    warning_fg = _thr(cfg, "warning_foreground_ratio")
    retake_w = _thr(cfg, "retake_bbox_width_ratio")
    warning_w = _thr(cfg, "warning_bbox_width_ratio")
    retake_h = _thr(cfg, "retake_bbox_height_ratio")
    warning_h = _thr(cfg, "warning_bbox_height_ratio")
    # 兼容旧单阈值字段
    if retake_fg is None:
        retake_fg = _thr(cfg, "retake")
    if warning_fg is None:
        warning_fg = _thr(cfg, "warning")

    fg = features.foreground_ratio
    width_ratio = features.bbox_width_ratio
    height_ratio = features.bbox_height_ratio
    if fg is None or width_ratio is None or height_ratio is None:
        return make_not_evaluated_check(check_id, reason="missing_scale_features")

    retake_hits = 0
    warning_hits = 0
    if retake_fg is not None and fg < retake_fg:
        retake_hits += 1
    elif warning_fg is not None and fg < warning_fg:
        warning_hits += 1
    if retake_w is not None and width_ratio < retake_w:
        retake_hits += 1
    elif warning_w is not None and width_ratio < warning_w:
        warning_hits += 1
    if retake_h is not None and height_ratio < retake_h:
        retake_hits += 1
    elif warning_h is not None and height_ratio < warning_h:
        warning_hits += 1

    evidence = {
        "foreground_ratio": fg,
        "bbox_width_ratio": width_ratio,
        "bbox_height_ratio": height_ratio,
        "bbox_area_ratio": features.bbox_area_ratio,
        "bbox_source": "tight",
        "retake_hits": retake_hits,
        "warning_hits": warning_hits,
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    # 严重：至少两项达到 retake，或前景极低
    if retake_hits >= 2 or (
        retake_fg is not None and fg < retake_fg and retake_hits >= 1
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="too_small",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=fg,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_TOO_SMALL.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if retake_hits >= 1 or warning_hits >= 1:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="small",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=fg,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_SLIGHTLY_SMALL.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="adequate",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=fg,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_tongue_completeness(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    """top-only 不自动 RETAKE；左右/底部高接触才严重。"""
    check_id = CheckId.TONGUE_COMPLETENESS.value
    cfg = policy.check_config(CheckId.TONGUE_COMPLETENESS)
    warning_side = _thr(cfg, "warning_side_touch_ratio")
    retake_side = _thr(cfg, "retake_side_touch_ratio")
    warning_bottom = _thr(cfg, "warning_bottom_touch_ratio")
    retake_bottom = _thr(cfg, "retake_bottom_touch_ratio")
    warning_top = _thr(cfg, "warning_top_touch_ratio")

    left = features.left_touch_ratio
    right = features.right_touch_ratio
    top = features.top_touch_ratio
    bottom = features.bottom_touch_ratio
    if None in (left, right, top, bottom):
        return make_not_evaluated_check(check_id, reason="missing_border_features")

    evidence = {
        "left_touch_ratio": left,
        "right_touch_ratio": right,
        "top_touch_ratio": top,
        "bottom_touch_ratio": bottom,
        "border_touch_ratio": features.border_touch_ratio,
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    severe_side = (
        (retake_side is not None and left >= retake_side)
        or (retake_side is not None and right >= retake_side)
        or (retake_bottom is not None and bottom >= retake_bottom)
    )
    if severe_side:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="cropped",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=features.border_touch_ratio,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_CROPPED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    mild = (
        (warning_side is not None and (left >= warning_side or right >= warning_side))
        or (warning_bottom is not None and bottom >= warning_bottom)
        or (warning_top is not None and top >= warning_top)
    )
    if mild:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="possibly_cropped",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=features.border_touch_ratio,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_TOUCHES_FRAME.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="complete",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=features.border_touch_ratio,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_segmentation_integrity(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    check_id = CheckId.SEGMENTATION_INTEGRITY.value
    cfg = policy.check_config(CheckId.SEGMENTATION_INTEGRITY)
    retake_ratio = _thr(cfg, "retake_largest_component_ratio")
    warning_ratio = _thr(cfg, "warning_largest_component_ratio")
    warning_prob = _thr(cfg, "warning_mean_probability")
    retake_components = _thr(cfg, "retake_component_count")

    largest = features.largest_component_ratio
    components = features.component_count
    mean_prob = features.mean_foreground_probability
    if largest is None or components is None:
        return make_not_evaluated_check(check_id, reason="missing_segmentation_features")

    evidence = {
        "component_count_before": components,
        "largest_component_ratio": largest,
        "mean_foreground_probability": mean_prob,
        "max_probability": features.max_probability,
        "confidence_note": "mean_foreground_probability is a model confidence proxy only",
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    if retake_ratio is not None and largest < retake_ratio:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="fragmented",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=largest,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.SEGMENTATION_FRAGMENTED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if retake_components is not None and components >= retake_components and (
        warning_ratio is not None and largest < warning_ratio
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="uncertain",
            severity=Severity.MODERATE.value,
            decision_effect=Decision.RETAKE.value,
            score=largest,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.SEGMENTATION_IMPLAUSIBLE.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if (warning_ratio is not None and largest < warning_ratio) or (
        warning_prob is not None and mean_prob is not None and mean_prob < warning_prob
    ):
        reason = (
            ReasonCode.SEGMENTATION_LOW_CONFIDENCE.value
            if warning_prob is not None
            and mean_prob is not None
            and mean_prob < warning_prob
            else ReasonCode.SEGMENTATION_FRAGMENTED.value
        )
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="uncertain",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=largest,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=reason,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="good",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=largest,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_focus(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    """越低越模糊；优先 ROI blur。"""
    check_id = CheckId.FOCUS.value
    cfg = policy.check_config(CheckId.FOCUS)
    retake = _thr(cfg, "retake_roi_laplacian")
    warning = _thr(cfg, "warning_roi_laplacian")
    score = features.roi_blur_score
    if score is None:
        return make_not_evaluated_check(check_id, reason="missing_roi_blur")
    evidence = {
        "roi_blur_score": features.roi_blur_score,
        "blur_score": features.blur_score,
        "roi_gradient_energy": features.roi_gradient_energy,
        "image_gradient_energy": features.image_gradient_energy,
        "blur_analysis_long_side": features.blur_analysis_long_side,
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    # 边界：score < retake → RETAKE；score < warning → WARNING；== 边界按 < 不触发更严
    if retake is not None and score < retake:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="blurred",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=score,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_BLUR.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if warning is not None and score < warning:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="slightly_blurred",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=score,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.IMAGE_BLUR.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="sharp",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=score,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_exposure(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    check_id = CheckId.EXPOSURE.value
    cfg = policy.check_config(CheckId.EXPOSURE)
    retake_dark = _thr(cfg, "retake_dark_pixel_ratio")
    warning_dark = _thr(cfg, "warning_dark_pixel_ratio")
    retake_bright = _thr(cfg, "retake_bright_pixel_ratio")
    warning_bright = _thr(cfg, "warning_bright_pixel_ratio")
    retake_shadow = _thr(cfg, "retake_shadow_clip_ratio")
    retake_highlight = _thr(cfg, "retake_highlight_clip_ratio")
    warning_shadow = _thr(cfg, "warning_shadow_clip_ratio")
    warning_highlight = _thr(cfg, "warning_highlight_clip_ratio")

    dark = features.dark_pixel_ratio
    bright = features.bright_pixel_ratio
    shadow = features.shadow_clip_ratio
    highlight = features.highlight_clip_ratio
    if None in (dark, bright, shadow, highlight, features.mean_luminance):
        return make_not_evaluated_check(check_id, reason="missing_exposure_features")

    evidence = {
        "mean_luminance": features.mean_luminance,
        "roi_luminance_p01": features.roi_luminance_p01,
        "roi_luminance_p50": features.roi_luminance_p50,
        "roi_luminance_p99": features.roi_luminance_p99,
        "dark_pixel_ratio": dark,
        "bright_pixel_ratio": bright,
        "shadow_clip_ratio": shadow,
        "highlight_clip_ratio": highlight,
        "note": "exposure uses percentiles/clipping; not tongue-color phenotype",
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }

    if (retake_highlight is not None and highlight >= retake_highlight) or (
        retake_bright is not None and bright >= retake_bright
    ):
        reason = (
            ReasonCode.HIGHLIGHT_CLIPPING.value
            if retake_highlight is not None and highlight >= retake_highlight
            else ReasonCode.OVEREXPOSED.value
        )
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="overexposed",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=bright,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=reason,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if (retake_shadow is not None and shadow >= retake_shadow) or (
        retake_dark is not None and dark >= retake_dark
    ):
        reason = (
            ReasonCode.SHADOW_CLIPPING.value
            if retake_shadow is not None and shadow >= retake_shadow
            else ReasonCode.UNDEREXPOSED.value
        )
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="underexposed",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=dark,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=reason,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if (warning_highlight is not None and highlight >= warning_highlight) or (
        warning_bright is not None and bright >= warning_bright
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="slightly_overexposed",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=bright,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.OVEREXPOSED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if (warning_shadow is not None and shadow >= warning_shadow) or (
        warning_dark is not None and dark >= warning_dark
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="slightly_underexposed",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=dark,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.UNDEREXPOSED.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="normal",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=features.mean_luminance,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_illumination(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    check_id = CheckId.ILLUMINATION_UNIFORMITY.value
    cfg = policy.check_config(CheckId.ILLUMINATION_UNIFORMITY)
    retake = _thr(cfg, "retake_relative_range")
    warning = _thr(cfg, "warning_relative_range")
    score = features.relative_luminance_range
    if score is None:
        return make_not_evaluated_check(check_id, reason="missing_illumination_features")
    evidence = {
        "relative_luminance_range": score,
        "left_right_difference": features.left_right_difference,
        "top_bottom_difference": features.top_bottom_difference,
        "spatial_luminance_cv": features.spatial_luminance_cv,
        "valid_grid_cells": features.valid_grid_cells,
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    if retake is not None and score >= retake:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="nonuniform",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=score,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.STRONG_SHADOW.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if warning is not None and score >= warning:
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="mildly_nonuniform",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=score,
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.UNEVEN_LIGHTING.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="uniform",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=score,
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


def evaluate_resolution(
    features: InputGuardFeatures, policy: InputGuardPolicy
) -> CheckResult:
    check_id = CheckId.RESOLUTION.value
    cfg = policy.check_config(CheckId.RESOLUTION)
    retake_pixels = _thr(cfg, "retake_tongue_pixel_count")
    warning_pixels = _thr(cfg, "warning_tongue_pixel_count")
    retake_short = _thr(cfg, "retake_effective_short_side_px")
    warning_short = _thr(cfg, "warning_effective_short_side_px")
    pixels = features.tongue_pixel_count
    short_side = features.effective_short_side_px
    if pixels is None or short_side is None:
        return make_not_evaluated_check(check_id, reason="missing_resolution_features")
    evidence = {
        "tongue_pixel_count": pixels,
        "effective_short_side_px": short_side,
        "tight_bbox_width_px": features.tight_bbox_width_px,
        "tight_bbox_height_px": features.tight_bbox_height_px,
        "roi_width_px": features.roi_width_px,
        "roi_height_px": features.roi_height_px,
        "note": "engineering heuristic for V1 model design range; not clinical",
        "thresholds": cfg.get("thresholds"),
        **_policy_meta(policy),
    }
    if (retake_pixels is not None and pixels < retake_pixels) or (
        retake_short is not None and short_side < retake_short
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="too_low",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=float(pixels),
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.TONGUE_RESOLUTION_TOO_LOW.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    if (warning_pixels is not None and pixels < warning_pixels) or (
        warning_short is not None and short_side < warning_short
    ):
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="low",
            severity=Severity.MILD.value,
            decision_effect=Decision.WARNING.value,
            score=float(pixels),
            thresholds=cfg.get("thresholds"),
            evidence=evidence,
            reason_code=ReasonCode.IMAGE_RESOLUTION_TOO_LOW.value,
            source=EvidenceSource.SIGNAL_RULE.value,
        )
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="adequate",
        severity=Severity.NONE.value,
        decision_effect=Decision.PASS.value,
        score=float(pixels),
        thresholds=cfg.get("thresholds"),
        evidence=evidence,
        reason_code=None,
        source=EvidenceSource.SIGNAL_RULE.value,
    )


_EVALUATORS = {
    CheckId.TONGUE_PRESENCE: evaluate_tongue_presence,
    CheckId.TONGUE_SCALE: evaluate_tongue_scale,
    CheckId.TONGUE_COMPLETENESS: evaluate_tongue_completeness,
    CheckId.SEGMENTATION_INTEGRITY: evaluate_segmentation_integrity,
    CheckId.FOCUS: evaluate_focus,
    CheckId.EXPOSURE: evaluate_exposure,
    CheckId.ILLUMINATION_UNIFORMITY: evaluate_illumination,
    CheckId.RESOLUTION: evaluate_resolution,
}


def evaluate_signal_checks(
    features: InputGuardFeatures,
    policy: InputGuardPolicy,
) -> dict[str, CheckResult]:
    """评估全部 enabled checks；未实现项 not_evaluated。"""
    no_tongue = features.segmentation_status == "no_tongue_detected" or (
        features.tongue_pixel_count is not None and int(features.tongue_pixel_count) <= 0
    )
    checks: dict[str, CheckResult] = {}
    for check_id, meta in CHECK_DEFINITIONS.items():
        key = check_id.value
        if not policy.is_check_enabled(check_id):
            checks[key] = make_not_evaluated_check(key, reason="check_disabled")
            continue
        if check_id in DEFERRED_CHECKS or not meta.get("implemented", False):
            checks[key] = make_not_evaluated_check(
                key,
                reason=f"implementation_stage={meta.get('implementation_stage')}",
            )
            continue
        if no_tongue and meta.get("depends_on_roi") and check_id != CheckId.TONGUE_PRESENCE:
            checks[key] = make_not_evaluated_check(
                key, reason="no_tongue_roi_unavailable"
            )
            continue
        evaluator = _EVALUATORS.get(check_id)
        if evaluator is None:
            checks[key] = make_not_evaluated_check(key, reason="no_evaluator")
            continue
        checks[key] = evaluator(features, policy)
    return checks
