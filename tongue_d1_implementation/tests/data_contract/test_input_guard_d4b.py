"""D4-B：信号规则 / 阈值边界 / partial runtime 测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tongue_data.input_guard.features import InputGuardFeatures
from tongue_data.input_guard.ontology import CheckId, Decision, EvaluationState, implemented_checks_count
from tongue_data.input_guard.policy import InputGuardPolicy
from tongue_data.input_guard.signal_checks import evaluate_signal_checks
from tongue_data.input_guard.signal_features import (
    compute_border_touch_stats,
    compute_exposure_features,
    compute_focus_features,
    compute_illumination_features,
    resize_long_side_gray,
    rgb_to_luminance,
    tenengrad_energy,
    variance_of_laplacian,
)
from tongue_data.input_guard.decision import aggregate_decision, build_result_from_check_effects
from tongue_data.input_guard.runtime import InputGuardRuntime
from tongue_data.input_guard.calibration import CALIBRATION_SPLITS, FORBIDDEN_SPLITS

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "input_guard_v1.yaml"


def _policy_with_thresholds(tmp_path: Path, overrides: dict) -> InputGuardPolicy:
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    doc["version"] = "1.1"
    doc["policy_version"] = "1.1"
    # 为已实现 checks 填最小可用阈值，避免 validation fail
    defaults = {
        "tongue_presence": {"needs_calibration": False, "thresholds": {}},
        "tongue_scale": {
            "needs_calibration": False,
            "thresholds": {
                "warning_foreground_ratio": 0.10,
                "retake_foreground_ratio": 0.05,
                "warning_bbox_width_ratio": 0.20,
                "retake_bbox_width_ratio": 0.10,
                "warning_bbox_height_ratio": 0.20,
                "retake_bbox_height_ratio": 0.10,
            },
        },
        "tongue_completeness": {
            "needs_calibration": False,
            "thresholds": {
                "warning_side_touch_ratio": 0.05,
                "retake_side_touch_ratio": 0.20,
                "warning_bottom_touch_ratio": 0.05,
                "retake_bottom_touch_ratio": 0.20,
                "warning_top_touch_ratio": 0.30,
            },
        },
        "segmentation_integrity": {
            "needs_calibration": False,
            "thresholds": {
                "warning_largest_component_ratio": 0.90,
                "retake_largest_component_ratio": 0.70,
                "warning_mean_probability": 0.50,
                "retake_component_count": 8,
            },
        },
        "focus": {
            "needs_calibration": False,
            "thresholds": {
                "warning_roi_laplacian": 50.0,
                "retake_roi_laplacian": 20.0,
            },
        },
        "exposure": {
            "needs_calibration": False,
            "thresholds": {
                "warning_dark_pixel_ratio": 0.40,
                "retake_dark_pixel_ratio": 0.70,
                "warning_bright_pixel_ratio": 0.40,
                "retake_bright_pixel_ratio": 0.70,
                "warning_shadow_clip_ratio": 0.20,
                "retake_shadow_clip_ratio": 0.50,
                "warning_highlight_clip_ratio": 0.20,
                "retake_highlight_clip_ratio": 0.50,
            },
        },
        "illumination_uniformity": {
            "needs_calibration": False,
            "thresholds": {
                "warning_relative_range": 0.40,
                "retake_relative_range": 0.80,
            },
        },
        "resolution": {
            "needs_calibration": False,
            "thresholds": {
                "warning_tongue_pixel_count": 5000,
                "retake_tongue_pixel_count": 2000,
                "warning_effective_short_side_px": 80,
                "retake_effective_short_side_px": 40,
            },
        },
        "color_cast": {"needs_calibration": True, "thresholds": {"warning": None, "retake": None}},
        "occlusion": {"needs_calibration": True, "thresholds": {"warning": None, "retake": None}},
        "stain_suspected": {
            "enabled": False,
            "needs_calibration": True,
            "thresholds": {"warning": None, "retake": None},
        },
    }
    for key, cfg in defaults.items():
        doc["checks"][key].update(cfg)
    for key, cfg in overrides.items():
        doc["checks"][key]["thresholds"].update(cfg.get("thresholds", {}))
        doc["checks"][key].update({k: v for k, v in cfg.items() if k != "thresholds"})
    path = tmp_path / "policy_test.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return InputGuardPolicy(path)


def _base_features(**kwargs) -> InputGuardFeatures:
    payload = dict(
        segmentation_status="success",
        original_width=200,
        original_height=200,
        foreground_ratio=0.25,
        tongue_pixel_count=10000,
        bbox_width_ratio=0.5,
        bbox_height_ratio=0.5,
        bbox_area_ratio=0.25,
        bbox_tight=(50, 50, 150, 150),
        tight_bbox_width_px=100,
        tight_bbox_height_px=100,
        effective_short_side_px=100,
        left_touch_ratio=0.0,
        right_touch_ratio=0.0,
        top_touch_ratio=0.0,
        bottom_touch_ratio=0.0,
        border_touch_ratio=0.0,
        touches_left=False,
        touches_right=False,
        touches_top=False,
        touches_bottom=False,
        component_count=1,
        largest_component_ratio=1.0,
        mean_foreground_probability=0.9,
        max_probability=0.99,
        roi_blur_score=100.0,
        blur_score=80.0,
        roi_gradient_energy=20.0,
        mean_luminance=120.0,
        dark_pixel_ratio=0.05,
        bright_pixel_ratio=0.05,
        shadow_clip_ratio=0.01,
        highlight_clip_ratio=0.01,
        relative_luminance_range=0.1,
        left_right_difference=5.0,
        top_bottom_difference=5.0,
        valid_grid_cells=9,
    )
    payload.update(kwargs)
    return InputGuardFeatures(**payload)


def test_no_tongue_retake_and_roi_not_evaluated(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(
        segmentation_status="no_tongue_detected",
        tongue_pixel_count=0,
        foreground_ratio=0.0,
        roi_blur_score=None,
        mean_luminance=None,
    )
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_PRESENCE.value].decision_effect == "retake"
    assert checks[CheckId.FOCUS.value].evaluation_state == "not_evaluated"
    assert checks[CheckId.EXPOSURE.value].finding is None


def test_scale_uses_tight_bbox_ratios_and_boundaries(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    # exact warning boundary: fg == warning → 不触发 < → pass
    features = _base_features(foreground_ratio=0.10, bbox_width_ratio=0.5, bbox_height_ratio=0.5)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect == "pass"
    # exact retake boundary: fg == retake → 不触发 retake，但仍可能 warning
    features = _base_features(foreground_ratio=0.05, bbox_width_ratio=0.5, bbox_height_ratio=0.5)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect != "retake"
    # just below retake on fg + width
    features = _base_features(foreground_ratio=0.049, bbox_width_ratio=0.09, bbox_height_ratio=0.5)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect == "retake"
    assert checks[CheckId.TONGUE_SCALE.value].evidence["bbox_source"] == "tight"
    # warning region
    features = _base_features(foreground_ratio=0.09, bbox_width_ratio=0.5, bbox_height_ratio=0.5)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect == "warning"


def test_top_border_not_auto_retake(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(top_touch_ratio=0.5, left_touch_ratio=0.0, right_touch_ratio=0.0, bottom_touch_ratio=0.0)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_COMPLETENESS.value].decision_effect != "retake"


def test_side_crop_retake_and_border_ratios():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, 0] = 1
    mask[10:90, 10:90] = 1
    stats = compute_border_touch_stats(mask)
    assert stats["touches_left"] is True
    assert stats["left_touch_ratio"] == pytest.approx(1.0)


def test_segmentation_integrity_and_probability_proxy(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(largest_component_ratio=0.60, component_count=3, mean_foreground_probability=0.4)
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.SEGMENTATION_INTEGRITY.value].decision_effect == "retake"
    assert "proxy" in checks[CheckId.SEGMENTATION_INTEGRITY.value].evidence["confidence_note"]


def test_blur_resize_and_sharp_gt_blur():
    rng = np.random.default_rng(0)
    sharp = np.zeros((128, 128, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    blurred = np.asarray(
        __import__("PIL").Image.fromarray(sharp).resize((32, 32)).resize((128, 128)),
        dtype=np.uint8,
    )
    sharp_score = variance_of_laplacian(resize_long_side_gray(sharp, 256))
    blur_score = variance_of_laplacian(resize_long_side_gray(blurred, 256))
    assert sharp_score > blur_score
    assert tenengrad_energy(resize_long_side_gray(sharp, 256)) > tenengrad_energy(
        resize_long_side_gray(blurred, 256)
    )
    focus = compute_focus_features(sharp, sharp, long_side=256)
    assert focus["blur_analysis_long_side"] == 256


def test_focus_threshold_boundaries(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    # exactly retake threshold → not < → pass/warning path
    features = _base_features(roi_blur_score=20.0)
    assert evaluate_signal_checks(features, policy)[CheckId.FOCUS.value].decision_effect != "retake"
    features = _base_features(roi_blur_score=19.9)
    assert evaluate_signal_checks(features, policy)[CheckId.FOCUS.value].decision_effect == "retake"
    features = _base_features(roi_blur_score=49.9)
    assert evaluate_signal_checks(features, policy)[CheckId.FOCUS.value].decision_effect == "warning"
    features = _base_features(roi_blur_score=50.0)
    assert evaluate_signal_checks(features, policy)[CheckId.FOCUS.value].decision_effect == "pass"


def test_luminance_and_exposure_ratios():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:] = (10, 10, 10)
    luma = rgb_to_luminance(rgb)
    assert luma.mean() == pytest.approx(0.2126 * 10 + 0.7152 * 10 + 0.0722 * 10)
    dark = compute_exposure_features(rgb, np.ones((10, 10), dtype=np.uint8))
    assert dark["dark_pixel_ratio"] == pytest.approx(1.0)
    bright_rgb = np.full((10, 10, 3), 250, dtype=np.uint8)
    bright = compute_exposure_features(bright_rgb, None)
    assert bright["highlight_clip_ratio"] > 0.9


def test_exposure_rules(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(dark_pixel_ratio=0.80, shadow_clip_ratio=0.60, bright_pixel_ratio=0.0, highlight_clip_ratio=0.0)
    assert evaluate_signal_checks(features, policy)[CheckId.EXPOSURE.value].finding == "underexposed"
    features = _base_features(bright_pixel_ratio=0.80, highlight_clip_ratio=0.60, dark_pixel_ratio=0.0, shadow_clip_ratio=0.0)
    assert evaluate_signal_checks(features, policy)[CheckId.EXPOSURE.value].finding == "overexposed"
    features = _base_features()
    assert evaluate_signal_checks(features, policy)[CheckId.EXPOSURE.value].finding == "normal"


def test_illumination_mask_aware_and_uniform(tmp_path):
    rgb = np.zeros((90, 90, 3), dtype=np.uint8)
    rgb[:, :30] = 20
    rgb[:, 30:60] = 120
    rgb[:, 60:] = 220
    mask = np.zeros((90, 90), dtype=np.uint8)
    mask[10:80, 10:80] = 1
    uneven = compute_illumination_features(rgb, mask)
    assert uneven["valid_grid_cells"] >= 2
    assert uneven["relative_luminance_range"] is not None
    # 空 cell 不把 unavailable 当 0：构造几乎空 mask
    tiny_mask = np.zeros((90, 90), dtype=np.uint8)
    tiny_mask[0, 0] = 1
    sparse = compute_illumination_features(rgb, tiny_mask, min_cell_pixels=16)
    assert sparse["relative_luminance_range"] is None

    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(relative_luminance_range=0.9)
    assert evaluate_signal_checks(features, policy)[CheckId.ILLUMINATION_UNIFORMITY.value].decision_effect == "retake"
    features = _base_features(relative_luminance_range=0.1)
    assert evaluate_signal_checks(features, policy)[CheckId.ILLUMINATION_UNIFORMITY.value].decision_effect == "pass"


def test_resolution_vs_scale_semantics(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    # 高像素但占比小：scale warning/retake，resolution pass
    features = _base_features(
        foreground_ratio=0.04,
        bbox_width_ratio=0.08,
        bbox_height_ratio=0.08,
        tongue_pixel_count=50000,
        effective_short_side_px=200,
    )
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect == "retake"
    assert checks[CheckId.RESOLUTION.value].decision_effect == "pass"
    # 低像素但占比大
    features = _base_features(
        foreground_ratio=0.4,
        bbox_width_ratio=0.6,
        bbox_height_ratio=0.6,
        tongue_pixel_count=1500,
        effective_short_side_px=30,
    )
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.TONGUE_SCALE.value].decision_effect == "pass"
    assert checks[CheckId.RESOLUTION.value].decision_effect == "retake"


def test_null_feature_not_zero_and_deferred_checks(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(color_cast_score=None, roi_blur_score=100.0)
    assert features.color_cast_score is None
    checks = evaluate_signal_checks(features, policy)
    assert checks[CheckId.COLOR_CAST.value].evaluation_state == "not_evaluated"
    assert checks[CheckId.OCCLUSION.value].evaluation_state == "not_evaluated"
    assert checks[CheckId.STAIN_SUSPECTED.value].evaluation_state == "not_evaluated"


def test_signal_rule_source_and_evidence_thresholds(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    checks = evaluate_signal_checks(_base_features(), policy)
    focus = checks[CheckId.FOCUS.value]
    assert focus.source == "signal_rule"
    assert focus.evidence.get("thresholds") is not None


def test_policy_v11_and_uncalibrated_implemented_fail(tmp_path):
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    doc["version"] = "1.1"
    doc["policy_version"] = "1.1"
    doc["checks"]["focus"]["needs_calibration"] = True
    doc["checks"]["focus"]["thresholds"] = {"warning": None, "retake": None}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="needs_calibration"):
        InputGuardPolicy(path)


def test_partial_pass_and_retake_semantics(tmp_path):
    policy = _policy_with_thresholds(tmp_path, {})
    checks = evaluate_signal_checks(_base_features(), policy)
    result = build_result_from_check_effects(
        checks=checks, policy=policy, evaluation_complete=False
    )
    # build_result_from_check_effects sets evaluation_complete from arg
    assert result.evaluation_complete is False
    # force retake presence
    features = _base_features(segmentation_status="no_tongue_detected", tongue_pixel_count=0, roi_blur_score=None, mean_luminance=None, relative_luminance_range=None)
    checks = evaluate_signal_checks(features, policy)
    result = build_result_from_check_effects(checks=checks, policy=policy, evaluation_complete=False)
    assert result.decision == "retake"
    assert result.usable is False


def test_aggregation_and_primary_reason(tmp_path):
    assert aggregate_decision(["pass", "warning", "retake"]) == Decision.RETAKE
    policy = _policy_with_thresholds(tmp_path, {})
    features = _base_features(roi_blur_score=10.0, foreground_ratio=0.04, bbox_width_ratio=0.08, bbox_height_ratio=0.5)
    checks = evaluate_signal_checks(features, policy)
    result = build_result_from_check_effects(checks=checks, policy=policy, evaluation_complete=False)
    assert result.decision == "retake"
    assert result.primary_reason is not None


def test_calibration_splits_forbid_test():
    assert "test" in FORBIDDEN_SPLITS
    assert "train" in CALIBRATION_SPLITS and "val" in CALIBRATION_SPLITS


def test_original_rgb_not_mutated_by_focus():
    rgb = np.random.default_rng(1).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    before = rgb.copy()
    compute_focus_features(rgb, rgb, long_side=32)
    assert np.array_equal(rgb, before)


def test_implemented_count_is_eight():
    assert implemented_checks_count() == 8


@pytest.mark.skipif(
    not (ROOT / "runs/segmentation/d3c/baseline/best.pt").exists(),
    reason="checkpoint missing",
)
def test_runtime_deterministic_same_image(tmp_path):
    # 需要已校准 policy；若仍为 1.0 且 needs_calibration，跳过
    try:
        policy = InputGuardPolicy(POLICY_PATH)
    except ValueError:
        pytest.skip("policy not yet calibrated to 1.1")
    if not str(policy.policy_version).startswith("1.1"):
        pytest.skip("policy not yet calibrated to 1.1")
    runtime = InputGuardRuntime(
        checkpoint_path=ROOT / "runs/segmentation/d3c/baseline/best.pt",
        data_config=ROOT / "configs/segmentation_v1.yaml",
        train_config=ROOT / "configs/segmentation_train_v1.yaml",
        policy_path=POLICY_PATH,
        device="cpu",
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[40:200, 60:260] = (180, 90, 90)
    one = runtime.evaluate(image, sample_id="det")
    two = runtime.evaluate(image, sample_id="det")
    assert one.decision == two.decision
    assert one.reason_codes == two.reason_codes
    assert one.evaluation_complete is False
    assert one.guard_ready is False
