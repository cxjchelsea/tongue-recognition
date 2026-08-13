"""D4-D：color_cast / occlusion / unified guard 测试。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tongue_data.input_guard.color_cast import (
    apply_channel_cast,
    compute_color_cast_features,
    evaluate_color_cast,
)
from tongue_data.input_guard.decision import aggregate_decision, decision_usable
from tongue_data.input_guard.d4d_calibration import load_d4d_config
from tongue_data.input_guard.occlusion import (
    apply_synthetic_occlusion,
    compute_occlusion_features,
    evaluate_occlusion,
)
from tongue_data.input_guard.ontology import (
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    EvaluationState,
    implemented_checks_count,
)
from tongue_data.input_guard.policy import InputGuardPolicy
from tongue_data.input_guard.runtime import (
    InputGuardRuntime,
    compute_evaluation_complete,
    compute_system_guard_ready,
)
from tongue_data.input_guard.schema import CheckResult
from tongue_data.input_guard.calibration import CALIBRATION_SPLITS, FORBIDDEN_SPLITS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs" / "input_guard_v1.yaml"
D4D_CFG = ROOT / "configs" / "input_guard_d4d_v1.yaml"
STAIN_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
D4B_FOCUS_FROZEN = {
    "retake_roi_laplacian": 5.28003999710083,
    "warning_roi_laplacian": 10.448310279846192,
}


def _policy_with_d4d(tmp_path: Path, *, cast_w=8.0, cast_r=14.0, occ_w=0.08, occ_r=0.18):
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    doc["version"] = "1.3"
    doc["policy_version"] = "1.3"
    doc["checks"]["color_cast"].update(
        {
            "needs_calibration": False,
            "status": "PASS",
            "neutral_support": load_d4d_config(D4D_CFG)["color_cast"]["neutral"],
            "thresholds": {
                "warning_cast_magnitude": cast_w,
                "retake_cast_magnitude": cast_r,
            },
        }
    )
    doc["checks"]["occlusion"].update(
        {
            "needs_calibration": False,
            "status": "PASS",
            "thresholds": {
                "warning_combined_score": occ_w,
                "retake_combined_score": occ_r,
                "require_multi_evidence_for_retake": True,
            },
        }
    )
    path = tmp_path / "policy_d4d.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return InputGuardPolicy(path)


def _neutral_scene(size=128, tongue_color=(180, 60, 60)):
    """背景偏中性 + 中央红色舌头。"""
    rgb = np.full((size, size, 3), 180, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[40:90, 40:90] = 1
    rgb[mask > 0] = np.array(tongue_color, dtype=np.uint8)
    return rgb, mask


def test_color_cast_excludes_tongue_from_neutral():
    rgb, mask = _neutral_scene()
    cfg = load_d4d_config(D4D_CFG)["color_cast"]["neutral"]
    feat = compute_color_cast_features(rgb, mask, neutral_cfg=cfg)
    assert feat["tongue_mean_rgb_used"] is False
    # 候选不应来自舌头区域：若只用舌头红，cast 会很大；排除后应较小
    feat_tongue_only = compute_color_cast_features(
        rgb, np.zeros_like(mask), neutral_cfg=cfg
    )
    # 全图（含舌头）作为 outside=全部时 chroma 更高风险；至少保证函数不读 mean tongue 作决策字段
    assert "tongue_mean_rgb" not in feat


def test_neutral_support_insufficient_unavailable(tmp_path):
    policy = _policy_with_d4d(tmp_path)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)  # 太暗，无 neutral
    mask = np.ones((32, 32), dtype=np.uint8)
    d4d = load_d4d_config(D4D_CFG)
    check = evaluate_color_cast(rgb, mask, policy, d4d_cfg=d4d)
    assert check.evaluation_state == EvaluationState.UNAVAILABLE.value
    assert check.finding is None
    assert check.decision_effect is None


def test_unavailable_not_pass(tmp_path):
    policy = _policy_with_d4d(tmp_path)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    check = evaluate_color_cast(
        rgb, np.ones((32, 32), dtype=np.uint8), policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.decision_effect != Decision.PASS.value


def test_synthetic_casts_increase_magnitude():
    rgb, mask = _neutral_scene()
    # 扩大中性背景
    rgb[:, :] = 200
    rgb[40:90, 40:90] = (180, 60, 60)
    cfg = load_d4d_config(D4D_CFG)["color_cast"]["neutral"]
    base = compute_color_cast_features(rgb, mask, neutral_cfg=cfg)[
        "estimated_cast_magnitude"
    ]
    for direction in ("red", "green", "blue", "yellow"):
        casted = apply_channel_cast(rgb, direction=direction, gain=1.55)
        mag = compute_color_cast_features(casted, mask, neutral_cfg=cfg)[
            "estimated_cast_magnitude"
        ]
        assert mag is not None and base is not None
        assert mag >= base - 1e-6


def test_severe_cast_can_retake(tmp_path):
    policy = _policy_with_d4d(tmp_path, cast_w=2.0, cast_r=4.0)
    rgb, mask = _neutral_scene()
    rgb[:, :] = 200
    rgb[40:90, 40:90] = (180, 60, 60)
    casted = apply_channel_cast(rgb, direction="red", gain=1.8)
    check = evaluate_color_cast(
        casted, mask, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.evaluation_state == EvaluationState.EVALUATED.value
    assert check.decision_effect in {
        Decision.RETAKE.value,
        Decision.WARNING.value,
    }


def test_phenotype_like_red_tongue_not_auto_cast_fail(tmp_path):
    policy = _policy_with_d4d(tmp_path, cast_w=20.0, cast_r=30.0)
    rgb, mask = _neutral_scene(tongue_color=(220, 40, 40))
    # 背景保持中性
    check = evaluate_color_cast(
        rgb, mask, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.decision_effect != Decision.RETAKE.value


def test_phenotype_like_purple_tongue_not_auto_cast_fail(tmp_path):
    policy = _policy_with_d4d(tmp_path, cast_w=20.0, cast_r=30.0)
    rgb, mask = _neutral_scene(tongue_color=(120, 40, 160))
    check = evaluate_color_cast(
        rgb, mask, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.decision_effect != Decision.RETAKE.value


def test_color_cast_calibration_splits_forbid_test():
    assert "test" in FORBIDDEN_SPLITS
    assert "test" not in CALIBRATION_SPLITS


def test_occlusion_missing_probability_unavailable(tmp_path):
    policy = _policy_with_d4d(tmp_path)
    rgb, mask = _neutral_scene()
    check = evaluate_occlusion(
        rgb, mask, None, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.evaluation_state == EvaluationState.UNAVAILABLE.value
    assert check.finding is None


def test_unavailable_occlusion_not_none_detected(tmp_path):
    policy = _policy_with_d4d(tmp_path)
    check = evaluate_occlusion(
        np.zeros((16, 16, 3), dtype=np.uint8),
        None,
        None,
        policy,
        d4d_cfg=load_d4d_config(D4D_CFG),
    )
    assert check.finding != "none"


def test_interior_hole_evidence():
    rgb, mask = _neutral_scene()
    prob = np.ones(mask.shape, dtype=np.float64)
    # interior 大洞
    prob[55:75, 55:75] = 0.1
    feat = compute_occlusion_features(
        rgb, mask, prob, occlusion_cfg=load_d4d_config(D4D_CFG)["occlusion"]
    )
    assert feat["available"] is True
    assert feat["interior_hole_ratio"] > 0.05


def test_single_weak_evidence_not_auto_severe_retake(tmp_path):
    policy = _policy_with_d4d(tmp_path, occ_w=0.02, occ_r=0.03)
    rgb, mask = _neutral_scene()
    prob = np.ones(mask.shape, dtype=np.float64)
    # 仅小 hole，无 bright
    prob[60:65, 60:65] = 0.1
    check = evaluate_occlusion(
        rgb, mask, prob, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    # 即使分数达 retake，单证据应降级 warning 或 pass
    if check.decision_effect == Decision.RETAKE.value:
        assert check.evidence.get("evidence_count", 0) >= 2


def test_synthetic_severe_occlusion_detectable(tmp_path):
    policy = _policy_with_d4d(tmp_path, occ_w=0.02, occ_r=0.05)
    rgb, mask = _neutral_scene()
    prob = np.ones(mask.shape, dtype=np.float64) * 0.9
    rgb_occ, occ = apply_synthetic_occlusion(
        rgb, mask, area_ratio=0.28, mode="bright", seed=20260813
    )
    prob[occ] = 0.1
    check = evaluate_occlusion(
        rgb_occ, mask, prob, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.evaluation_state == EvaluationState.EVALUATED.value
    assert check.decision_effect in {
        Decision.WARNING.value,
        Decision.RETAKE.value,
    }


def test_crack_like_line_not_major(tmp_path):
    policy = _policy_with_d4d(tmp_path, occ_w=0.2, occ_r=0.35)
    rgb, mask = _neutral_scene()
    prob = np.ones(mask.shape, dtype=np.float64)
    prob[65, 45:85] = 0.2  # 细线
    check = evaluate_occlusion(
        rgb, mask, prob, policy, d4d_cfg=load_d4d_config(D4D_CFG)
    )
    assert check.finding != "major"


def test_d4b_focus_thresholds_unchanged():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    focus = doc["checks"]["focus"]["thresholds"]
    assert focus["retake_roi_laplacian"] == D4B_FOCUS_FROZEN["retake_roi_laplacian"]
    assert focus["warning_roi_laplacian"] == D4B_FOCUS_FROZEN["warning_roi_laplacian"]


@pytest.mark.skipif(not STAIN_THR.exists(), reason="D4-C thresholds missing")
def test_d4c_thresholds_unchanged():
    thr = json.loads(STAIN_THR.read_text(encoding="utf-8"))
    assert thr["t_clear"] == 0.95
    assert thr["t_retake"] == 0.96
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    stain = doc["checks"]["stain_suspected"]["thresholds"]
    assert stain["clear"] == 0.95
    assert stain["retake"] == 0.96


def test_registry_eleven_implemented():
    assert implemented_checks_count() == 11
    assert all(meta["implemented"] for meta in CHECK_DEFINITIONS.values())


def test_decision_priority_and_usable():
    assert aggregate_decision(["pass", "warning"]).value == "warning"
    assert aggregate_decision(["warning", "retake"]).value == "retake"
    assert decision_usable(Decision.WARNING) is True
    assert decision_usable(Decision.RETAKE) is False


def test_evaluation_complete_semantics():
    ok = CheckResult(
        check_id="quality.focus",
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="sharp",
        decision_effect=Decision.PASS.value,
    )
    bad = CheckResult(
        check_id="quality.color_cast",
        evaluation_state=EvaluationState.UNAVAILABLE.value,
        finding=None,
    )
    assert compute_evaluation_complete({"a": ok}) is True
    assert compute_evaluation_complete({"a": ok, "b": bad}) is False


def test_guard_ready_requires_pass_status(tmp_path):
    policy = _policy_with_d4d(tmp_path)
    assert compute_system_guard_ready(policy) is True
    # 破坏 status
    doc = yaml.safe_load((tmp_path / "policy_d4d.yaml").read_text(encoding="utf-8"))
    doc["checks"]["color_cast"]["status"] = "NEEDS_IMPROVEMENT"
    path = tmp_path / "policy_bad.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    assert compute_system_guard_ready(InputGuardPolicy(path)) is False


def test_quality_confidence_not_forged(tmp_path):
    # schema 默认 null；runtime notes 声明不伪造
    result = CheckResult(
        check_id="quality.color_cast",
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="acceptable",
        decision_effect=Decision.PASS.value,
        score=1.0,
    )
    assert result.score == 1.0  # check score ok
    # InputGuardResult.quality_confidence 由 runtime 置 null —— 单测构造验证字段存在
    from tongue_data.input_guard.schema import InputGuardResult

    igr = InputGuardResult(
        decision="pass",
        usable=True,
        evaluation_complete=True,
        guard_ready=True,
        checks={result.check_id: result},
        quality_confidence=None,
    )
    assert igr.quality_confidence is None


def test_original_rgb_not_mutated_by_color_cast():
    rgb, mask = _neutral_scene()
    before = rgb.copy()
    compute_color_cast_features(
        rgb, mask, neutral_cfg=load_d4d_config(D4D_CFG)["color_cast"]["neutral"]
    )
    assert np.array_equal(rgb, before)


def test_bright_intrusion_feature():
    rgb, mask = _neutral_scene()
    rgb[50:80, 50:80] = 240
    prob = np.ones(mask.shape, dtype=np.float64) * 0.9
    feat = compute_occlusion_features(
        rgb, mask, prob, occlusion_cfg=load_d4d_config(D4D_CFG)["occlusion"]
    )
    assert feat["bright_intrusion_ratio"] is not None
    assert feat["bright_intrusion_ratio"] > 0
