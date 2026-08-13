"""D4-A：Input Guard Contract / Ontology / Decision schema 测试。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tongue_data.input_guard.decision import (
    aggregate_decision,
    build_contract_skeleton_result,
    build_result_from_check_effects,
    select_primary_reason,
)
from tongue_data.input_guard.features import (
    InputGuardFeatures,
    features_from_segmentation_result,
)
from tongue_data.input_guard.guidance import FALLBACK_GUIDANCE, guidance_for_reason
from tongue_data.input_guard.ontology import (
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    EvaluationState,
    ReasonCode,
    Severity,
    assert_not_phenotype_as_qc_reason,
    defined_checks_count,
    implemented_checks_count,
    parse_decision,
    parse_reason_code,
    parse_severity,
)
from tongue_data.input_guard.policy import InputGuardPolicy
from tongue_data.input_guard.schema import CheckResult, InputGuardResult
from tongue_data.input_guard.validators import validate_input_guard_contract

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "input_guard_v1.yaml"


@dataclass
class FakeSegResult:
    status: str = "success"
    sample_id: str | None = "demo"
    original_width: int = 1000
    original_height: int = 500
    threshold: float = 0.5
    mask_foreground_pixels: int = 100000
    mask_foreground_ratio: float = 0.2
    component_count: int = 1
    largest_component_ratio: float = 1.0
    bbox_tight: tuple[int, int, int, int] | None = (100, 50, 700, 400)
    bbox_roi: tuple[int, int, int, int] | None = (70, 32, 730, 418)
    roi_size: tuple[int, int] | None = (660, 386)
    tongue_roi_mask: object | None = None
    mean_foreground_probability: float | None = 0.91
    max_probability: float | None = 0.99
    warnings: list[str] | None = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _evaluated(
    check_id: str,
    *,
    effect: str,
    reason: str | None = None,
    finding: str = "x",
    severity: str = "mild",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding=finding,
        severity=severity,
        decision_effect=effect,
        reason_code=reason,
        source="signal_rule",
    )


def test_pass_usable_true():
    result = build_result_from_check_effects(
        checks={
            "quality.focus": _evaluated(
                "quality.focus", effect="pass", reason=None, finding="sharp", severity="none"
            )
        },
        evaluation_complete=True,
    )
    # pass effect with no reason_code is ok for PASS
    assert result.decision == "pass"
    assert result.usable is True


def test_warning_usable_true():
    result = build_result_from_check_effects(
        checks={
            "quality.tongue_scale": _evaluated(
                "quality.tongue_scale",
                effect="warning",
                reason="TONGUE_SLIGHTLY_SMALL",
                finding="small",
            )
        }
    )
    assert result.decision == "warning"
    assert result.usable is True


def test_retake_usable_false():
    result = build_result_from_check_effects(
        checks={
            "quality.focus": _evaluated(
                "quality.focus",
                effect="retake",
                reason="TONGUE_BLUR",
                finding="blurred",
                severity="severe",
            )
        }
    )
    assert result.decision == "retake"
    assert result.usable is False


def test_decision_priority_retake_over_warning_over_pass():
    assert aggregate_decision(["pass", "warning", "retake"]) == Decision.RETAKE
    assert aggregate_decision(["pass", "warning"]) == Decision.WARNING
    assert aggregate_decision(["pass", "pass"]) == Decision.PASS


def test_multiple_warnings_aggregate_warning():
    result = build_result_from_check_effects(
        checks={
            "quality.tongue_scale": _evaluated(
                "quality.tongue_scale",
                effect="warning",
                reason="TONGUE_SLIGHTLY_SMALL",
            ),
            "quality.focus": _evaluated(
                "quality.focus", effect="warning", reason="IMAGE_BLUR"
            ),
        }
    )
    assert result.decision == "warning"


def test_one_retake_with_passes():
    result = build_result_from_check_effects(
        checks={
            "quality.focus": _evaluated(
                "quality.focus", effect="retake", reason="TONGUE_BLUR", severity="severe"
            ),
            "quality.exposure": _evaluated(
                "quality.exposure", effect="pass", finding="normal", severity="none"
            ),
            "quality.resolution": _evaluated(
                "quality.resolution", effect="pass", finding="adequate", severity="none"
            ),
        }
    )
    assert result.decision == "retake"


def test_primary_reason_priority_deterministic():
    policy = InputGuardPolicy(POLICY_PATH)
    primary = select_primary_reason(
        ["TONGUE_TOO_SMALL", "NO_TONGUE_DETECTED", "IMAGE_BLUR"],
        priority=policy.primary_reason_priority,
        retake_reasons={"TONGUE_TOO_SMALL", "NO_TONGUE_DETECTED", "IMAGE_BLUR"},
    )
    assert primary == "NO_TONGUE_DETECTED"


def test_unknown_reason_fail_fast():
    with pytest.raises(ValueError, match="unknown reason"):
        parse_reason_code("RED_TONGUE_BAD")


def test_unknown_decision_fail_fast():
    with pytest.raises(ValueError, match="unknown decision"):
        parse_decision("fail")


def test_unknown_severity_fail_fast():
    with pytest.raises(ValueError, match="unknown severity"):
        parse_severity("critical")


def test_missing_feature_stays_null_not_zero():
    features = InputGuardFeatures()
    assert features.blur_score is None
    assert features.mean_luminance is None
    assert features.dark_pixel_ratio is None
    assert 0 not in (
        features.blur_score,
        features.roi_blur_score,
        features.color_cast_score,
    )


def test_not_evaluated_not_treated_as_pass():
    checks = {
        "quality.focus": CheckResult(
            check_id="quality.focus",
            evaluation_state=EvaluationState.NOT_EVALUATED.value,
            finding=None,
            decision_effect=None,
        ),
        "quality.exposure": CheckResult(
            check_id="quality.exposure",
            evaluation_state=EvaluationState.NOT_EVALUATED.value,
            finding=None,
            decision_effect=None,
        ),
    }
    with pytest.raises(ValueError):
        CheckResult(
            check_id="quality.focus",
            evaluation_state="not_evaluated",
            finding="sharp",
            decision_effect="pass",
        ).validate()
    result = build_result_from_check_effects(checks=checks)
    # 无 evaluated effect → PASS skeleton，但 evaluation 未完成语义由调用方设置
    assert result.decision == "pass"
    assert all(
        check.evaluation_state == "not_evaluated" for check in result.checks.values()
    )


def test_no_tongue_detected_retake_and_roi_checks_not_evaluated():
    policy = InputGuardPolicy(POLICY_PATH)
    seg = FakeSegResult(
        status="no_tongue_detected",
        bbox_tight=None,
        bbox_roi=None,
        roi_size=None,
        mask_foreground_ratio=0.0,
        mask_foreground_pixels=0,
        mean_foreground_probability=None,
    )
    result = build_contract_skeleton_result(seg, policy)
    assert result.decision == "retake"
    assert result.usable is False
    assert result.primary_reason == "NO_TONGUE_DETECTED"
    assert result.evaluation_complete is False
    # ROI 依赖 check 不得 PASS
    focus = result.checks[CheckId.FOCUS.value]
    assert focus.evaluation_state == "not_evaluated"
    assert focus.finding is None
    presence = result.checks[CheckId.TONGUE_PRESENCE.value]
    assert presence.evaluation_state == "evaluated"
    assert presence.finding == "absent"


def test_d3e_feature_adapter_ratios_and_tight_bbox():
    import numpy as np

    seg = FakeSegResult(
        bbox_tight=(100, 50, 700, 400),
        bbox_roi=(0, 0, 1000, 500),  # margin 故意不同
        tongue_roi_mask=np.ones((386, 660), dtype=np.uint8),
    )
    features = features_from_segmentation_result(seg)
    assert features.foreground_ratio == pytest.approx(0.2)
    # tight: w=600, h=350
    assert features.bbox_width_ratio == pytest.approx(0.6)
    assert features.bbox_height_ratio == pytest.approx(0.7)
    assert features.bbox_area_ratio == pytest.approx(0.42)
    # 必须用 tight，而非 ROI margin 全图
    assert features.bbox_area_ratio != pytest.approx(1.0)
    assert features.touches_left is False
    assert features.touches_top is False
    assert features.component_count == 1
    assert features.largest_component_ratio == pytest.approx(1.0)
    assert features.mean_foreground_probability == pytest.approx(0.91)
    assert features.roi_width_px == 660
    assert features.roi_height_px == 386
    assert features.blur_score is None


def test_touches_border_from_tight_bbox():
    seg = FakeSegResult(bbox_tight=(0, 10, 100, 500), original_width=1000, original_height=500)
    features = features_from_segmentation_result(seg)
    assert features.touches_left is True
    assert features.touches_bottom is True
    assert features.touches_image_border is True


def test_reason_guidance_and_fallback():
    text = guidance_for_reason("TONGUE_BLUR")
    assert "对焦" in text or "稳定" in text
    assert guidance_for_reason("NOT_A_REAL_REASON") == FALLBACK_GUIDANCE


def test_phenotype_labels_cannot_be_qc_reason():
    with pytest.raises(ValueError):
        assert_not_phenotype_as_qc_reason("red_tongue")
    with pytest.raises(ValueError):
        assert_not_phenotype_as_qc_reason("yellow_coating")
    with pytest.raises(ValueError):
        assert_not_phenotype_as_qc_reason("crack")
    with pytest.raises(ValueError):
        assert_not_phenotype_as_qc_reason("toothmark")
    # stain 可以
    parse_reason_code("STAIN_SUSPECTED")


def test_disabled_and_unimplemented_checks_do_not_affect_decision():
    policy = InputGuardPolicy(POLICY_PATH)
    seg = FakeSegResult()
    result = build_contract_skeleton_result(seg, policy)
    assert result.decision == "pass"
    assert result.evaluation_complete is False
    assert result.guard_ready is False
    # stain disabled → not_evaluated
    stain = result.checks[CheckId.STAIN_SUSPECTED.value]
    assert stain.evaluation_state == "not_evaluated"
    # unimplemented focus 不影响
    assert result.checks[CheckId.FOCUS.value].decision_effect is None


def test_policy_unknown_check_fail_fast(tmp_path: Path):
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    doc["checks"]["not_a_real_check"] = {"enabled": True, "needs_calibration": True}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown QC check"):
        InputGuardPolicy(bad)


def test_policy_unknown_reason_fail_fast(tmp_path: Path):
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    doc["primary_reason_priority"].append("NOT_REGISTERED_REASON")
    bad = tmp_path / "bad_reason.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown reason"):
        InputGuardPolicy(bad)


def test_threshold_null_without_needs_calibration_fails(tmp_path: Path):
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    doc["checks"]["focus"]["needs_calibration"] = False
    doc["checks"]["focus"]["thresholds"] = {"warning": None, "retake": None}
    bad = tmp_path / "bad_threshold.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="needs_calibration"):
        InputGuardPolicy(bad)


def test_result_schema_serializable_and_check_order_independent():
    checks_a = {
        "quality.focus": _evaluated(
            "quality.focus", effect="warning", reason="IMAGE_BLUR"
        ),
        "quality.tongue_scale": _evaluated(
            "quality.tongue_scale",
            effect="retake",
            reason="TONGUE_TOO_SMALL",
            severity="severe",
        ),
    }
    checks_b = {
        "quality.tongue_scale": checks_a["quality.tongue_scale"],
        "quality.focus": checks_a["quality.focus"],
    }
    result_a = build_result_from_check_effects(checks=checks_a)
    result_b = build_result_from_check_effects(checks=checks_b)
    assert result_a.decision == result_b.decision == "retake"
    payload = result_a.to_dict()
    assert payload["usable"] is False
    assert "checks" in payload
    assert payload["quality_confidence"] is None


def test_policy_and_contract_validate_ok():
    errors, warnings = validate_input_guard_contract(POLICY_PATH)
    assert errors == []
    assert defined_checks_count() == 11
    # D4-C 后 ontology 标记 9 项 implemented；policy 版本随校准更新
    assert implemented_checks_count() == 11
    assert len(CHECK_DEFINITIONS) == 11
    policy = InputGuardPolicy(POLICY_PATH)
    assert policy.version in {"1.0", "1.1", "1.2", "1.3"}


def test_retake_requires_reason_on_check():
    with pytest.raises(ValueError, match="RETAKE requires reason_code"):
        CheckResult(
            check_id="quality.focus",
            evaluation_state="evaluated",
            finding="blurred",
            decision_effect="retake",
            reason_code=None,
        ).validate()


def test_input_guard_result_pass_usable_invariant():
    with pytest.raises(ValueError, match="PASS requires usable=true"):
        InputGuardResult(
            decision="pass",
            usable=False,
            evaluation_complete=False,
            checks={},
        ).validate()
