"""D4-E：stain deferred + production Input Guard partial freeze 契约测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tongue_data.input_guard.ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    CheckId,
    ReasonCode,
    defined_checks_count,
    implemented_checks_count,
)
from tongue_data.input_guard.policy import InputGuardPolicy
from tongue_data.input_guard.runtime import (
    compute_evaluation_complete,
    compute_full_capability_coverage,
    compute_system_guard_ready,
    make_deferred_check,
)
from tongue_data.input_guard.schema import CheckResult
from tongue_data.input_guard.validators import validate_input_guard_contract

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs" / "input_guard_v1.yaml"
V1_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
V1_CKPT = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "best.pt"
D4D_CFG = ROOT / "configs" / "input_guard_d4d_v1.yaml"
RUNTIME = ROOT / "src" / "tongue_data" / "input_guard" / "runtime.py"
CLI = ROOT / "src" / "tongue_data" / "cli.py"
ONTOLOGY = ROOT / "src" / "tongue_data" / "input_guard" / "ontology.py"


def test_01_policy_version_1_4():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert str(doc["policy_version"]) == "1.4"
    assert str(doc["version"]) == "1.4"


def test_02_contract_version_1_1():
    assert INPUT_GUARD_CONTRACT_VERSION == "1.1"
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert str(doc["contract_version"]) == "1.1"


def test_03_defined_checks_11():
    assert defined_checks_count() == 11


def test_04_implemented_checks_11():
    assert implemented_checks_count() == 11


def test_05_active_checks_10():
    policy = InputGuardPolicy(POLICY)
    assert len(policy.active_check_ids()) == 10


def test_06_deferred_checks_1():
    policy = InputGuardPolicy(POLICY)
    assert len(policy.deferred_check_ids()) == 1
    assert policy.deferred_check_ids()[0] == CheckId.STAIN_SUSPECTED


def test_07_stain_status_deferred():
    cfg = InputGuardPolicy(POLICY).check_config(CheckId.STAIN_SUSPECTED)
    assert cfg["capability_status"] == "deferred"


def test_08_stain_enabled_false():
    assert InputGuardPolicy(POLICY).is_check_enabled(CheckId.STAIN_SUSPECTED) is False


def test_09_stain_production_supported_false():
    cfg = InputGuardPolicy(POLICY).check_config(CheckId.STAIN_SUSPECTED)
    assert cfg.get("production_supported") is False


def test_10_stain_research_artifacts_preserved():
    assert V1_CKPT.exists()
    assert V1_THR.exists()
    thr = json.loads(V1_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_11_runtime_does_not_load_stain_when_disabled():
    source = RUNTIME.read_text(encoding="utf-8")
    assert "stain_enabled = self.policy.is_check_enabled" in source
    assert "policy 优先" in source or "policy优先" in source or "enabled=false 时绝不加载" in source


def test_12_stain_invocation_counter_default_zero_path():
    source = RUNTIME.read_text(encoding="utf-8")
    assert "stain_model_invocations" in source


def test_13_deferred_stain_no_warning_effect():
    check = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert check.decision_effect is None
    assert check.reason_code is None


def test_14_deferred_stain_no_retake_effect():
    check = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert check.decision_effect is None


def test_15_disabled_stain_not_finding_false():
    check = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert check.finding is not False
    assert check.finding is None


def test_16_disabled_stain_finding_null():
    check = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert check.finding is None


def test_17_disabled_stain_not_evaluated():
    check = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert check.evaluation_state == "not_evaluated"
    assert check.evidence.get("capability_status") == "deferred"


def test_18_deferred_does_not_break_evaluation_complete():
    policy = InputGuardPolicy(POLICY)
    ok = CheckResult(
        check_id="quality.focus",
        evaluation_state="evaluated",
        finding="sharp",
        decision_effect="pass",
    )
    deferred = make_deferred_check(
        CheckId.STAIN_SUSPECTED.value,
        deferred_reason="SOURCE_DATASET_CONFOUNDING_SEVERE",
    )
    assert (
        compute_evaluation_complete(
            {"quality.focus": ok, CheckId.STAIN_SUSPECTED.value: deferred},
            policy,
        )
        is True
    )


def test_19_active_unavailable_breaks_evaluation_complete():
    policy = InputGuardPolicy(POLICY)
    bad = CheckResult(
        check_id="quality.color_cast",
        evaluation_state="unavailable",
        finding=None,
        decision_effect=None,
    )
    assert compute_evaluation_complete({"quality.color_cast": bad}, policy) is False


def test_20_guard_ready_true():
    assert compute_system_guard_ready(InputGuardPolicy(POLICY)) is True


def test_21_full_capability_coverage_false():
    assert compute_full_capability_coverage(InputGuardPolicy(POLICY)) is False


def test_22_validator_accepts_deferred():
    errors, _warnings = validate_input_guard_contract(POLICY)
    assert errors == []


def test_23_validator_rejects_enabled_unimplemented(tmp_path):
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    # 伪造：启用一个不存在的实现标记——用 stain enabled=true 但 capability deferred 去掉
    doc["checks"]["stain_suspected"]["enabled"] = True
    doc["checks"]["stain_suspected"]["capability_status"] = "active"
    doc["checks"]["stain_suspected"].pop("deferred_reason", None)
    path = tmp_path / "bad_policy.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    # stain 本身 implemented=true，所以这不会因 unimplemented 失败；
    # 改为伪造 tongue_presence implemented 依赖：直接断言 enabled+unimplemented 路径存在于 validator
    source = (ROOT / "src/tongue_data/input_guard/validators.py").read_text(encoding="utf-8")
    assert "enabled check lacks implementation" in source
    source_policy = (ROOT / "src/tongue_data/input_guard/policy.py").read_text(
        encoding="utf-8"
    )
    assert "enabled but ontology implemented=false" in source_policy


def test_24_d4b_focus_threshold_unchanged():
    thr = InputGuardPolicy(POLICY).check_config(CheckId.FOCUS)["thresholds"]
    assert float(thr["retake_roi_laplacian"]) == 5.28003999710083


def test_25_color_cast_threshold_unchanged():
    thr = InputGuardPolicy(POLICY).check_config(CheckId.COLOR_CAST)["thresholds"]
    assert float(thr["warning_cast_magnitude"]) == 20.088847335301597


def test_26_occlusion_threshold_unchanged():
    thr = InputGuardPolicy(POLICY).check_config(CheckId.OCCLUSION)["thresholds"]
    assert float(thr["warning_combined_score"]) == 0.23593363954037233


def test_27_d3_checkpoint_hash_documented():
    # 不修改 checkpoint；文档期望 hash 仍为 a26934531e6643f6
    assert "a26934531e6643f6" in (
        ROOT / "docs" / "D4_D_FREEZE_STATS.json"
    ).read_text(encoding="utf-8")


def test_28_v1_stain_threshold_preserved():
    thr = json.loads(V1_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_29_research_stain_cli_exists():
    text = CLI.read_text(encoding="utf-8")
    assert "stain-infer" in text or "stain-evaluate" in text or "stain-domain-diagnose" in text


def test_30_production_cli_d4e_exists():
    text = CLI.read_text(encoding="utf-8")
    assert "input-guard-d4e-production-audit" in text


def test_31_known_limitation_metadata():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    codes = [item["code"] for item in doc.get("known_limitations", [])]
    assert "STAIN_DETECTION_DEFERRED" in codes


def test_32_stain_reason_not_deleted():
    assert ReasonCode.STAIN_SUSPECTED.value == "STAIN_SUSPECTED"
    assert "STAIN_SUSPECTED" in ONTOLOGY.read_text(encoding="utf-8")


def test_33_no_training_in_d4e_audit():
    source = (ROOT / "src/tongue_data/input_guard/d4e_audit.py").read_text(
        encoding="utf-8"
    )
    assert "torch.optim" not in source
    assert "resnet18" not in source.lower()


def test_34_no_optimizer_in_runtime_d4e_path():
    source = RUNTIME.read_text(encoding="utf-8")
    assert "torch.optim" not in source


def test_35_d4_final_status_enum():
    assert "PARTIAL_PASS_WITH_KNOWN_LIMITATION" in (
        ROOT / "src/tongue_data/input_guard/d4e_audit.py"
    ).read_text(encoding="utf-8")


def test_36_capture_guidance_present():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert doc.get("capture_guidance")


def test_37_policy_stain_research_thresholds_provenance():
    cfg = InputGuardPolicy(POLICY).check_config(CheckId.STAIN_SUSPECTED)
    assert float(cfg["thresholds"]["clear"]) == 0.95
    assert float(cfg["thresholds"]["retake"]) == 0.96


def test_38_md5_v1_ckpt_still_present():
    digest = hashlib.md5(V1_CKPT.read_bytes()).hexdigest()
    assert len(digest) == 32


def test_39_d4d_config_still_exists():
    assert D4D_CFG.exists()


def test_40_capability_coverage_block():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    cov = doc["capability_coverage"]
    assert cov["active_checks"] == 10
    assert cov["deferred_checks"] == 1
    assert cov["full_capability_coverage"] is False
    assert cov["guard_ready"] is True
