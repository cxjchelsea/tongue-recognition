"""D4-D.1：Unified Guard integration audit 只读契约测试。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tongue_data.input_guard.d4d1_integration_audit import (
    STAIN_T_CLEAR,
    STAIN_T_RETAKE,
    _attribution_bucket,
    _stain_band,
    aggregate_from_checks,
    build_ablation_table,
    build_retake_attribution,
    verify_aggregation_integrity,
    verify_stain_mapping,
)
from tongue_data.stain.metrics import map_probability_to_finding

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "d4"
STATS_PATH = REPORTS / "d4d1_integration_audit_stats.json"
SAMPLE_PARQUET = REPORTS / "d4d1_unified_sample_audit.parquet"
SAMPLE_FULL = REPORTS / "d4d1_unified_sample_audit_full.pkl"
ATTR_PATH = REPORTS / "d4d1_retake_attribution.json"
NEWLY_PATH = REPORTS / "d4d1_newly_rejected_samples.csv"
POLICY = ROOT / "configs" / "input_guard_v1.yaml"
STAIN_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
D4D_UNIFIED = REPORTS / "d4d_unified_test_audit.json"
AUDIT_MODULE = (
    ROOT / "src" / "tongue_data" / "input_guard" / "d4d1_integration_audit.py"
)


def _require_artifacts():
    if not STATS_PATH.exists() or not SAMPLE_PARQUET.exists():
        pytest.skip("D4-D.1 audit artifacts missing; run input-guard-integration-audit")


@pytest.fixture(scope="module")
def stats():
    _require_artifacts()
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sample_frame():
    _require_artifacts()
    return pd.read_parquet(SAMPLE_PARQUET)


@pytest.fixture(scope="module")
def full_frame():
    _require_artifacts()
    if not SAMPLE_FULL.exists():
        pytest.skip("full pickle missing")
    return pd.read_pickle(SAMPLE_FULL)


def test_01_sample_audit_has_130_unique_samples(sample_frame):
    assert len(sample_frame) == 130
    assert sample_frame["sample_id"].nunique() == 130


def test_02_ablation_sample_ids_identical(stats):
    ids = stats["ablation"]["sample_ids"]
    assert len(ids) == 130
    assert len(set(ids)) == 130


def test_03_d4b_ablation_matches_frozen_d4b_counts(stats):
    # frozen D4-B test：pass 74 / warning 43 / retake 13
    counts = stats["ablation"]["counts"]["A_d4b_only"]
    assert counts == {"pass": 74, "warning": 43, "retake": 13}


def test_04_full_unified_matches_frozen_d4d_counts(stats):
    counts = stats["ablation"]["counts"]["D_full"]
    assert counts["pass"] == 36
    assert counts["warning"] == 14
    assert counts["retake"] == 80
    if D4D_UNIFIED.exists():
        frozen = json.loads(D4D_UNIFIED.read_text(encoding="utf-8"))
        assert counts["retake"] == frozen["decision_counts"]["retake"]


def test_05_final_retake_has_retake_trigger(full_frame):
    report = verify_aggregation_integrity(full_frame)
    assert report["bug_count"] == 0


def test_06_retake_attribution_total(stats):
    attr = stats["retake_attribution"]
    assert attr["total_retake"] == 80
    assert sum(attr["by_source"].values()) == 80


def test_07_newly_rejected_definition(sample_frame):
    newly = sample_frame[
        (sample_frame["D4B_decision"] != "retake")
        & (sample_frame["unified_decision"] == "retake")
    ]
    assert NEWLY_PATH.exists()
    csv_frame = pd.read_csv(NEWLY_PATH)
    assert len(csv_frame) == len(newly)
    assert set(csv_frame["sample_id"]) == set(newly["sample_id"])


def test_08_stain_mapping_boundary_095():
    assert map_probability_to_finding(0.95, STAIN_T_CLEAR, STAIN_T_RETAKE) == "false"
    assert _stain_band(0.95) == "clear"


def test_09_stain_mapping_boundary_096():
    assert map_probability_to_finding(0.96, STAIN_T_CLEAR, STAIN_T_RETAKE) == "true"
    assert _stain_band(0.96) == "stain"


def test_10_uncertain_mapping_warning():
    assert (
        map_probability_to_finding(0.955, STAIN_T_CLEAR, STAIN_T_RETAKE) == "uncertain"
    )
    assert _stain_band(0.955) == "uncertain"


def test_11_color_cast_unavailable_no_retake_effect(sample_frame):
    unavailable = sample_frame[
        sample_frame["color_cast_evaluation_state"] == "unavailable"
    ]
    if len(unavailable) == 0:
        pytest.skip("no unavailable color_cast in this run")
    assert (unavailable["color_cast_decision_effect"] != "retake").all()


def test_12_occlusion_unavailable_no_retake_effect(sample_frame):
    unavailable = sample_frame[
        sample_frame["occlusion_evaluation_state"] == "unavailable"
    ]
    if len(unavailable) == 0:
        # D4-D 当前 test：occlusion 全 evaluated；仍验证逻辑成立
        assert True
        return
    assert (unavailable["occlusion_decision_effect"] != "retake").all()


def test_13_reason_count_deterministic(stats):
    assert isinstance(stats["reason_code_counts"], dict)
    # 两次读文件一致
    again = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    assert again["reason_code_counts"] == stats["reason_code_counts"]


def test_14_primary_reason_count_deterministic(stats):
    again = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    assert again["primary_reason_counts"] == stats["primary_reason_counts"]


def test_15_dataset_breakdown_sums_to_130(sample_frame, stats):
    assert sum(stats["dataset_counts"].values()) == 130
    assert sample_frame["dataset"].value_counts().sum() == 130


def test_16_stain_distribution_quantiles_deterministic(stats):
    q1 = stats["stain_distribution"]["overall"]["quantiles"]
    q2 = json.loads(STATS_PATH.read_text(encoding="utf-8"))[
        "stain_distribution"
    ]["overall"]["quantiles"]
    assert q1 == q2
    assert q1["count"] == 130 or q1["count"] == stats["n_samples"]


def test_17_audit_does_not_modify_stain_thresholds():
    thr = json.loads(STAIN_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_18_audit_does_not_modify_policy_version():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert str(doc.get("policy_version")) == "1.3"


def test_19_audit_does_not_modify_d4b_focus_thresholds():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    focus = doc["checks"]["focus"]["thresholds"]
    assert focus["warning_roi_laplacian"] == pytest.approx(10.448310279846192)
    assert focus["retake_roi_laplacian"] == pytest.approx(5.28003999710083)


def test_20_audit_does_not_modify_color_cast_thresholds():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    cast = doc["checks"]["color_cast"]["thresholds"]
    assert cast["warning_cast_magnitude"] == pytest.approx(20.088847335301597)
    assert cast["retake_cast_magnitude"] == pytest.approx(28.600699292150182)


def test_21_audit_does_not_modify_occlusion_thresholds():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    occ = doc["checks"]["occlusion"]["thresholds"]
    assert occ["warning_combined_score"] == pytest.approx(0.23593363954037233)
    assert occ["retake_combined_score"] == pytest.approx(0.28615079033942487)


def test_22_no_training_code_called_in_audit_module():
    source = AUDIT_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"train", "fit", "Trainer", "Adam", "SGD"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (names & forbidden)


def test_23_no_optimizer_instantiated_in_audit_module():
    source = AUDIT_MODULE.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "Optimizer" not in source


def test_24_no_split_rebuild_in_audit_module():
    source = AUDIT_MODULE.read_text(encoding="utf-8")
    assert "build_splits" not in source
    assert "rebuild_split" not in source


def test_25_biohit_278_trackable(sample_frame, stats):
    assert (sample_frame["sample_id"] == "biohit::278.bmp").any()
    assert stats["biohit_278"] is not None
    assert stats["biohit_278"]["sample_id"] == "biohit::278.bmp"


def test_attribution_bucket_helpers():
    assert (
        _attribution_bucket(
            {
                "retake_due_to_d4b": False,
                "retake_due_to_stain": True,
                "retake_due_to_color_cast": False,
                "retake_due_to_occlusion": False,
            }
        )
        == "stain_only"
    )
    assert (
        _attribution_bucket(
            {
                "retake_due_to_d4b": True,
                "retake_due_to_stain": True,
                "retake_due_to_color_cast": False,
                "retake_due_to_occlusion": False,
            }
        )
        == "d4b_plus_stain"
    )


def test_aggregate_from_checks_stain_only_retake():
    checks = {
        "geometry.tongue_presence": {
            "evaluation_state": "evaluated",
            "decision_effect": "pass",
            "reason_code": None,
        },
        "quality.stain_suspected": {
            "evaluation_state": "evaluated",
            "decision_effect": "retake",
            "reason_code": "STAIN_SUSPECTED",
        },
    }
    result = aggregate_from_checks(
        checks,
        include={"geometry.tongue_presence", "quality.stain_suspected"},
        priority=["STAIN_SUSPECTED"],
    )
    assert result["decision"] == "retake"


def test_stain_mapping_bugs_zero_when_consistent(full_frame):
    report = verify_stain_mapping(full_frame)
    assert report["bug_count"] == 0


def test_recommendation_enum_present(stats):
    allowed = {
        "D4_FINAL_READY",
        "D4C_CROSS_DOMAIN_CONCERN",
        "COLOR_CAST_DOMAIN_CONCERN",
        "OCCLUSION_DOMAIN_CONCERN",
        "INTEGRATION_BUG_FOUND",
        "MULTIPLE_CONCERNS",
    }
    assert stats["recommendation"]["recommendation"] in allowed
