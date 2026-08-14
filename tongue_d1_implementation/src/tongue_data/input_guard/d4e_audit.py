"""D4-E：production unified audit（stain disabled）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    CheckId,
    ReasonCode,
    defined_checks_count,
    implemented_checks_count,
)
from .policy import load_input_guard_policy
from .runtime import (
    InputGuardRuntime,
    compute_full_capability_coverage,
    compute_system_guard_ready,
)


STAIN_ID = CheckId.STAIN_SUSPECTED.value
D3_HASH = "a26934531e6643f6"
FOCUS_RETAKE = 5.28003999710083
CAST_WARN = 20.088847335301597
OCC_WARN = 0.23593363954037233


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_d4e_production_unified_audit(
    *,
    checkpoint_path: str | Path = "runs/segmentation/d3c/baseline/best.pt",
    segmentation_dir: str | Path = "data/segmentation/v1",
    data_config_path: str | Path = "configs/segmentation_v1.yaml",
    train_config_path: str | Path = "configs/segmentation_train_v1.yaml",
    policy_path: str | Path = "configs/input_guard_v1.yaml",
    output_path: str | Path = "reports/d4/d4e_production_unified_audit.json",
    # 故意传入 research checkpoint：验证 policy 仍禁用
    stain_checkpoint: str | Path | None = "runs/input_guard/d4c/stain/best.pt",
    stain_thresholds: str | Path | None = "runs/input_guard/d4c/stain/thresholds.json",
    device: str = "auto",
) -> dict[str, Any]:
    from tongue_data.segmentation.inference import load_rgb_image

    policy = load_input_guard_policy(policy_path)
    if str(policy.policy_version).split("-")[0] != "1.4":
        raise RuntimeError(f"D4-E requires policy 1.4, got {policy.policy_version}")
    if not policy.is_check_deferred(CheckId.STAIN_SUSPECTED):
        raise RuntimeError("stain_suspected must be deferred")
    if policy.is_check_enabled(CheckId.STAIN_SUSPECTED):
        raise RuntimeError("stain_suspected must be enabled=false")

    # frozen thresholds sanity
    focus = policy.check_config(CheckId.FOCUS)["thresholds"]
    if float(focus["retake_roi_laplacian"]) != FOCUS_RETAKE:
        raise RuntimeError("D4-B focus threshold drifted")
    cast = policy.check_config(CheckId.COLOR_CAST)["thresholds"]
    if float(cast["warning_cast_magnitude"]) != CAST_WARN:
        raise RuntimeError("color_cast threshold drifted")
    occ = policy.check_config(CheckId.OCCLUSION)["thresholds"]
    if float(occ["warning_combined_score"]) != OCC_WARN:
        raise RuntimeError("occlusion threshold drifted")

    manifest = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    test = manifest[manifest["split"].astype(str) == "test"].copy()
    if len(test) != 130:
        raise RuntimeError(f"expected 130 test samples, got {len(test)}")

    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
    )
    if runtime.stain_detector is not None:
        raise RuntimeError("production runtime must not load stain detector")

    decision_counts = {"pass": 0, "warning": 0, "retake": 0}
    eval_complete = {"true": 0, "false": 0}
    eval_false_reasons: dict[str, int] = {}
    per_check_finding: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, int] = {}
    rows = []
    stain_warning = 0
    stain_retake = 0
    biohit_278 = None

    for _index, row in test.sort_values("sample_id").iterrows():
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        rgb, _mode = load_rgb_image(str(row["image_path"]))
        result = runtime.evaluate(rgb, sample_id=sample_id)
        decision_counts[result.decision] = decision_counts.get(result.decision, 0) + 1
        eval_complete["true" if result.evaluation_complete else "false"] += 1

        # deferred stain 不得成为 incomplete 原因
        stain_check = result.checks[STAIN_ID]
        if stain_check.finding is not None:
            raise RuntimeError("deferred stain must have finding=null")
        if stain_check.decision_effect is not None:
            raise RuntimeError("deferred stain must have decision_effect=null")
        if ReasonCode.STAIN_SUSPECTED.value in result.reason_codes:
            raise RuntimeError("STAIN_SUSPECTED must not appear in production reasons")

        if not result.evaluation_complete:
            for check_id, check in result.checks.items():
                if not policy.is_check_enabled(check_id):
                    continue
                if check.evaluation_state != "evaluated":
                    key = f"{check_id.split('.', 1)[-1]}_{check.evaluation_state}"
                    eval_false_reasons[key] = eval_false_reasons.get(key, 0) + 1

        for code in result.reason_codes:
            reason_counts[code] = reason_counts.get(code, 0) + 1
            if code == ReasonCode.STAIN_SUSPECTED.value:
                if result.decision == "retake":
                    stain_retake += 1
                if result.decision == "warning":
                    stain_warning += 1

        for check_id, check in result.checks.items():
            short = check_id.split(".", 1)[-1]
            bucket = per_check_finding.setdefault(
                short, {"evaluated": 0, "unavailable": 0, "not_evaluated": 0}
            )
            bucket[check.evaluation_state if check.evaluation_state in bucket else "not_evaluated"] = (
                bucket.get(
                    check.evaluation_state
                    if check.evaluation_state in bucket
                    else "not_evaluated",
                    0,
                )
                + 1
            )

        # aggregation regression：RETAKE 必须有 active reason
        if result.decision == "retake" and not result.reason_codes:
            raise RuntimeError(f"RETAKE without active reason: {sample_id}")

        row_out = {
            "sample_id": sample_id,
            "dataset": dataset,
            "decision": result.decision,
            "evaluation_complete": result.evaluation_complete,
            "guard_ready": result.guard_ready,
            "full_capability_coverage": result.full_capability_coverage,
            "primary_reason": result.primary_reason,
            "reason_codes": list(result.reason_codes),
            "stain_finding": stain_check.finding,
            "stain_evaluation_state": stain_check.evaluation_state,
            "stain_decision_effect": stain_check.decision_effect,
            "stain_model_invocations_cumulative": result.stain_model_invocations,
        }
        rows.append(row_out)
        if sample_id == "biohit::278.bmp" or sample_id.endswith("278.bmp") or "278" in sample_id:
            if "biohit" in sample_id.lower() and "278" in sample_id:
                biohit_278 = row_out

    if runtime.stain_model_invocations != 0:
        raise RuntimeError(
            f"stain_model_invocations must be 0, got {runtime.stain_model_invocations}"
        )
    if stain_retake != 0 or stain_warning != 0:
        raise RuntimeError("stain triggered warning/retake must be 0")

    report = {
        "stage": "D4-E",
        "policy_version": policy.policy_version,
        "contract_version": INPUT_GUARD_CONTRACT_VERSION,
        "samples": int(len(test)),
        "decision_counts": decision_counts,
        "pass": decision_counts.get("pass", 0),
        "warning": decision_counts.get("warning", 0),
        "retake": decision_counts.get("retake", 0),
        "active_checks": [item.value for item in policy.active_check_ids()],
        "deferred_checks": [item.value for item in policy.deferred_check_ids()],
        "defined_checks": defined_checks_count(),
        "implemented_checks": implemented_checks_count(),
        "per_check_finding_counts": per_check_finding,
        "reason_counts": reason_counts,
        "evaluation_complete": eval_complete,
        "evaluation_complete_false_reasons": eval_false_reasons,
        "stain_model_invocations": int(runtime.stain_model_invocations),
        "stain_triggered_warning": stain_warning,
        "stain_triggered_retake": stain_retake,
        "stain_decision_contribution": 0,
        "guard_ready": compute_system_guard_ready(policy),
        "full_capability_coverage": compute_full_capability_coverage(policy),
        "known_limitations": list(policy.doc.get("known_limitations") or []),
        "expected_retake_near_d4b": 13,
        "biohit_278": biohit_278,
        "sample_rows": rows,
        "threshold_retuned_on_test": False,
        "cnn_training_performed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 压缩写入：sample_rows 保留；大文件可接受
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def write_d4_final_docs(
    *,
    audit: dict[str, Any],
    docs_dir: str | Path = "docs",
    pytest_result: str = "pending",
    validator_result: str = "pending",
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "D4_E_STAIN_DEFERRED_CONTRACT.md").write_text(
        "\n".join(
            [
                "# D4-E Stain Deferred Contract",
                "",
                "- capability: `quality.stain_suspected`",
                "- status: **deferred**",
                "- implemented: true",
                "- production_supported: false",
                "- enabled: false",
                "- reason: `SOURCE_DATASET_CONFOUNDING_SEVERE`",
                "- research artifacts: preserved (v1/v2/v3 + D4-C.1-A/B/C/D)",
                "- production decision_effect: **none**",
                "- disabled ≠ finding=false",
                "- evaluation_state: not_evaluated / deferred metadata",
                "",
                "## Capture guidance",
                "",
                "> 拍摄前请尽量避免近期摄入可能明显改变舌面颜色的有色食物、饮料、药物或漱口液；",
                "> 当前版本暂不具备可靠的外源染色自动识别能力。",
                "",
                "## Downstream D5 note",
                "",
                "coating color phenotype 不得解释为已自动排除外源染色。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    stats = {
        "stage": "D4-E",
        "d4_final_status": "PARTIAL_PASS_WITH_KNOWN_LIMITATION",
        "input_guard_contract_version": INPUT_GUARD_CONTRACT_VERSION,
        "input_guard_policy_version": "1.4",
        "defined_checks": 11,
        "implemented_checks": 11,
        "active_checks": 10,
        "production_supported_checks": 10,
        "deferred_checks": 1,
        "deferred_capabilities": ["stain_suspected"],
        "deferred_reason": "source_dataset_severe_confounding",
        "guard_ready": True,
        "full_capability_coverage": False,
        "stain_production_enabled": False,
        "stain_research_model_preserved": True,
        "stain_model_invocations": audit.get("stain_model_invocations", 0),
        "unified_test_samples": audit.get("samples"),
        "unified_pass": audit.get("pass"),
        "unified_warning": audit.get("warning"),
        "unified_retake": audit.get("retake"),
        "stain_triggered_retake": 0,
        "stain_triggered_warning": 0,
        "evaluation_complete_true": audit.get("evaluation_complete", {}).get("true"),
        "evaluation_complete_false": audit.get("evaluation_complete", {}).get("false"),
        "known_limitations": ["STAIN_DETECTION_DEFERRED"],
        "d3_checkpoint_hash_expected": D3_HASH,
        "d4b_thresholds_unchanged": True,
        "color_cast_thresholds_unchanged": True,
        "occlusion_thresholds_unchanged": True,
        "pytest_result": pytest_result,
        "validator_result": validator_result,
        "biohit_278": audit.get("biohit_278"),
    }
    (docs_dir / "D4_FINAL_FREEZE_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = "\n".join(
        [
            "# D4 Final Freeze Report",
            "",
            f"**D4_FINAL_STATUS = `PARTIAL_PASS_WITH_KNOWN_LIMITATION`**",
            "",
            "## Capability table",
            "",
            "| Check | Status |",
            "|---|---|",
            "| tongue_presence | ACTIVE |",
            "| tongue_scale | ACTIVE |",
            "| tongue_completeness | ACTIVE |",
            "| segmentation_integrity | ACTIVE |",
            "| focus | ACTIVE |",
            "| exposure | ACTIVE |",
            "| illumination_uniformity | ACTIVE |",
            "| resolution | ACTIVE |",
            "| color_cast | ACTIVE |",
            "| occlusion | ACTIVE |",
            "| stain_suspected | **DEFERRED** |",
            "",
            "## Provenance story",
            "",
            "1. D4-C v1: in-domain TARGET_PASS; cross-domain FAIL",
            "2. D4-D.1: D4C_CROSS_DOMAIN_CONCERN (stain-triggered retake surge)",
            "3. D4-C.1-A: COLOR_ACQUISITION_STYLE shortcut",
            "4. D4-C.1-B: style aug + consistency NEEDS_IMPROVEMENT_STOP",
            "5. D4-C.1-C: representation invariance FAILED",
            "6. D4-C.1-D: SOURCE_CONFOUNDING_SEVERE; EXISTING_DATA_RESCUABLE=false",
            "7. D4-E: stain deferred; production Input Guard partial freeze",
            "",
            "## Production unified (n=130, stain disabled)",
            "",
            f"- pass/warning/retake = {audit.get('pass')}/{audit.get('warning')}/{audit.get('retake')}",
            f"- stain invocations = {audit.get('stain_model_invocations')}",
            f"- stain-triggered warning/retake = 0/0",
            f"- evaluation_complete true/false = "
            f"{audit.get('evaluation_complete', {}).get('true')}/"
            f"{audit.get('evaluation_complete', {}).get('false')}",
            f"- guard_ready = true",
            f"- full_capability_coverage = false",
            "",
            "## Known limitation",
            "",
            "- `STAIN_DETECTION_DEFERRED` / `SOURCE_DATASET_CONFOUNDING_SEVERE`",
            "- D5 coating-color must not claim external staining is excluded.",
            "",
            "## STOP",
            "",
            "Do not auto-enter D5 without confirmation.",
            "",
        ]
    )
    (docs_dir / "D4_FINAL_FREEZE_REPORT.md").write_text(report, encoding="utf-8")
    return stats
