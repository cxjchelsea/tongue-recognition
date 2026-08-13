"""D4-B frozen test engineering audit（不改 threshold）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .policy import load_input_guard_policy
from .runtime import InputGuardRuntime

KNOWN_FAILURE_SAMPLE_ID = "biohit::278.bmp"


def run_test_engineering_audit(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """
    仅在 policy Freeze 后运行一次。
    禁止根据结果改 threshold。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/d4")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 记录 policy 指纹，确保 audit 后未改
    policy_text_before = Path(policy_path).read_text(encoding="utf-8")
    policy = load_input_guard_policy(policy_path)

    frame = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    test_frame = frame[frame["split"].astype(str) == "test"].copy()
    test_frame = test_frame.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    if len(test_frame) != 130:
        raise ValueError(f"expected 130 test samples, got {len(test_frame)}")

    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
    )

    samples: list[dict[str, Any]] = []
    decision_counts = {"pass": 0, "warning": 0, "retake": 0}
    per_dataset = {
        "biohit": {"pass": 0, "warning": 0, "retake": 0},
        "tongueset3": {"pass": 0, "warning": 0, "retake": 0},
    }
    reason_counts: dict[str, int] = {}
    known_failure: dict[str, Any] | None = None

    for _index, row in test_frame.iterrows():
        sample_id = str(row["sample_id"])
        dataset_name = str(row["dataset"])
        result = runtime.evaluate(str(row["image_path"]), sample_id=sample_id)
        decision_counts[result.decision] = decision_counts.get(result.decision, 0) + 1
        if dataset_name in per_dataset:
            per_dataset[dataset_name][result.decision] += 1
        for reason in result.reason_codes:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        sample_record = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "decision": result.decision,
            "usable": result.usable,
            "evaluation_complete": result.evaluation_complete,
            "guard_ready": result.guard_ready,
            "primary_reason": result.primary_reason,
            "reason_codes": result.reason_codes,
            "foreground_ratio": (result.features or {}).get("foreground_ratio"),
            "roi_blur_score": (result.features or {}).get("roi_blur_score"),
        }
        samples.append(sample_record)
        if sample_id == KNOWN_FAILURE_SAMPLE_ID:
            known_failure = {
                "sample_id": sample_id,
                "decision": result.decision,
                "usable": result.usable,
                "primary_reason": result.primary_reason,
                "reason_codes": result.reason_codes,
                "checks": {
                    key: {
                        "evaluation_state": check.evaluation_state,
                        "finding": check.finding,
                        "decision_effect": check.decision_effect,
                        "reason_code": check.reason_code,
                        "score": check.score,
                    }
                    for key, check in result.checks.items()
                },
                "features_subset": {
                    key: (result.features or {}).get(key)
                    for key in (
                        "foreground_ratio",
                        "component_count",
                        "largest_component_ratio",
                        "mean_foreground_probability",
                        "roi_blur_score",
                        "mean_luminance",
                        "tongue_pixel_count",
                        "bbox_width_ratio",
                        "bbox_height_ratio",
                    )
                },
                "note": "do not retune thresholds for this sample",
            }

    retake_rate = decision_counts["retake"] / max(len(samples), 1)
    calibration_review_required = retake_rate >= 0.25
    report = {
        "stage": "D4-B",
        "purpose": "engineering_audit_only",
        "note": (
            "Do NOT tune thresholds from this audit. "
            "BioHit/TongueSet3 are not QC gold labels."
        ),
        "total": len(samples),
        "decision_counts": decision_counts,
        "per_dataset_decisions": per_dataset,
        "reason_counts": reason_counts,
        "retake_rate": retake_rate,
        "calibration_review_required": calibration_review_required,
        "policy_version": policy.doc.get("policy_version", policy.version),
        "evaluation_complete_always_false": all(
            item["evaluation_complete"] is False for item in samples
        ),
        "guard_ready_always_false": all(
            item["guard_ready"] is False for item in samples
        ),
        "retake_samples": [
            item for item in samples if item["decision"] == "retake"
        ],
        "warning_samples": [
            item for item in samples if item["decision"] == "warning"
        ],
        "known_failure": known_failure,
    }
    report_path = reports_dir / "d4b_test_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if known_failure is not None:
        (reports_dir / "d4b_known_failure_analysis.json").write_text(
            json.dumps(known_failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "d4b_test_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    policy_text_after = Path(policy_path).read_text(encoding="utf-8")
    if policy_text_after != policy_text_before:
        raise RuntimeError("policy file was modified during test audit; abort")
    report["policy_unchanged_after_audit"] = True
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
