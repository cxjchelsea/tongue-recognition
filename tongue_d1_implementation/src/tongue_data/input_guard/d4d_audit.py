"""D4-D unified frozen test audit + smoke。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .runtime import InputGuardRuntime, format_runtime_summary


def run_unified_test_audit(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    stain_checkpoint: str | Path | None = None,
    stain_thresholds: str | Path | None = None,
    device: str = "auto",
    allow_test: bool = False,
) -> dict[str, Any]:
    if not allow_test:
        raise ValueError("unified test audit requires allow_test=True after freeze")

    from tongue_data.segmentation.inference import load_rgb_image

    manifest = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    test = manifest[manifest["split"].astype(str) == "test"].copy()
    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
    )

    decision_counts = {"pass": 0, "warning": 0, "retake": 0}
    eval_complete = {"true": 0, "false": 0}
    per_check_state: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, int] = {}
    multi_reason = 0
    per_dataset: dict[str, dict[str, int]] = {}
    rows_out = []

    for _index, row in test.iterrows():
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        rgb, _mode = load_rgb_image(str(row["image_path"]))
        result = runtime.evaluate(rgb, sample_id=sample_id)
        decision_counts[result.decision] = decision_counts.get(result.decision, 0) + 1
        key = "true" if result.evaluation_complete else "false"
        eval_complete[key] += 1
        if len(result.reason_codes) > 1:
            multi_reason += 1
        for code in result.reason_codes:
            reason_counts[code] = reason_counts.get(code, 0) + 1
        per_dataset.setdefault(
            dataset, {"pass": 0, "warning": 0, "retake": 0, "n": 0}
        )
        per_dataset[dataset]["n"] += 1
        per_dataset[dataset][result.decision] += 1
        for check_id, check in result.checks.items():
            short = check_id.split(".", 1)[-1]
            bucket = per_check_state.setdefault(
                short, {"evaluated": 0, "unavailable": 0, "not_evaluated": 0}
            )
            state = check.evaluation_state
            if state == "evaluated":
                bucket["evaluated"] += 1
            elif state == "unavailable":
                bucket["unavailable"] += 1
            else:
                bucket["not_evaluated"] += 1
        rows_out.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "decision": result.decision,
                "evaluation_complete": result.evaluation_complete,
                "guard_ready": result.guard_ready,
                "primary_reason": result.primary_reason,
                "reason_codes": list(result.reason_codes),
            }
        )

    report = {
        "stage": "D4-D",
        "total": int(len(test)),
        "decision_counts": decision_counts,
        "evaluation_complete": eval_complete,
        "per_check_state": per_check_state,
        "reason_counts": reason_counts,
        "multi_reason_samples": multi_reason,
        "per_dataset": per_dataset,
        "guard_ready_system": bool(rows_out[0]["guard_ready"]) if rows_out else False,
        "threshold_retuned_on_test": False,
        "samples": rows_out,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = Path("reports/d4")
    reports.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    (output_dir / "d4d_unified_test_audit.json").write_text(payload, encoding="utf-8")
    (reports / "d4d_unified_test_audit.json").write_text(payload, encoding="utf-8")
    return report


def run_unified_smoke(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    stain_checkpoint: str | Path | None = None,
    stain_thresholds: str | Path | None = None,
    device: str = "auto",
    per_dataset: int = 10,
) -> dict[str, Any]:
    from tongue_data.segmentation.inference import load_rgb_image

    manifest = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
    )
    selected = []
    for dataset_name in ("biohit", "tongueset3"):
        subset = manifest[manifest["dataset"].astype(str) == dataset_name].head(
            int(per_dataset)
        )
        selected.append(subset)
    # stained optional：若 segmentation manifest 无 stained，跳过
    frame = pd.concat(selected, ignore_index=True)
    results = []
    for _index, row in frame.iterrows():
        rgb, _mode = load_rgb_image(str(row["image_path"]))
        result = runtime.evaluate(rgb, sample_id=str(row["sample_id"]))
        results.append(
            {
                "sample_id": str(row["sample_id"]),
                "dataset": str(row["dataset"]),
                "summary": format_runtime_summary(result),
                "decision": result.decision,
                "evaluation_complete": result.evaluation_complete,
                "guard_ready": result.guard_ready,
            }
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(results),
        "results": results,
    }
    (output_dir / "d4d_unified_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def evaluate_known_biohit_278(
    *,
    checkpoint_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    image_path: str | Path,
    stain_checkpoint: str | Path | None = None,
    stain_thresholds: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
    )
    result = runtime.evaluate(image_path, sample_id="biohit::278.bmp")
    return {
        "sample_id": "biohit::278.bmp",
        "decision": result.decision,
        "usable": result.usable,
        "evaluation_complete": result.evaluation_complete,
        "guard_ready": result.guard_ready,
        "primary_reason": result.primary_reason,
        "reason_codes": list(result.reason_codes),
        "checks": {
            key: {
                "state": check.evaluation_state,
                "finding": check.finding,
                "effect": check.decision_effect,
                "reason": check.reason_code,
            }
            for key, check in result.checks.items()
        },
        "note": "known D3 failure; do not retune thresholds for this case",
    }
