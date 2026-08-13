"""D4-A contract smoke：真实 D3-E result → feature adapter → skeleton result。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decision import build_contract_skeleton_result
from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    defined_checks_count,
    implemented_checks_count,
    registered_reason_codes,
)
from .policy import load_input_guard_policy
from .validators import validate_input_guard_contract


def run_input_guard_contract_smoke(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    biohit_count: int = 5,
    tongueset3_count: int = 5,
) -> dict[str, Any]:
    """
    不训练、不调阈值。
    contract_status=PASS 仅表示 schema/adapter 正常，
    不表示照片质量已完整清关。
    """
    from tongue_data.segmentation.inference import TongueSegmentationInference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/d4")
    reports_dir.mkdir(parents=True, exist_ok=True)

    contract_errors, contract_warnings = validate_input_guard_contract(policy_path)
    policy = load_input_guard_policy(policy_path)

    segmentation_dir = Path(segmentation_dir)
    frame = pd.read_parquet(segmentation_dir / "segmentation_manifest.parquet")
    test_frame = frame[frame["split"].astype(str) == "test"].copy()

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
        return_model_space=False,
        return_probability=True,
        return_masked_roi=False,
    )

    selected_rows: list[pd.Series] = []
    for dataset_name, count in (
        ("biohit", biohit_count),
        ("tongueset3", tongueset3_count),
    ):
        subset = test_frame[test_frame["dataset"].astype(str) == dataset_name]
        subset = subset.sort_values("sample_id").reset_index(drop=True)
        if subset.empty:
            continue
        indices = list(
            dict.fromkeys(
                int(value)
                for value in np.linspace(
                    0, len(subset) - 1, num=min(count, len(subset)), dtype=int
                )
            )
        )
        for index in indices:
            selected_rows.append(subset.iloc[index])

    samples: list[dict[str, Any]] = []
    schema_ok = True
    for row in selected_rows:
        sample_id = str(row["sample_id"])
        seg_result = engine.predict(str(row["image_path"]), sample_id=sample_id)
        guard_result = build_contract_skeleton_result(seg_result, policy)
        try:
            payload = guard_result.to_dict()
        except Exception as exc:
            schema_ok = False
            payload = {"error": str(exc)}
        sample_dir = output_dir / sample_id.replace("::", "__")
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "input_guard.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": str(row["dataset"]),
                "segmentation_status": seg_result.status,
                "guard_decision": payload.get("decision"),
                "evaluation_complete": payload.get("evaluation_complete"),
                "guard_ready": payload.get("guard_ready"),
                "usable": payload.get("usable"),
                "not_evaluated_checks": sum(
                    1
                    for check in (payload.get("checks") or {}).values()
                    if check.get("evaluation_state") == "not_evaluated"
                ),
                "feature_null_blur_score": (
                    (payload.get("features") or {}).get("blur_score") is None
                ),
                "schema_ok": "error" not in payload,
            }
        )

    contract_status = "PASS" if (not contract_errors and schema_ok) else "FAIL"
    summary = {
        "stage": "D4-A",
        "contract_status": contract_status,
        "note": (
            "contract_status=PASS means schema/policy/adapter smoke OK; "
            "NOT a claim that these images fully passed quality checks."
        ),
        "input_guard_contract_version": INPUT_GUARD_CONTRACT_VERSION,
        "defined_checks_count": defined_checks_count(),
        "implemented_checks_count": implemented_checks_count(),
        "defined_reason_codes_count": len(registered_reason_codes()),
        "sample_count": len(samples),
        "all_evaluation_complete_false": all(
            item.get("evaluation_complete") is False for item in samples
        ),
        "all_guard_ready_false": all(
            item.get("guard_ready") is False for item in samples
        ),
        "all_schema_ok": all(item.get("schema_ok") for item in samples),
        "all_missing_blur_null": all(
            item.get("feature_null_blur_score") for item in samples
        ),
        "policy_validation_errors": contract_errors,
        "policy_validation_warnings": contract_warnings,
        "samples": samples,
    }
    (output_dir / "d4a_contract_smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "d4a_contract_smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
