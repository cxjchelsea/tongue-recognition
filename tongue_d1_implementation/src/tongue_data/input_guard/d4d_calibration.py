"""D4-D color_cast / occlusion calibration：仅 BioHit+TongueSet3 train+val。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .calibration import CALIBRATION_SPLITS, FORBIDDEN_SPLITS
from .color_cast import compute_color_cast_features, evaluate_color_cast
from .d4d_synthetic import (
    run_color_cast_synthetic_audit,
    run_occlusion_synthetic_audit,
)
from .occlusion import compute_occlusion_features, evaluate_occlusion
from .policy import InputGuardPolicy, load_input_guard_policy


def load_d4d_config(path: str | Path = "configs/input_guard_d4d_v1.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _assert_no_test(frame: pd.DataFrame) -> None:
    splits = set(frame["split"].astype(str).unique())
    if splits & set(FORBIDDEN_SPLITS):
        raise ValueError(f"D4-D calibration must not use test splits: {splits}")


def collect_d4d_calibration_rows(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    d4d_config: dict[str, Any],
    device: str = "auto",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    from tongue_data.segmentation.inference import (
        TongueSegmentationInference,
        load_rgb_image,
    )

    manifest = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    allowed = set(d4d_config.get("datasets", ["biohit", "tongueset3"]))
    frame = manifest[
        manifest["split"].astype(str).isin(CALIBRATION_SPLITS)
        & manifest["dataset"].astype(str).isin(allowed)
    ].copy()
    _assert_no_test(frame)
    if max_samples is not None:
        frame = frame.head(int(max_samples))

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
        return_probability=True,
        return_masked_roi=False,
    )
    rows: list[dict[str, Any]] = []
    neutral_cfg = d4d_config.get("color_cast", {}).get("neutral", {})
    occlusion_cfg = d4d_config.get("occlusion", {})
    for _index, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        image_path = str(row["image_path"])
        rgb, _mode = load_rgb_image(image_path)
        result = engine.predict(rgb, sample_id=sample_id)
        mask = getattr(result, "original_binary_mask", None)
        prob = getattr(result, "original_probability_mask", None)
        cast_feat = compute_color_cast_features(rgb, mask, neutral_cfg=neutral_cfg)
        occ_feat = compute_occlusion_features(
            rgb, mask, prob, occlusion_cfg=occlusion_cfg
        )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": str(row["dataset"]),
                "split": str(row["split"]),
                "rgb": rgb,
                "mask": mask,
                "probability": prob,
                "color_cast_features": cast_feat,
                "occlusion_features": occ_feat,
            }
        )
    return rows


def _percentile_threshold(values: list[float], percentile: float) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=np.float64), percentile))


def _synthetic_severe_cast_magnitudes(
    rows: list[dict[str, Any]],
    d4d_config: dict[str, Any],
) -> list[float]:
    """仅用 train/val 子集 + deterministic synthetic；不用 test。"""
    from .color_cast import apply_channel_cast, compute_color_cast_features
    from .d4d_synthetic import select_fixed_sample_indices

    synth_cfg = d4d_config.get("synthetic", {}).get("color_cast", {})
    seed = int(d4d_config.get("seed", 20260813))
    directions = list(synth_cfg.get("directions", ["red", "green", "blue", "yellow"]))
    severe_gain = float(synth_cfg.get("severe_gain", 1.55))
    count = int(synth_cfg.get("sample_count", 24))
    indices = select_fixed_sample_indices(len(rows), count, seed)
    neutral_cfg = d4d_config.get("color_cast", {}).get("neutral", {})
    magnitudes: list[float] = []
    for index in indices:
        row = rows[index]
        for direction in directions:
            casted = apply_channel_cast(
                row["rgb"], direction=direction, gain=severe_gain
            )
            feat = compute_color_cast_features(
                casted, row.get("mask"), neutral_cfg=neutral_cfg
            )
            if feat.get("estimated_cast_magnitude") is not None:
                magnitudes.append(float(feat["estimated_cast_magnitude"]))
    return magnitudes


def propose_color_cast_thresholds(
    clean_mags: list[float],
    synthetic_severe_mags: list[float],
    *,
    clean_retake_max: float = 0.02,
) -> tuple[float | None, float | None]:
    """
    在 clean false-retake <= target 约束下，
    尽量降低 retake 阈值以提高 synthetic severe 检出。
    """
    if not clean_mags:
        return None, None
    clean_arr = np.sort(np.asarray(clean_mags, dtype=np.float64))
    # 至多 clean_retake_max 比例样本触发 RETAKE
    allowed = int(np.floor(clean_retake_max * len(clean_arr)))
    if allowed <= 0:
        clean_cap = float(clean_arr[-1]) + 1e-3
    else:
        clean_cap = float(clean_arr[-allowed])

    if synthetic_severe_mags:
        # 目标：至少 90% synthetic severe >= retake_thr
        synth = np.sort(np.asarray(synthetic_severe_mags, dtype=np.float64))
        target_index = int(np.floor(0.10 * len(synth)))  # 10th percentile
        target_index = min(max(target_index, 0), len(synth) - 1)
        desired = float(synth[target_index])
        # 优先满足 clean false-retake 约束，再尽量贴近 synthetic 灵敏度目标
        cast_retake = max(clean_cap, desired)
    else:
        cast_retake = clean_cap

    # warning 更低：clean p95 与 retake 之间
    cast_warning = float(np.percentile(clean_arr, 95))
    if cast_warning >= cast_retake:
        cast_warning = max(cast_retake * 0.75, float(np.percentile(clean_arr, 90)))
    if cast_warning >= cast_retake:
        cast_warning = cast_retake * 0.9
    return cast_warning, cast_retake


def propose_d4d_thresholds(
    rows: list[dict[str, Any]],
    d4d_config: dict[str, Any],
) -> dict[str, Any]:
    cast_mags = [
        row["color_cast_features"].get("estimated_cast_magnitude")
        for row in rows
        if row["color_cast_features"].get("support_ok")
    ]
    cast_mags = [float(value) for value in cast_mags if value is not None]
    occ_scores = [
        row["occlusion_features"].get("combined_score")
        for row in rows
        if row["occlusion_features"].get("available")
    ]
    occ_scores = [float(value) for value in occ_scores if value is not None]

    synth_mags = _synthetic_severe_cast_magnitudes(rows, d4d_config)
    cast_gate = d4d_config.get("color_cast", {}).get("gates", {})
    cast_warning, cast_retake = propose_color_cast_thresholds(
        cast_mags,
        synth_mags,
        clean_retake_max=float(cast_gate.get("clean_retake_max", 0.02)),
    )

    occ_warning = _percentile_threshold(occ_scores, 98)
    occ_retake = _percentile_threshold(occ_scores, 99.5)
    if occ_warning is not None and occ_retake is not None:
        if occ_retake <= occ_warning:
            occ_retake = occ_warning * 1.2 + 0.01

    return {
        "color_cast": {
            "warning_cast_magnitude": cast_warning,
            "retake_cast_magnitude": cast_retake,
            "calibration_support_count": len(cast_mags),
            "cast_magnitude_p50": _percentile_threshold(cast_mags, 50),
            "cast_magnitude_p95": _percentile_threshold(cast_mags, 95),
            "synthetic_severe_magnitude_p10": _percentile_threshold(synth_mags, 10),
            "synthetic_severe_count": len(synth_mags),
        },
        "occlusion": {
            "warning_combined_score": occ_warning,
            "retake_combined_score": occ_retake,
            "require_multi_evidence_for_retake": True,
            "calibration_available_count": len(occ_scores),
            "combined_score_p50": _percentile_threshold(occ_scores, 50),
            "combined_score_p95": _percentile_threshold(occ_scores, 95),
        },
        "used_splits": list(CALIBRATION_SPLITS),
        "forbidden_splits": list(FORBIDDEN_SPLITS),
    }


def _decision_counts(checks: list) -> dict[str, int]:
    counts = {"pass": 0, "warning": 0, "retake": 0, "unavailable": 0, "not_evaluated": 0}
    for check in checks:
        state = check.evaluation_state
        if state == "unavailable":
            counts["unavailable"] += 1
        elif state != "evaluated":
            counts["not_evaluated"] += 1
        elif check.decision_effect == "retake":
            counts["retake"] += 1
        elif check.decision_effect == "warning":
            counts["warning"] += 1
        else:
            counts["pass"] += 1
    return counts


def apply_d4d_thresholds_to_policy(
    policy_path: str | Path,
    *,
    thresholds: dict[str, Any],
    d4d_config: dict[str, Any],
    color_cast_status: str,
    occlusion_status: str,
    output_path: str | Path | None = None,
    policy_version: str = "1.3",
) -> Path:
    path = Path(policy_path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    # 保护 D4-C stain thresholds
    stain_before = dict(doc["checks"]["stain_suspected"].get("thresholds") or {})

    doc["version"] = policy_version
    doc["policy_version"] = policy_version

    cast_ok = color_cast_status == "PASS"
    occ_ok = occlusion_status == "PASS"

    cast_thr = thresholds["color_cast"]
    doc["checks"]["color_cast"].update(
        {
            "enabled": True,
            "implementation_stage": "D4-D",
            "implementation": "signal_rule",
            "needs_calibration": not cast_ok,
            "status": color_cast_status,
            "neutral_support": d4d_config.get("color_cast", {}).get("neutral", {}),
            "thresholds": {
                "warning_cast_magnitude": cast_thr.get("warning_cast_magnitude"),
                "retake_cast_magnitude": cast_thr.get("retake_cast_magnitude"),
            },
            "warning_reason": "COLOR_CAST_SUSPECTED",
            "retake_reason": "SEVERE_COLOR_CAST",
            "fallback": "insufficient_neutral_support -> unavailable",
        }
    )
    # 若未 PASS，保持 unavailable 友好：阈值可写但仍标记 needs_calibration
    if not cast_ok:
        doc["checks"]["color_cast"]["enabled"] = True
        # 仍提供实现，但 guard_ready 会 false；阈值若存在可用于实验
        if cast_thr.get("warning_cast_magnitude") is None:
            doc["checks"]["color_cast"]["thresholds"] = {
                "warning": None,
                "retake": None,
            }

    occ_thr = thresholds["occlusion"]
    doc["checks"]["occlusion"].update(
        {
            "enabled": True,
            "implementation_stage": "D4-D",
            "implementation": "signal_rule",
            "needs_calibration": not occ_ok,
            "status": occlusion_status,
            "thresholds": {
                "warning_combined_score": occ_thr.get("warning_combined_score"),
                "retake_combined_score": occ_thr.get("retake_combined_score"),
                "require_multi_evidence_for_retake": True,
            },
            "warning_reason": "TONGUE_OCCLUDED",
            "retake_reason": "TONGUE_OCCLUDED",
            "fallback": "missing_probability_map -> unavailable",
        }
    )

    # 强制保留 D4-C stain
    doc["checks"]["stain_suspected"]["thresholds"] = stain_before
    doc["checks"]["stain_suspected"]["needs_calibration"] = False
    doc["checks"]["stain_suspected"]["enabled"] = True

    notes = list(doc.get("notes") or [])
    note = (
        "D4-D color_cast/occlusion calibrated on BioHit+TongueSet3 train+val only; "
        "test excluded. Synthetic perturbations used for sensitivity only."
    )
    if note not in notes:
        notes.append(note)
    doc["notes"] = notes

    out = Path(output_path) if output_path else path
    out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    # 二次校验 stain 未变
    verify = yaml.safe_load(out.read_text(encoding="utf-8"))
    if verify["checks"]["stain_suspected"]["thresholds"] != stain_before:
        raise RuntimeError("D4-C stain thresholds were modified unexpectedly")
    return out


def run_d4d_calibration_pipeline(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    d4d_config_path: str | Path = "configs/input_guard_d4d_v1.yaml",
    device: str = "auto",
    write_policy: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    d4d_config = load_d4d_config(d4d_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = Path("reports/d4")
    reports.mkdir(parents=True, exist_ok=True)

    rows = collect_d4d_calibration_rows(
        checkpoint_path=checkpoint_path,
        segmentation_dir=segmentation_dir,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        d4d_config=d4d_config,
        device=device,
        max_samples=max_samples,
    )
    proposed = propose_d4d_thresholds(rows, d4d_config)

    # 临时 policy 用于评估 clean + synthetic
    tmp_policy_path = output_dir / "policy_d4d_candidate.yaml"
    base_policy = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
    base_policy["checks"]["color_cast"]["needs_calibration"] = False
    base_policy["checks"]["color_cast"]["thresholds"] = {
        "warning_cast_magnitude": proposed["color_cast"]["warning_cast_magnitude"],
        "retake_cast_magnitude": proposed["color_cast"]["retake_cast_magnitude"],
    }
    base_policy["checks"]["color_cast"]["neutral_support"] = d4d_config.get(
        "color_cast", {}
    ).get("neutral", {})
    base_policy["checks"]["occlusion"]["needs_calibration"] = False
    base_policy["checks"]["occlusion"]["thresholds"] = {
        "warning_combined_score": proposed["occlusion"]["warning_combined_score"],
        "retake_combined_score": proposed["occlusion"]["retake_combined_score"],
        "require_multi_evidence_for_retake": True,
    }
    # policy validate 允许 1.2；candidate 先保持 1.2
    tmp_policy_path.write_text(
        yaml.safe_dump(base_policy, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    policy = InputGuardPolicy(tmp_policy_path)

    cast_checks = [
        evaluate_color_cast(
            row["rgb"], row["mask"], policy, d4d_cfg=d4d_config
        )
        for row in rows
    ]
    occ_checks = [
        evaluate_occlusion(
            row["rgb"],
            row["mask"],
            row["probability"],
            policy,
            d4d_cfg=d4d_config,
        )
        for row in rows
    ]
    cast_counts = _decision_counts(cast_checks)
    occ_counts = _decision_counts(occ_checks)
    cast_retake_rate = cast_counts["retake"] / max(1, len(cast_checks))
    occ_retake_rate = occ_counts["retake"] / max(1, len(occ_checks))

    cast_synth = run_color_cast_synthetic_audit(
        rows, policy=policy, d4d_cfg=d4d_config
    )
    occ_synth = run_occlusion_synthetic_audit(
        rows, policy=policy, d4d_cfg=d4d_config
    )

    cast_gate = d4d_config.get("color_cast", {}).get("gates", {})
    occ_gate = d4d_config.get("occlusion", {}).get("gates", {})
    color_cast_status = (
        "PASS"
        if (
            cast_retake_rate <= float(cast_gate.get("clean_retake_max", 0.02))
            and (cast_synth.get("severe_detection_rate") or 0)
            >= float(cast_gate.get("synthetic_severe_detection_min", 0.90))
        )
        else "NEEDS_IMPROVEMENT"
    )
    occlusion_status = (
        "PASS"
        if (
            occ_retake_rate <= float(occ_gate.get("clean_retake_max", 0.02))
            and (occ_synth.get("severe_detection_rate") or 0)
            >= float(occ_gate.get("synthetic_severe_detection_min", 0.90))
            and (occ_synth.get("small_retake_rate") or 0)
            <= float(occ_gate.get("synthetic_small_retake_max", 0.10))
        )
        else "NEEDS_IMPROVEMENT"
    )

    support_ok_rate = float(
        np.mean(
            [1.0 if row["color_cast_features"].get("support_ok") else 0.0 for row in rows]
        )
    )

    cast_audit = {
        "calibration_sample_count": len(rows),
        "neutral_support_rate": support_ok_rate,
        "clean_counts": cast_counts,
        "clean_false_retake_rate": cast_retake_rate,
        "thresholds": proposed["color_cast"],
        "synthetic": {
            "severe_detection_rate": cast_synth.get("severe_detection_rate"),
            "moderate_detection_rate": cast_synth.get("moderate_detection_rate"),
            "directions": cast_synth.get("directions"),
        },
        "status": color_cast_status,
        "used_splits": list(CALIBRATION_SPLITS),
        "test_used": False,
    }
    occ_audit = {
        "calibration_sample_count": len(rows),
        "clean_counts": occ_counts,
        "clean_false_retake_rate": occ_retake_rate,
        "thresholds": proposed["occlusion"],
        "synthetic": {
            "summary": occ_synth.get("summary"),
            "severe_detection_rate": occ_synth.get("severe_detection_rate"),
            "severe_retake_rate": occ_synth.get("severe_retake_rate"),
            "small_retake_rate": occ_synth.get("small_retake_rate"),
        },
        "status": occlusion_status,
        "used_splits": list(CALIBRATION_SPLITS),
        "test_used": False,
    }

    (reports / "d4d_color_cast_audit.json").write_text(
        json.dumps(cast_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "d4d_occlusion_audit.json").write_text(
        json.dumps(occ_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "d4d_color_cast_audit.json").write_text(
        json.dumps(cast_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "d4d_occlusion_audit.json").write_text(
        json.dumps(occ_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "d4d_proposed_thresholds.json").write_text(
        json.dumps(proposed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    policy_out = None
    if write_policy:
        # 仅当两者都 PASS 才升 1.3 并正式启用；否则仍写候选状态但 guard_ready=false
        version = "1.3" if (
            color_cast_status == "PASS" and occlusion_status == "PASS"
        ) else "1.2"
        policy_out = str(
            apply_d4d_thresholds_to_policy(
                policy_path,
                thresholds=proposed,
                d4d_config=d4d_config,
                color_cast_status=color_cast_status,
                occlusion_status=occlusion_status,
                policy_version=version,
            )
        )

    return {
        "sample_count": len(rows),
        "color_cast_status": color_cast_status,
        "occlusion_status": occlusion_status,
        "proposed": proposed,
        "color_cast_audit": cast_audit,
        "occlusion_audit": occ_audit,
        "policy_path": policy_out,
        "test_used": False,
    }
