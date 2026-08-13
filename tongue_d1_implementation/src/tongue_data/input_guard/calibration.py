"""D4-B threshold calibration：仅 train+val；禁止使用 test。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .ontology import CheckId
from .policy import load_input_guard_policy
from .signal_features import enrich_features_with_signals

CALIBRATION_SPLITS = ("train", "val")
FORBIDDEN_SPLITS = ("test",)

PERCENTILES = (1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99)

FEATURE_KEYS_FOR_AUDIT = [
    "foreground_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_area_ratio",
    "left_touch_ratio",
    "right_touch_ratio",
    "top_touch_ratio",
    "bottom_touch_ratio",
    "border_touch_ratio",
    "component_count",
    "largest_component_ratio",
    "mean_foreground_probability",
    "roi_blur_score",
    "blur_score",
    "roi_gradient_energy",
    "mean_luminance",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "shadow_clip_ratio",
    "highlight_clip_ratio",
    "relative_luminance_range",
    "left_right_difference",
    "tongue_pixel_count",
    "effective_short_side_px",
]


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "missing": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            **{f"p{p:02d}" if p < 10 else f"p{p}": None for p in PERCENTILES},
        }
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "count": int(array.size),
        "missing": 0,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
    for percentile in PERCENTILES:
        key = f"p{percentile:02d}" if percentile < 10 else f"p{percentile}"
        # 统一用 p01 风格
        if percentile == 1:
            key = "p01"
        elif percentile == 2:
            key = "p02"
        elif percentile == 5:
            key = "p05"
        summary[key] = float(np.percentile(array, percentile))
    return summary


def collect_calibration_rows(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    device: str = "auto",
    allow_splits: tuple[str, ...] = CALIBRATION_SPLITS,
) -> list[dict[str, Any]]:
    """对指定 splits 跑 D3-E + 信号特征；硬拒绝 test。"""
    for split_name in allow_splits:
        if split_name in FORBIDDEN_SPLITS:
            raise ValueError("test split is forbidden for calibration/feature audit")

    from tongue_data.segmentation.inference import TongueSegmentationInference, load_rgb_image

    segmentation_dir = Path(segmentation_dir)
    frame = pd.read_parquet(segmentation_dir / "segmentation_manifest.parquet")
    subset = frame[frame["split"].astype(str).isin(allow_splits)].copy()
    if subset.empty:
        raise ValueError(f"no samples for splits={allow_splits}")
    # 二次保险：不得包含 test
    if (subset["split"].astype(str) == "test").any():
        raise ValueError("test samples leaked into calibration frame")

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
        return_model_space=False,
        return_probability=True,
        return_masked_roi=False,
    )
    rows: list[dict[str, Any]] = []
    subset = subset.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    for _index, row in subset.iterrows():
        sample_id = str(row["sample_id"])
        dataset_name = str(row["dataset"])
        split_name = str(row["split"])
        if split_name == "test":
            raise ValueError("refusing to process test during calibration")
        rgb, _mode = load_rgb_image(str(row["image_path"]))
        rgb_fingerprint = int(rgb.sum())
        seg = engine.predict(rgb, sample_id=sample_id)
        features = enrich_features_with_signals(rgb, seg)
        if int(rgb.sum()) != rgb_fingerprint:
            raise RuntimeError(f"RGB mutated for {sample_id}")
        payload = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "split": split_name,
            "segmentation_status": seg.status,
            **{key: getattr(features, key) for key in FEATURE_KEYS_FOR_AUDIT},
            "touches_left": features.touches_left,
            "touches_right": features.touches_right,
            "touches_top": features.touches_top,
            "touches_bottom": features.touches_bottom,
        }
        rows.append(payload)
    return rows


def build_feature_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = {
        "overall": rows,
        "biohit": [row for row in rows if row["dataset"] == "biohit"],
        "tongueset3": [row for row in rows if row["dataset"] == "tongueset3"],
    }
    report: dict[str, Any] = {
        "calibration_scope": {"splits": list(CALIBRATION_SPLITS), "excluded": ["test"]},
        "sample_count": len(rows),
        "per_dataset_counts": {
            "biohit": len(scopes["biohit"]),
            "tongueset3": len(scopes["tongueset3"]),
        },
        "features": {},
    }
    for feature_name in FEATURE_KEYS_FOR_AUDIT:
        report["features"][feature_name] = {}
        for scope_name, scope_rows in scopes.items():
            values = [
                float(row[feature_name])
                for row in scope_rows
                if row.get(feature_name) is not None
            ]
            missing = sum(1 for row in scope_rows if row.get(feature_name) is None)
            summary = _summarize_values(values)
            summary["missing"] = int(missing)
            report["features"][feature_name][scope_name] = summary
    return report


def _p(dist: dict, feature: str, key: str, scope: str = "overall") -> float:
    return float(dist["features"][feature][scope][key])


def propose_thresholds(distribution: dict[str, Any]) -> dict[str, Any]:
    """
    基于 train+val 分位数提出 engineering heuristic thresholds。
    保守策略：RETAKE 抓极端下/上尾；WARNING 抓较轻尾部。
    """
    # scale：lower-is-worse
    scale = {
        "warning_foreground_ratio": _p(distribution, "foreground_ratio", "p05"),
        "retake_foreground_ratio": _p(distribution, "foreground_ratio", "p01"),
        "warning_bbox_width_ratio": _p(distribution, "bbox_width_ratio", "p05"),
        "retake_bbox_width_ratio": _p(distribution, "bbox_width_ratio", "p01"),
        "warning_bbox_height_ratio": _p(distribution, "bbox_height_ratio", "p05"),
        "retake_bbox_height_ratio": _p(distribution, "bbox_height_ratio", "p01"),
        "rationale": (
            "lower-tail quantiles on train+val; multi-feature coincidence for RETAKE"
        ),
    }
    # completeness：higher touch worse；top 仅 warning
    completeness = {
        "warning_side_touch_ratio": max(
            _p(distribution, "left_touch_ratio", "p95"),
            _p(distribution, "right_touch_ratio", "p95"),
            0.02,
        ),
        "retake_side_touch_ratio": max(
            _p(distribution, "left_touch_ratio", "p99"),
            _p(distribution, "right_touch_ratio", "p99"),
            0.08,
        ),
        "warning_bottom_touch_ratio": max(
            _p(distribution, "bottom_touch_ratio", "p95"), 0.02
        ),
        "retake_bottom_touch_ratio": max(
            _p(distribution, "bottom_touch_ratio", "p99"), 0.10
        ),
        "warning_top_touch_ratio": max(
            _p(distribution, "top_touch_ratio", "p95"), 0.05
        ),
        "rationale": (
            "top-only contact never auto-RETAKE; left/right/bottom high touch = crop risk"
        ),
    }
    # 本数据集 keep-largest 后 largest_component_ratio 常退化到 1.0；
    # 分位数不可用时回退到 conservative engineering floor。
    largest_p05 = _p(distribution, "largest_component_ratio", "p05")
    largest_p01 = _p(distribution, "largest_component_ratio", "p01")
    if largest_p05 >= 0.999:
        largest_p05 = 0.95
    if largest_p01 >= 0.999:
        largest_p01 = 0.85
    mean_prob_p05 = _p(distribution, "mean_foreground_probability", "p05")
    # 避免把“高置信代理”上尾误当成质量门禁中心
    mean_prob_warning = min(mean_prob_p05, 0.90)
    segmentation = {
        "warning_largest_component_ratio": float(largest_p05),
        "retake_largest_component_ratio": float(largest_p01),
        "warning_mean_probability": float(mean_prob_warning),
        "retake_component_count": max(
            _p(distribution, "component_count", "p99"), 5.0
        ),
        "rationale": (
            "fragmentation via largest_component_ratio; if train+val collapses to 1.0, "
            "use engineering floors 0.95/0.85; probability is proxy only"
        ),
    }
    # focus：lower laplacian worse；取更保守的 overall 与 domain 最小值，避免一域过严
    blur_p01_overall = _p(distribution, "roi_blur_score", "p01")
    blur_p05_overall = _p(distribution, "roi_blur_score", "p05")
    blur_p01_bio = _p(distribution, "roi_blur_score", "p01", "biohit")
    blur_p01_ts3 = _p(distribution, "roi_blur_score", "p01", "tongueset3")
    focus = {
        "retake_roi_laplacian": float(min(blur_p01_overall, blur_p01_bio, blur_p01_ts3)),
        "warning_roi_laplacian": float(blur_p05_overall),
        "rationale": (
            "conservative global RETAKE at min domain p01; WARNING at overall p05; "
            "reduces false RETAKE under domain shift"
        ),
    }
    exposure = {
        "warning_dark_pixel_ratio": _p(distribution, "dark_pixel_ratio", "p95"),
        "retake_dark_pixel_ratio": _p(distribution, "dark_pixel_ratio", "p99"),
        "warning_bright_pixel_ratio": _p(distribution, "bright_pixel_ratio", "p95"),
        "retake_bright_pixel_ratio": _p(distribution, "bright_pixel_ratio", "p99"),
        "warning_shadow_clip_ratio": _p(distribution, "shadow_clip_ratio", "p95"),
        "retake_shadow_clip_ratio": _p(distribution, "shadow_clip_ratio", "p99"),
        "warning_highlight_clip_ratio": _p(
            distribution, "highlight_clip_ratio", "p95"
        ),
        "retake_highlight_clip_ratio": _p(
            distribution, "highlight_clip_ratio", "p99"
        ),
        "rationale": (
            "clipping/percentile ratios; not mean RGB to avoid tongue-color confusion"
        ),
    }
    illumination = {
        "warning_relative_range": _p(
            distribution, "relative_luminance_range", "p95"
        ),
        "retake_relative_range": _p(
            distribution, "relative_luminance_range", "p99"
        ),
        "rationale": "mask-aware grid relative range; only extreme nonuniformity RETAKE",
    }
    resolution = {
        "warning_tongue_pixel_count": _p(distribution, "tongue_pixel_count", "p05"),
        "retake_tongue_pixel_count": _p(distribution, "tongue_pixel_count", "p01"),
        "warning_effective_short_side_px": _p(
            distribution, "effective_short_side_px", "p05"
        ),
        "retake_effective_short_side_px": _p(
            distribution, "effective_short_side_px", "p01"
        ),
        "rationale": "effective tongue pixels / short side; engineering V1 range only",
    }
    return {
        "status": "engineering_heuristic",
        "source": "train_val_distribution",
        "quality.tongue_presence": {
            "features": ["segmentation_status", "foreground_ratio"],
            "warning": None,
            "retake": "NO_TONGUE_DETECTED",
            "rationale": "direct mapping from D3-E no_tongue_detected",
        },
        "quality.tongue_scale": scale,
        "quality.tongue_completeness": completeness,
        "quality.segmentation_integrity": segmentation,
        "quality.focus": focus,
        "quality.exposure": exposure,
        "quality.illumination_uniformity": illumination,
        "quality.resolution": resolution,
    }


def apply_thresholds_to_policy_doc(
    policy_doc: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    doc = json.loads(json.dumps(policy_doc))  # deep copy via JSON
    doc["version"] = "1.1"
    doc["policy_version"] = "1.1"
    doc["contract_version"] = "1.0"
    doc["threshold_status"] = "engineering_heuristic"
    notes = list(doc.get("notes") or [])
    notes.append(
        "D4-B thresholds calibrated on BioHit+TongueSet3 train+val only; "
        "test excluded; engineering heuristic, not clinical."
    )
    doc["notes"] = notes

    mapping = {
        "tongue_scale": "quality.tongue_scale",
        "tongue_completeness": "quality.tongue_completeness",
        "segmentation_integrity": "quality.segmentation_integrity",
        "focus": "quality.focus",
        "exposure": "quality.exposure",
        "illumination_uniformity": "quality.illumination_uniformity",
        "resolution": "quality.resolution",
        "tongue_presence": "quality.tongue_presence",
    }
    for short_name, full_name in mapping.items():
        cfg = doc["checks"][short_name]
        cfg["needs_calibration"] = False
        cfg["implementation_stage"] = "D4-B"
        proposed = thresholds[full_name]
        # presence 无数值 threshold（空 mapping，避免 null+needs_calibration=false）
        if short_name == "tongue_presence":
            cfg["thresholds"] = {}
            cfg["needs_calibration"] = False
            continue
        numeric = {
            key: value
            for key, value in proposed.items()
            if key not in {"rationale", "features", "warning", "retake", "status", "source"}
            and isinstance(value, (int, float))
        }
        cfg["thresholds"] = numeric

    # 未实现保持 needs_calibration
    for short_name in ("color_cast", "occlusion", "stain_suspected"):
        doc["checks"][short_name]["needs_calibration"] = True
        if short_name == "stain_suspected":
            doc["checks"][short_name]["implementation_stage"] = "D4-C"
            doc["checks"][short_name]["enabled"] = False
        else:
            doc["checks"][short_name]["implementation_stage"] = "D4-D"
            doc["checks"][short_name]["enabled"] = True
        doc["checks"][short_name]["thresholds"] = {"warning": None, "retake": None}
    return doc


def run_calibration_pipeline(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    write_policy: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/d4")
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_calibration_rows(
        checkpoint_path=checkpoint_path,
        segmentation_dir=segmentation_dir,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        device=device,
        allow_splits=CALIBRATION_SPLITS,
    )
    distribution = build_feature_distribution(rows)
    thresholds = propose_thresholds(distribution)

    # border-touch top samples for manual review
    border_ranked = sorted(
        [row for row in rows if row.get("border_touch_ratio") is not None],
        key=lambda item: float(item["border_touch_ratio"]),
        reverse=True,
    )[:20]
    worst_lists = {
        "highest_border_touch": [
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "border_touch_ratio": row["border_touch_ratio"],
                "touches": {
                    "left": row["touches_left"],
                    "right": row["touches_right"],
                    "top": row["touches_top"],
                    "bottom": row["touches_bottom"],
                },
            }
            for row in border_ranked
        ],
        "lowest_roi_blur": sorted(
            [row for row in rows if row.get("roi_blur_score") is not None],
            key=lambda item: float(item["roi_blur_score"]),
        )[:20],
        "lowest_foreground_ratio": sorted(
            [row for row in rows if row.get("foreground_ratio") is not None],
            key=lambda item: float(item["foreground_ratio"]),
        )[:20],
    }

    dist_path = reports_dir / "d4b_feature_distribution.json"
    dist_path.write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calib_path = reports_dir / "d4b_threshold_calibration.json"
    calib_payload = {
        "stage": "D4-B",
        "calibration_datasets": ["biohit", "tongueset3"],
        "calibration_splits": list(CALIBRATION_SPLITS),
        "calibration_sample_count": len(rows),
        "test_used_for_calibration": False,
        "threshold_status": "engineering_heuristic",
        "checks": thresholds,
        "manual_review_lists": {
            "highest_border_touch": worst_lists["highest_border_touch"],
            "lowest_roi_blur_sample_ids": [
                row["sample_id"] for row in worst_lists["lowest_roi_blur"]
            ],
            "lowest_foreground_sample_ids": [
                row["sample_id"] for row in worst_lists["lowest_foreground_ratio"]
            ],
        },
    }
    calib_path.write_text(
        json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    policy_path = Path(policy_path)
    policy_doc = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    updated = apply_thresholds_to_policy_doc(policy_doc, thresholds)
    if write_policy:
        policy_path.write_text(
            yaml.safe_dump(updated, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        # 校验可加载
        load_input_guard_policy(policy_path)

    (output_dir / "d4b_feature_distribution.json").write_text(
        dist_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "d4b_threshold_calibration.json").write_text(
        calib_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {
        "sample_count": len(rows),
        "distribution_path": str(dist_path),
        "calibration_path": str(calib_path),
        "policy_path": str(policy_path),
        "policy_version": updated.get("policy_version"),
        "thresholds": thresholds,
    }
