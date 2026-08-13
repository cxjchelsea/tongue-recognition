"""D3-E engineering regression：原图分辨率 Dice vs D3-C model-space。

用途仅限实现正确性验证，禁止据此调 threshold / 重训练 / 换模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import TongueSegmentationInference, load_rgb_image, save_inference_outputs
from .mask_ops import load_mask_raw, normalize_binary_mask
from .metrics import dice_coefficient


D3C_OVERALL_DICE = 0.974801113202165
D3C_BIOHIT_DICE = 0.9539436306959639
D3C_TONGUESET3_DICE = 0.9810583579540253
KNOWN_FAILURE_SAMPLE_ID = "biohit::278.bmp"


def _gt_coverage_by_roi(
    gt_mask: np.ndarray, bbox_roi: tuple[int, int, int, int] | None
) -> float | None:
    """GT 前景像素落在 predicted ROI 内的比例。"""
    if bbox_roi is None:
        return None
    gt_foreground = gt_mask > 0
    total = int(gt_foreground.sum())
    if total == 0:
        return None
    x1, y1, x2, y2 = bbox_roi
    roi_region = np.zeros_like(gt_foreground, dtype=bool)
    roi_region[y1:y2, x1:x2] = True
    covered = int(np.logical_and(gt_foreground, roi_region).sum())
    return float(covered / total)


def run_d3e_test_regression(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    known_failure_output: str | Path | None = None,
) -> dict[str, Any]:
    """在 frozen test split（130）上跑原图级 regression。"""
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/d3")
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = segmentation_dir / "segmentation_manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    test_frame = frame[frame["split"].astype(str) == "test"].copy()
    test_frame = test_frame.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    if len(test_frame) != 130:
        raise ValueError(f"expected 130 test samples, got {len(test_frame)}")

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
        return_model_space=False,
        return_probability=True,
        return_masked_roi=False,
    )

    per_sample: list[dict[str, Any]] = []
    invalid_masks = 0
    invalid_bbox = 0
    empty_predictions = 0
    empty_roi_success = 0
    border_touch_count = 0
    dice_all: list[float] = []
    dice_biohit: list[float] = []
    dice_tongueset3: list[float] = []
    coverage_scores: list[float] = []
    known_failure: dict[str, Any] | None = None

    for _index, row in test_frame.iterrows():
        sample_id = str(row["sample_id"])
        dataset_name = str(row["dataset"])
        image_path = str(row["image_path"])
        mask_path = str(row["mask_path"])

        gt_mask = normalize_binary_mask(load_mask_raw(mask_path)).astype(np.uint8)
        result = engine.predict(image_path, sample_id=sample_id)
        pred_mask = result.original_binary_mask
        assert pred_mask is not None

        # integrity
        if pred_mask.shape != gt_mask.shape:
            invalid_masks += 1
        unique_values = set(np.unique(pred_mask).tolist())
        if not unique_values.issubset({0, 1}):
            invalid_masks += 1

        if result.status == "no_tongue_detected":
            empty_predictions += 1
            dice_value = 0.0
            coverage = 0.0
        else:
            if result.bbox_tight is None or result.bbox_roi is None:
                invalid_bbox += 1
            if result.tongue_roi_rgb is None or result.tongue_roi_rgb.size == 0:
                empty_roi_success += 1
            if result.touches_image_border:
                border_touch_count += 1
            dice_value = dice_coefficient(pred_mask, gt_mask, threshold=0.5)
            coverage = _gt_coverage_by_roi(gt_mask, result.bbox_roi)
            if coverage is not None:
                coverage_scores.append(coverage)

        dice_all.append(float(dice_value))
        if dataset_name == "biohit":
            dice_biohit.append(float(dice_value))
        elif dataset_name == "tongueset3":
            dice_tongueset3.append(float(dice_value))

        sample_record = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "status": result.status,
            "original_size": [result.original_width, result.original_height],
            "dice": float(dice_value),
            "foreground_ratio_pred": result.mask_foreground_ratio,
            "foreground_ratio_gt": float(gt_mask.mean()),
            "component_count": result.component_count,
            "largest_component_ratio": result.largest_component_ratio,
            "bbox_tight": list(result.bbox_tight) if result.bbox_tight else None,
            "bbox_roi": list(result.bbox_roi) if result.bbox_roi else None,
            "gt_coverage_by_predicted_roi": coverage,
            "warnings": result.warnings,
        }
        per_sample.append(sample_record)

        if sample_id == KNOWN_FAILURE_SAMPLE_ID:
            failure_dir = output_dir / "known_failure" / sample_id.replace("::", "__")
            original_rgb, _mode = load_rgb_image(image_path)
            save_inference_outputs(
                result,
                failure_dir,
                original_rgb=original_rgb,
                save_overlay=True,
                save_probability=True,
            )
            # 粗分类：不武断断言标注错误
            failure_category = "undetermined"
            notes = []
            intersection = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
            if pred_mask.shape != gt_mask.shape:
                failure_category = "geometry_restoration_error"
                notes.append("restored mask shape != original/GT")
            elif dice_value < 0.05 and float(gt_mask.mean()) > 0.05:
                notes.append(
                    f"severe mismatch: intersection={intersection}, "
                    f"pred_fg={int(pred_mask.sum())}, gt_fg={int(gt_mask.sum())}; "
                    "shape restore OK; unlikely geometry bug"
                )
                if result.mask_foreground_ratio < 0.01:
                    failure_category = "model_undersegmentation"
                    notes.append("pred almost empty while GT has substantial foreground")
                elif intersection < 0.01 * max(int(gt_mask.sum()), 1):
                    # 预测非空但几乎不与 GT 重叠：暂不判定标注错误
                    failure_category = "undetermined"
                    notes.append(
                        "non-empty prediction barely overlaps GT; "
                        "could be model error, unusual capture, or annotation issue"
                    )
                else:
                    failure_category = "model_undersegmentation"

            known_failure = {
                "sample_id": sample_id,
                "original_size": [result.original_width, result.original_height],
                "gt_foreground_ratio": float(gt_mask.mean()),
                "pred_foreground_ratio": result.mask_foreground_ratio,
                "dice": float(dice_value),
                "component_count": result.component_count,
                "bbox_tight": list(result.bbox_tight) if result.bbox_tight else None,
                "bbox_roi": list(result.bbox_roi) if result.bbox_roi else None,
                "probability_summary": {
                    "mean_foreground_probability": result.mean_foreground_probability,
                    "max_probability": result.max_probability,
                },
                "failure_category": failure_category,
                "notes": notes,
                "overlay_dir": str(failure_dir),
                "purpose": "understanding only; do not tune frozen pipeline for this sample",
            }

    overall_dice = float(np.mean(dice_all)) if dice_all else 0.0
    biohit_dice = float(np.mean(dice_biohit)) if dice_biohit else 0.0
    tongueset3_dice = float(np.mean(dice_tongueset3)) if dice_tongueset3 else 0.0

    report = {
        "stage": "D3-E",
        "purpose": "engineering_regression_only",
        "note": (
            "Test access validates unletterbox/ROI implementation. "
            "Do NOT tune threshold, retrain, or change model based on this run."
        ),
        "total": len(test_frame),
        "overall_original_resolution_dice": overall_dice,
        "biohit_original_resolution_dice": biohit_dice,
        "tongueset3_original_resolution_dice": tongueset3_dice,
        "d3c_reference": {
            "overall_dice": D3C_OVERALL_DICE,
            "biohit_dice": D3C_BIOHIT_DICE,
            "tongueset3_dice": D3C_TONGUESET3_DICE,
        },
        "difference_vs_d3c": {
            "overall": overall_dice - D3C_OVERALL_DICE,
            "biohit": biohit_dice - D3C_BIOHIT_DICE,
            "tongueset3": tongueset3_dice - D3C_TONGUESET3_DICE,
        },
        "gates": {
            "overall_drop_max": 0.01,
            "domain_drop_max": 0.02,
            "overall_drop_ok": (D3C_OVERALL_DICE - overall_dice) <= 0.01,
            "biohit_drop_ok": (D3C_BIOHIT_DICE - biohit_dice) <= 0.02,
            "tongueset3_drop_ok": (D3C_TONGUESET3_DICE - tongueset3_dice) <= 0.02,
        },
        "invalid_masks": invalid_masks,
        "empty_predictions": empty_predictions,
        "invalid_bbox": invalid_bbox,
        "empty_roi": empty_roi_success,
        "border_touch_count": border_touch_count,
        "component_stats": {
            "mean_component_count": float(
                np.mean([item["component_count"] for item in per_sample])
            ),
            "mean_largest_component_ratio": float(
                np.mean([item["largest_component_ratio"] for item in per_sample])
            ),
        },
        "roi_coverage": {
            "mean": float(np.mean(coverage_scores)) if coverage_scores else None,
            "median": float(np.median(coverage_scores)) if coverage_scores else None,
            "p10": float(np.percentile(coverage_scores, 10)) if coverage_scores else None,
            "count": len(coverage_scores),
            "sanity_mean_median_gt_0_95": bool(
                coverage_scores
                and float(np.mean(coverage_scores)) > 0.95
                and float(np.median(coverage_scores)) > 0.95
            ),
        },
        "threshold": engine.threshold,
        "restoration_strategy": engine.model_metadata
        and "probability_remove_pad_bilinear_then_threshold_original",
        "config_hash": engine.model_metadata.get("config_hash"),
        "per_sample_path": str(output_dir / "per_sample.json"),
    }

    (output_dir / "per_sample.json").write_text(
        json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    regression_path = reports_dir / "d3e_inference_regression.json"
    regression_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(regression_path)

    if known_failure is not None:
        failure_path = Path(
            known_failure_output
            if known_failure_output is not None
            else reports_dir / "d3e_known_failure_analysis.json"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(known_failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["known_failure"] = known_failure
        report["known_failure_path"] = str(failure_path)

    (output_dir / "d3e_inference_regression.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_real_image_integration(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    biohit_count: int = 5,
    tongueset3_count: int = 5,
) -> dict[str, Any]:
    """从 test split 选取横/纵混合真实图做集成检查。"""
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(segmentation_dir / "segmentation_manifest.parquet")
    test_frame = frame[frame["split"].astype(str) == "test"].copy()

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
    )

    selected_rows: list[pd.Series] = []
    for dataset_name, count in (("biohit", biohit_count), ("tongueset3", tongueset3_count)):
        subset = test_frame[test_frame["dataset"].astype(str) == dataset_name].copy()
        subset = subset.sort_values("sample_id").reset_index(drop=True)
        # 尽量覆盖不同 foreground_ratio：低/中/高
        if len(subset) == 0:
            continue
        indices = np.linspace(0, len(subset) - 1, num=min(count, len(subset)), dtype=int)
        for index in indices:
            selected_rows.append(subset.iloc[int(index)])

    results: list[dict[str, Any]] = []
    for row in selected_rows:
        sample_id = str(row["sample_id"])
        image_path = str(row["image_path"])
        result = engine.predict(image_path, sample_id=sample_id)
        original_rgb, _mode = load_rgb_image(image_path)
        sample_dir = output_dir / sample_id.replace("::", "__")
        save_inference_outputs(result, sample_dir, original_rgb=original_rgb)
        orientation = (
            "landscape"
            if result.original_width > result.original_height
            else "portrait"
            if result.original_height > result.original_width
            else "square"
        )
        results.append(
            {
                "sample_id": sample_id,
                "dataset": str(row["dataset"]),
                "status": result.status,
                "orientation": orientation,
                "original_size": [result.original_width, result.original_height],
                "foreground_ratio": result.mask_foreground_ratio,
                "bbox_tight": list(result.bbox_tight) if result.bbox_tight else None,
                "roi_size": list(result.roi_size) if result.roi_size else None,
                "mask_shape_ok": bool(
                    result.original_binary_mask is not None
                    and result.original_binary_mask.shape
                    == (result.original_height, result.original_width)
                ),
                "roi_shape_ok": bool(
                    result.tongue_roi_rgb is not None
                    and result.tongue_roi_mask is not None
                    and result.tongue_roi_rgb.shape[:2]
                    == result.tongue_roi_mask.shape[:2]
                ),
                "output_dir": str(sample_dir),
            }
        )

    summary = {
        "count": len(results),
        "all_success": all(item["status"] == "success" for item in results),
        "all_mask_shape_ok": all(item["mask_shape_ok"] for item in results),
        "all_roi_shape_ok": all(item["roi_shape_ok"] for item in results),
        "samples": results,
    }
    (output_dir / "integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
