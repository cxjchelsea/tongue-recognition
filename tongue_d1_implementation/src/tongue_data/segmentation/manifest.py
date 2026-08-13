"""从 D2 clean + split 构建 segmentation_manifest。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SegmentationConfig
from .mask_ops import (
    foreground_ratio,
    load_image_rgb,
    load_mask_raw,
    normalize_binary_mask,
    unique_pixel_values,
)


def _percentile_stats(values: list[float]) -> dict:
    if not values:
        return {
            "min": None,
            "p01": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
            "count": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "count": int(len(array)),
    }


def select_segmentation_masks(
    spatial_clean: pd.DataFrame,
    samples_clean: pd.DataFrame,
    config: SegmentationConfig,
) -> tuple[pd.DataFrame, list[dict]]:
    """选择 BioHit/TongueSet3 的 segmentation.tongue mask，并解决多 mask。"""
    allowed_datasets = set(config.datasets)
    task = config.task
    annotation_type = config.annotation_type
    policy = str(config.data.get("multi_mask_policy", "prefer_canonical_origin"))

    candidates = spatial_clean[
        (spatial_clean["annotation_task"].astype(str) == task)
        & (spatial_clean["annotation_type"].astype(str) == annotation_type)
        & (spatial_clean["source_dataset"].astype(str).isin(allowed_datasets))
    ].copy()

    reports = []
    selected_rows = []
    for sample_id, group in candidates.groupby(candidates["sample_id"].astype(str), sort=True):
        if len(group) == 1:
            selected_rows.append(group.iloc[0].to_dict())
            continue

        # 多 mask
        if policy == "prefer_canonical_origin":
            same_origin = group[group["origin_sample_id"].astype(str) == str(sample_id)]
            if len(same_origin) == 1:
                chosen = same_origin.iloc[0].to_dict()
                dropped = group[group["annotation_id"] != chosen["annotation_id"]]
                reports.append(
                    {
                        "type": "multi_mask_resolved_by_canonical_origin",
                        "sample_id": str(sample_id),
                        "kept_annotation_id": str(chosen["annotation_id"]),
                        "dropped_annotation_ids": dropped["annotation_id"].astype(str).tolist(),
                        "dropped_origin_sample_ids": dropped["origin_sample_id"].astype(str).tolist(),
                    }
                )
                selected_rows.append(chosen)
                continue
        reports.append(
            {
                "type": "multi_mask_unresolved",
                "sample_id": str(sample_id),
                "annotation_ids": group["annotation_id"].astype(str).tolist(),
                "mask_paths": group["mask_path"].astype(str).tolist(),
            }
        )

    selected = pd.DataFrame(selected_rows)
    # 仅保留 clean samples 中存在的
    clean_ids = set(samples_clean["sample_id"].astype(str))
    if len(selected):
        selected = selected[selected["sample_id"].astype(str).isin(clean_ids)].copy()
    return selected, reports


def build_segmentation_manifest(
    processed_dir: str | Path,
    split_dir: str | Path,
    config: SegmentationConfig,
) -> tuple[pd.DataFrame, dict]:
    """构建 segmentation_manifest + audit。"""
    processed_dir = Path(processed_dir)
    split_dir = Path(split_dir)

    samples_clean = pd.read_parquet(processed_dir / "samples_clean.parquet")
    spatial_clean = pd.read_parquet(processed_dir / "spatial_clean.parquet")
    split_assignments = pd.read_parquet(split_dir / "split_assignments.parquet")
    supervision = None
    supervision_path = processed_dir / "supervision_assignments.parquet"
    if supervision_path.exists():
        supervision = pd.read_parquet(supervision_path)

    selected_masks, multi_mask_reports = select_segmentation_masks(
        spatial_clean, samples_clean, config
    )

    errors = []
    warnings = []
    for item in multi_mask_reports:
        if item["type"] == "multi_mask_unresolved":
            errors.append(f"unresolved multi-mask: {item['sample_id']}")
        else:
            warnings.append(
                f"multi-mask resolved for {item['sample_id']}: kept {item['kept_annotation_id']}"
            )

    # 禁止混入其他数据集
    if len(selected_masks):
        bad_ds = set(selected_masks["source_dataset"].astype(str)) - set(config.datasets)
        if bad_ds:
            errors.append(f"unexpected datasets in selection: {sorted(bad_ds)}")

    sample_meta = samples_clean.set_index(samples_clean["sample_id"].astype(str), drop=False)
    split_meta = split_assignments.set_index(split_assignments["sample_id"].astype(str), drop=False)

    # supervision pool lookup（spatial unit）
    pool_by_sample = {}
    if supervision is not None and len(supervision):
        spatial_sup = supervision[
            (supervision["unit_type"].astype(str) == "spatial")
            & (supervision["canonical_task"].astype(str) == config.task)
        ]
        for _, row in spatial_sup.iterrows():
            pool_by_sample[str(row["sample_id"])] = str(row["supervision_pool"])

    rows = []
    mask_value_audit = defaultdict(lambda: {"samples": 0, "unique_values_examples": []})
    foreground_ratios = []
    empty_masks = 0
    full_masks = 0
    missing_images = 0
    missing_masks = 0
    shape_mismatches = 0
    pixel_audits = []

    warn_min = float(config.foreground_ratio_warning.get("min", 0.01))
    warn_max = float(config.foreground_ratio_warning.get("max", 0.95))

    for _, mask_row in selected_masks.iterrows():
        sample_id = str(mask_row["sample_id"])
        dataset_name = str(mask_row["source_dataset"])
        if sample_id not in sample_meta.index:
            errors.append(f"mask sample missing in samples_clean: {sample_id}")
            continue
        if sample_id not in split_meta.index:
            errors.append(f"mask sample missing in split_assignments: {sample_id}")
            continue

        sample = sample_meta.loc[sample_id]
        if isinstance(sample, pd.DataFrame):
            sample = sample.iloc[0]
        split_row = split_meta.loc[sample_id]
        if isinstance(split_row, pd.DataFrame):
            split_row = split_row.iloc[0]

        image_path = str(sample["source_image_path"])
        mask_path = str(mask_row["mask_path"])
        split_name = str(split_row["split"])
        if split_name not in {
            str(config.data.get("train_split", "train")),
            str(config.data.get("val_split", "val")),
            str(config.data.get("test_split", "test")),
        }:
            errors.append(f"unexpected split for segmentation sample {sample_id}: {split_name}")
            continue

        image_exists = Path(image_path).exists()
        mask_exists = Path(mask_path).exists()
        if not image_exists:
            missing_images += 1
            errors.append(f"missing image: {sample_id} -> {image_path}")
            continue
        if not mask_exists:
            missing_masks += 1
            errors.append(f"missing mask: {sample_id} -> {mask_path}")
            continue

        image = load_image_rgb(image_path)
        mask_raw = load_mask_raw(mask_path)
        mask_binary = normalize_binary_mask(mask_raw)
        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask_binary.shape[:2]
        shape_match = (image_height == mask_height) and (image_width == mask_width)
        if not shape_match:
            shape_mismatches += 1
            errors.append(
                f"shape mismatch: {sample_id} image=({image_width},{image_height}) "
                f"mask=({mask_width},{mask_height})"
            )
            continue

        ratio = foreground_ratio(mask_binary)
        if ratio <= 0:
            empty_masks += 1
            errors.append(f"empty GT mask: {sample_id}")
            continue
        if ratio >= 1.0:
            full_masks += 1
            warnings.append(f"full mask: {sample_id}")
        if ratio < warn_min or ratio > warn_max:
            warnings.append(
                f"extreme foreground_ratio={ratio:.6f} sample={sample_id}"
            )

        unique_values = unique_pixel_values(mask_raw)
        mask_value_audit[dataset_name]["samples"] += 1
        if len(mask_value_audit[dataset_name]["unique_values_examples"]) < 5:
            mask_value_audit[dataset_name]["unique_values_examples"].append(
                {"sample_id": sample_id, "unique": unique_values}
            )
        pixel_audits.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "unique": unique_values,
                "foreground_ratio": ratio,
            }
        )
        foreground_ratios.append(ratio)

        # metadata 宽高以 image 为准；与 samples_clean 不一致时 warning
        meta_width = int(sample["width"]) if pd.notna(sample.get("width")) else image_width
        meta_height = int(sample["height"]) if pd.notna(sample.get("height")) else image_height
        if meta_width != image_width or meta_height != image_height:
            warnings.append(
                f"samples_clean size != image file for {sample_id}: "
                f"meta=({meta_width},{meta_height}) file=({image_width},{image_height})"
            )

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "split": split_name,
                "image_path": image_path,
                "mask_path": mask_path,
                "width": int(image_width),
                "height": int(image_height),
                "md5": str(sample["md5"]),
                "duplicate_group_id": sample.get("duplicate_group_id"),
                "foreground_ratio": float(ratio),
                "mask_valid": True,
                "image_mask_shape_match": True,
                "supervision_pool": pool_by_sample.get(sample_id, "unknown"),
                "annotation_id": str(mask_row["annotation_id"]),
                "origin_sample_id": str(mask_row.get("origin_sample_id", sample_id)),
                "contract_version": config.version,
                "split_group_id": str(split_row.get("split_group_id")),
            }
        )

    manifest = pd.DataFrame(rows)

    # leakage within segmentation manifest
    sample_leakage = 0
    md5_leakage = 0
    if len(manifest):
        for sample_id, group in manifest.groupby("sample_id"):
            if group["split"].nunique() > 1:
                sample_leakage += 1
                errors.append(f"sample leakage in seg manifest: {sample_id}")
        for md5_value, group in manifest.groupby("md5"):
            if group["split"].nunique() > 1:
                md5_leakage += 1
                errors.append(f"md5 leakage in seg manifest: {md5_value}")

        # split 继承校验
        for _, row in manifest.iterrows():
            expected = str(split_meta.loc[str(row["sample_id"])]["split"])
            if str(row["split"]) != expected:
                errors.append(
                    f"split drift: {row['sample_id']} manifest={row['split']} d2={expected}"
                )

        train_ids = set(manifest.loc[manifest["split"] == "train", "sample_id"])
        val_ids = set(manifest.loc[manifest["split"] == "val", "sample_id"])
        test_ids = set(manifest.loc[manifest["split"] == "test", "sample_id"])
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            errors.append("split sample overlap in segmentation manifest")

    def _count(dataset_name: str | None = None, split_name: str | None = None) -> int:
        frame = manifest
        if dataset_name is not None:
            frame = frame[frame["dataset"] == dataset_name]
        if split_name is not None:
            frame = frame[frame["split"] == split_name]
        return int(len(frame))

    audit = {
        "contract_version": config.version,
        "task": config.task,
        "datasets": config.datasets,
        "foreground_rule": config.foreground_rule,
        "total_samples": int(len(manifest)),
        "per_split": {
            "train": _count(split_name="train"),
            "val": _count(split_name="val"),
            "test": _count(split_name="test"),
        },
        "per_dataset": {
            dataset_name: {
                "total": _count(dataset_name=dataset_name),
                "train": _count(dataset_name, "train"),
                "val": _count(dataset_name, "val"),
                "test": _count(dataset_name, "test"),
            }
            for dataset_name in config.datasets
        },
        "image_mask_pairing_rate": 1.0 if len(manifest) and not errors else (
            float(len(manifest) / max(len(selected_masks), 1))
        ),
        "missing_images": int(missing_images),
        "missing_masks": int(missing_masks),
        "shape_mismatches": int(shape_mismatches),
        "empty_masks": int(empty_masks),
        "full_masks": int(full_masks),
        "duplicate_sample_ids": int(
            manifest["sample_id"].duplicated().sum() if len(manifest) else 0
        ),
        "sample_leakage": int(sample_leakage),
        "md5_leakage": int(md5_leakage),
        "foreground_ratio": _percentile_stats(foreground_ratios),
        "mask_pixel_value_audit": dict(mask_value_audit),
        "multi_mask_reports": multi_mask_reports,
        "errors": errors,
        "warnings": warnings[:200],
        "warnings_count": len(warnings),
        "errors_count": len(errors),
    }
    return manifest, audit
