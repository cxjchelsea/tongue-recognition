"""D3-A segmentation manifest / contract 校验。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import SegmentationConfig


def validate_segmentation(
    segmentation_dir: str | Path,
    config_path: str | Path | None = None,
    split_dir: str | Path | None = None,
):
    """硬校验：pairing / leakage / 数据集白名单 / split 继承。"""
    root = Path(segmentation_dir)
    errors, warnings = [], []

    manifest_path = root / "segmentation_manifest.parquet"
    meta_path = root / "segmentation_metadata.json"
    if not manifest_path.exists():
        return ["missing: segmentation_manifest.parquet"], warnings
    if not meta_path.exists():
        errors.append("missing: segmentation_metadata.json")

    manifest = pd.read_parquet(manifest_path)
    config = SegmentationConfig(config_path) if config_path else None

    if manifest.empty:
        errors.append("segmentation_manifest is empty")
        return errors, warnings

    if manifest["sample_id"].duplicated().any():
        errors.append("duplicate sample_id in segmentation_manifest")

    allowed = set(config.datasets) if config else {"biohit", "tongueset3"}
    unexpected = set(manifest["dataset"].astype(str)) - allowed
    if unexpected:
        errors.append(f"unexpected datasets: {sorted(unexpected)}")

    # 路径存在
    for _, row in manifest.iterrows():
        if not Path(str(row["image_path"])).exists():
            errors.append(f"missing image: {row['sample_id']}")
            break
    for _, row in manifest.iterrows():
        if not Path(str(row["mask_path"])).exists():
            errors.append(f"missing mask: {row['sample_id']}")
            break

    if (~manifest["mask_valid"].astype(bool)).any():
        errors.append("mask_valid contains false")
    if (~manifest["image_mask_shape_match"].astype(bool)).any():
        errors.append("image_mask_shape_match contains false")
    if (manifest["foreground_ratio"].astype(float) <= 0).any():
        errors.append("empty GT mask present")
    if (manifest["foreground_ratio"].astype(float) >= 1.0).any():
        warnings.append("full mask present")

    # split overlap
    split_sets = {
        name: set(manifest.loc[manifest["split"].astype(str) == name, "sample_id"].astype(str))
        for name in ["train", "val", "test"]
    }
    if split_sets["train"] & split_sets["val"]:
        errors.append("train/val sample overlap")
    if split_sets["train"] & split_sets["test"]:
        errors.append("train/test sample overlap")
    if split_sets["val"] & split_sets["test"]:
        errors.append("val/test sample overlap")

    md5_sets = {
        name: set(manifest.loc[manifest["split"].astype(str) == name, "md5"].astype(str))
        for name in ["train", "val", "test"]
    }
    if md5_sets["train"] & md5_sets["val"]:
        errors.append("train/val md5 overlap")
    if md5_sets["train"] & md5_sets["test"]:
        errors.append("train/test md5 overlap")
    if md5_sets["val"] & md5_sets["test"]:
        errors.append("val/test md5 overlap")

    # D2 split 继承
    if split_dir is not None:
        split_assignments = pd.read_parquet(Path(split_dir) / "split_assignments.parquet")
        split_map = dict(
            zip(
                split_assignments["sample_id"].astype(str),
                split_assignments["split"].astype(str),
            )
        )
        for _, row in manifest.iterrows():
            sample_id = str(row["sample_id"])
            if sample_id not in split_map:
                errors.append(f"sample not in D2 split: {sample_id}")
                break
            if split_map[sample_id] != str(row["split"]):
                errors.append(
                    f"split mismatch vs D2: {sample_id} "
                    f"seg={row['split']} d2={split_map[sample_id]}"
                )
                break

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in [
            "missing_images",
            "missing_masks",
            "shape_mismatches",
            "empty_masks",
            "sample_leakage",
            "md5_leakage",
        ]:
            if int(meta.get(key, 0)) > 0:
                errors.append(f"metadata {key}={meta.get(key)}")
        if int(meta.get("errors_count", 0)) > 0:
            errors.append(f"metadata errors_count={meta.get('errors_count')}")

    return errors, warnings
