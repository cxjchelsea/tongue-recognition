"""D3-A：Segmentation Dataset Contract 测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tongue_data.segmentation.config import SegmentationConfig
from tongue_data.segmentation.dataset import TongueSegmentationDataset
from tongue_data.segmentation.mask_ops import normalize_binary_mask
from tongue_data.segmentation.manifest import build_segmentation_manifest, select_segmentation_masks
from tongue_data.segmentation.metrics import (
    dice_coefficient,
    iou_score,
    binarize_prediction,
)
from tongue_data.segmentation.transforms import letterbox_pair, preprocess_pair
from tongue_data.segmentation.validators import validate_segmentation


CONFIG = SegmentationConfig("configs/segmentation_v1.yaml")


def _write_image(path: Path, width: int = 32, height: int = 24, value: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_mask(path: Path, width: int = 32, height: int = 24, mode: str = "255", fg_ratio: float = 0.25):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    fg_h = max(1, int(height * fg_ratio))
    fg_w = max(1, int(width * fg_ratio))
    if mode == "255":
        mask[:fg_h, :fg_w] = 255
    elif mode == "1":
        mask[:fg_h, :fg_w] = 1
    elif mode == "bool":
        mask = mask.astype(bool)
        mask[:fg_h, :fg_w] = True
        Image.fromarray(mask).save(path)
        return
    elif mode == "empty":
        pass
    Image.fromarray(mask, mode="L").save(path)


def _toy_tables(tmp_path: Path, mask_mode: str = "1"):
    image_a = tmp_path / "img" / "a.jpg"
    image_b = tmp_path / "img" / "b.jpg"
    image_c = tmp_path / "img" / "c.jpg"
    mask_a = tmp_path / "mask" / "a.png"
    mask_b = tmp_path / "mask" / "b.png"
    mask_c = tmp_path / "mask" / "c.png"
    _write_image(image_a)
    _write_image(image_b)
    _write_image(image_c)
    _write_mask(mask_a, mode=mask_mode)
    _write_mask(mask_b, mode=mask_mode)
    _write_mask(mask_c, mode=mask_mode)

    samples = pd.DataFrame(
        [
            {
                "sample_id": "biohit::a",
                "dataset": "biohit",
                "source_image_path": str(image_a),
                "md5": "md5a",
                "width": 32,
                "height": 24,
                "duplicate_group_id": "dup::biohit::md5a",
            },
            {
                "sample_id": "tongueset3::b",
                "dataset": "tongueset3",
                "source_image_path": str(image_b),
                "md5": "md5b",
                "width": 32,
                "height": 24,
                "duplicate_group_id": "dup::tongueset3::md5b",
            },
            {
                "sample_id": "tongueset3::c",
                "dataset": "tongueset3",
                "source_image_path": str(image_c),
                "md5": "md5c",
                "width": 32,
                "height": 24,
                "duplicate_group_id": "dup::tongueset3::md5c",
            },
        ]
    )
    spatial = pd.DataFrame(
        [
            {
                "sample_id": "biohit::a",
                "annotation_id": "biohit::a::m0",
                "annotation_task": "segmentation.tongue",
                "canonical_label": "tongue",
                "annotation_type": "mask",
                "mask_path": str(mask_a),
                "source_dataset": "biohit",
                "origin_sample_id": "biohit::a",
            },
            {
                "sample_id": "tongueset3::b",
                "annotation_id": "tongueset3::b::m0",
                "annotation_task": "segmentation.tongue",
                "canonical_label": "tongue",
                "annotation_type": "mask",
                "mask_path": str(mask_b),
                "source_dataset": "tongueset3",
                "origin_sample_id": "tongueset3::b",
            },
            {
                "sample_id": "tongueset3::c",
                "annotation_id": "tongueset3::c::m0",
                "annotation_task": "segmentation.tongue",
                "canonical_label": "tongue",
                "annotation_type": "mask",
                "mask_path": str(mask_c),
                "source_dataset": "tongueset3",
                "origin_sample_id": "tongueset3::c",
            },
            # 不应进入：TonguExpert
            {
                "sample_id": "tonguexpert::x",
                "annotation_id": "tonguexpert::x::m0",
                "annotation_task": "segmentation.tongue",
                "canonical_label": "tongue",
                "annotation_type": "mask",
                "mask_path": str(mask_a),
                "source_dataset": "tonguexpert",
                "origin_sample_id": "tonguexpert::x",
            },
        ]
    )
    splits = pd.DataFrame(
        [
            {"sample_id": "biohit::a", "dataset": "biohit", "split": "train", "split_group_id": "g1", "md5": "md5a"},
            {"sample_id": "tongueset3::b", "dataset": "tongueset3", "split": "val", "split_group_id": "g2", "md5": "md5b"},
            {"sample_id": "tongueset3::c", "dataset": "tongueset3", "split": "test", "split_group_id": "g3", "md5": "md5c"},
        ]
    )
    return samples, spatial, splits


def _dump_processed(tmp_path: Path, samples, spatial, splits):
    processed = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    processed.mkdir()
    split_dir.mkdir()
    samples.to_parquet(processed / "samples_clean.parquet", index=False)
    spatial.to_parquet(processed / "spatial_clean.parquet", index=False)
    splits.to_parquet(split_dir / "split_assignments.parquet", index=False)
    return processed, split_dir


def test_only_biohit_tongueset3_allowed(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    selected, _ = select_segmentation_masks(spatial, samples, CONFIG)
    assert set(selected["source_dataset"]) <= {"biohit", "tongueset3"}
    assert "tonguexpert" not in set(selected["source_dataset"])


def test_d2_split_inherited(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["errors_count"] == 0
    merged = manifest.merge(splits, on="sample_id", suffixes=("_seg", "_d2"))
    assert (merged["split_seg"] == merged["split_d2"]).all()


def test_no_sample_overlap(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, _ = build_segmentation_manifest(processed, split_dir, CONFIG)
    train = set(manifest.loc[manifest.split == "train", "sample_id"])
    val = set(manifest.loc[manifest.split == "val", "sample_id"])
    test = set(manifest.loc[manifest.split == "test", "sample_id"])
    assert not (train & val or train & test or val & test)


def test_no_md5_overlap(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["md5_leakage"] == 0


def test_mask_gt0_to_one():
    mask = np.array([[0, 255], [1, 128]], dtype=np.uint8)
    binary = normalize_binary_mask(mask)
    assert set(np.unique(binary).tolist()) <= {0.0, 1.0}
    assert binary[0, 1] == 1.0
    assert binary[1, 0] == 1.0
    assert binary[1, 1] == 1.0


def test_tongueset3_value_one_foreground(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path, mask_mode="1")
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["errors_count"] == 0
    assert (manifest["foreground_ratio"] > 0).all()
    # 证明不是 mask==255 逻辑：纯 1 值 mask 也能识别
    assert manifest.loc[manifest.dataset == "tongueset3", "foreground_ratio"].iloc[0] > 0


def test_mask_resize_nearest_binary():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.float32)
    mask[5:15, 5:20] = 1.0
    _, mask_out, _ = letterbox_pair(image, mask, 64, 64)
    assert set(np.unique(mask_out).tolist()) <= {0.0, 1.0}


def test_geometry_transform_sync():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :] = 10
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    image_out, mask_out, meta = letterbox_pair(image, mask, 64, 64)
    assert image_out.shape[:2] == mask_out.shape[:2] == (64, 64)
    assert meta.pad_left == meta.pad_top  # square source -> square letterbox pads equal-ish


def test_val_transform_deterministic():
    image = np.random.default_rng(0).integers(0, 255, size=(40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.float32)
    mask[5:20, 5:25] = 1.0
    out1 = preprocess_pair(image, mask, CONFIG, "val")
    out2 = preprocess_pair(image, mask, CONFIG, "val")
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])


def test_missing_mask_fails(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    spatial.loc[spatial.sample_id == "biohit::a", "mask_path"] = str(tmp_path / "nope.png")
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    _manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["missing_masks"] >= 1
    assert audit["errors_count"] >= 1


def test_missing_image_fails(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    samples.loc[samples.sample_id == "biohit::a", "source_image_path"] = str(tmp_path / "missing.jpg")
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    _manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["missing_images"] >= 1
    assert audit["errors_count"] >= 1


def test_shape_mismatch_fails(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    # 改写 mask 尺寸
    bad_mask = tmp_path / "mask" / "bad.png"
    _write_mask(bad_mask, width=16, height=16, mode="1")
    spatial.loc[spatial.sample_id == "biohit::a", "mask_path"] = str(bad_mask)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    _manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["shape_mismatches"] >= 1
    assert audit["errors_count"] >= 1


def test_empty_gt_mask_fails(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    empty_mask = tmp_path / "mask" / "empty.png"
    _write_mask(empty_mask, mode="empty")
    spatial.loc[spatial.sample_id == "biohit::a", "mask_path"] = str(empty_mask)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    _manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["empty_masks"] >= 1
    assert audit["errors_count"] >= 1


def test_foreground_ratio_correct():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[:5, :5] = 1.0
    assert abs(float(mask.mean()) - 0.25) < 1e-6


def test_dataset_output_shapes(tmp_path: Path):
    pytest.importorskip("torch")
    samples, spatial, splits = _toy_tables(tmp_path)
    # 全部设 train 以便构建 dataset
    splits["split"] = "train"
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["errors_count"] == 0
    dataset = TongueSegmentationDataset(manifest, CONFIG, split="train")
    item = dataset[0]
    assert tuple(item["image"].shape) == (3, CONFIG.input_height, CONFIG.input_width)
    assert tuple(item["mask"].shape) == (1, CONFIG.input_height, CONFIG.input_width)


def test_mask_range_binary(tmp_path: Path):
    pytest.importorskip("torch")
    samples, spatial, splits = _toy_tables(tmp_path)
    splits["split"] = "val"
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, _ = build_segmentation_manifest(processed, split_dir, CONFIG)
    dataset = TongueSegmentationDataset(manifest, CONFIG, split="val")
    item = dataset[0]
    unique = set(item["mask"].unique().tolist())
    assert unique.issubset({0.0, 1.0})


def test_dice_perfect_is_one():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:4, :4] = 1.0
    assert dice_coefficient(mask, mask, threshold=0.5) == pytest.approx(1.0)


def test_dice_zero_overlap_is_zero():
    target = np.zeros((8, 8), dtype=np.float32)
    target[:4, :4] = 1.0
    pred = np.zeros((8, 8), dtype=np.float32)
    pred[4:, 4:] = 1.0
    assert dice_coefficient(pred, target, threshold=0.5) == pytest.approx(0.0, abs=1e-5)


def test_iou_perfect_is_one():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    assert iou_score(mask, mask, threshold=0.5) == pytest.approx(1.0)


def test_metric_threshold_half():
    target = np.ones((4, 4), dtype=np.float32)
    pred = np.full((4, 4), 0.49, dtype=np.float32)
    assert binarize_prediction(pred, 0.5).sum() == 0
    pred2 = np.full((4, 4), 0.5, dtype=np.float32)
    assert binarize_prediction(pred2, 0.5).sum() == 16


def test_metadata_preserved(tmp_path: Path):
    pytest.importorskip("torch")
    samples, spatial, splits = _toy_tables(tmp_path)
    splits["split"] = "test"
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, _ = build_segmentation_manifest(processed, split_dir, CONFIG)
    dataset = TongueSegmentationDataset(manifest, CONFIG, split="test")
    item = dataset[0]
    assert "sample_id" in item and "dataset" in item
    assert item["dataset"] in {"biohit", "tongueset3"}


def test_preprocess_deterministic_same_seed():
    image = np.random.default_rng(1).integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
    mask = np.zeros((30, 40), dtype=np.float32)
    mask[3:15, 3:20] = 1.0
    rng1 = np.random.default_rng(20260813)
    rng2 = np.random.default_rng(20260813)
    out1 = preprocess_pair(image, mask, CONFIG, "train", rng=rng1)
    out2 = preprocess_pair(image, mask, CONFIG, "train", rng=rng2)
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])


def test_per_domain_audit_keys(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    _manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert "biohit" in audit["per_dataset"]
    assert "tongueset3" in audit["per_dataset"]
    assert set(audit["per_dataset"]["biohit"]) >= {"train", "val", "test", "total"}


def test_multi_mask_canonical_origin_resolution(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    # 给 tongueset3::b 增加 alias mask
    alias_mask = tmp_path / "mask" / "alias.png"
    _write_mask(alias_mask, mode="1", fg_ratio=0.4)
    extra = spatial.iloc[1].to_dict()
    extra["annotation_id"] = "tongueset3::b::m1"
    extra["mask_path"] = str(alias_mask)
    extra["origin_sample_id"] = "tongueset3::b_alias"
    spatial = pd.concat([spatial, pd.DataFrame([extra])], ignore_index=True)
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["errors_count"] == 0
    assert any(item["type"] == "multi_mask_resolved_by_canonical_origin" for item in audit["multi_mask_reports"])
    kept = manifest.loc[manifest.sample_id == "tongueset3::b"].iloc[0]
    assert kept["origin_sample_id"] == "tongueset3::b"


def test_validate_segmentation_pass(tmp_path: Path):
    samples, spatial, splits = _toy_tables(tmp_path)
    # 需要 train/val/test 都有才能更完整；这里仅检查无硬错误路径
    processed, split_dir = _dump_processed(tmp_path, samples, spatial, splits)
    manifest, audit = build_segmentation_manifest(processed, split_dir, CONFIG)
    assert audit["errors_count"] == 0
    out = tmp_path / "seg"
    out.mkdir()
    manifest.to_parquet(out / "segmentation_manifest.parquet", index=False)
    meta = {
        "missing_images": 0,
        "missing_masks": 0,
        "shape_mismatches": 0,
        "empty_masks": 0,
        "sample_leakage": 0,
        "md5_leakage": 0,
        "errors_count": 0,
    }
    (out / "segmentation_metadata.json").write_text(
        __import__("json").dumps(meta), encoding="utf-8"
    )
    errors, _warnings = validate_segmentation(out, "configs/segmentation_v1.yaml", split_dir)
    assert errors == []
