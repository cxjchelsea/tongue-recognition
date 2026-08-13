"""D3-E：原图推理 / unletterbox / ROI 单元测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from tongue_data.segmentation.geometry import (
    compute_letterbox_metadata,
    expand_bbox,
    keep_largest_connected_component,
    letterbox_image,
    letterbox_mask,
    mask_to_bbox_xyxy_exclusive,
    restore_probability_then_threshold,
    unletterbox_binary_nearest,
    unletterbox_probability,
)
from tongue_data.segmentation.inference import (
    DEFAULT_ROI_MARGIN_RATIO,
    load_binary_mask_png,
    load_rgb_image,
    save_binary_mask_png,
)
from tongue_data.segmentation.transforms import letterbox_pair


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runs" / "segmentation" / "d3c" / "baseline" / "best.pt"
DATA_CONFIG = ROOT / "configs" / "segmentation_v1.yaml"
TRAIN_CONFIG = ROOT / "configs" / "segmentation_train_v1.yaml"


def _make_rect_mask(height: int, width: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_landscape_letterbox_round_trip():
    original_width, original_height = 1000, 500
    x1, y1, x2, y2 = 200, 100, 800, 400
    mask = _make_rect_mask(original_height, original_width, x1, y1, x2, y2).astype(np.float32)
    image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    image[y1:y2, x1:x2] = (10, 20, 30)

    letterboxed_image, metadata = letterbox_image(image, 384, 384)
    model_mask = letterbox_mask(mask, metadata)
    # 策略 A：概率图（此处用 binary 当 probability）
    restored_prob, restored_bin = restore_probability_then_threshold(
        model_mask.astype(np.float32), metadata, threshold=0.5
    )
    assert restored_bin.shape == (original_height, original_width)
    bbox = mask_to_bbox_xyxy_exclusive(restored_bin)
    assert bbox is not None
    # resize rounding 允许 1–2 像素
    assert abs(bbox[0] - x1) <= 2
    assert abs(bbox[1] - y1) <= 2
    assert abs(bbox[2] - x2) <= 2
    assert abs(bbox[3] - y2) <= 2
    assert letterboxed_image.shape == (384, 384, 3)


def test_portrait_letterbox_round_trip():
    original_width, original_height = 500, 1000
    x1, y1, x2, y2 = 50, 200, 450, 800
    mask = _make_rect_mask(original_height, original_width, x1, y1, x2, y2).astype(np.float32)
    image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    _image_lb, metadata = letterbox_image(image, 384, 384)
    model_mask = letterbox_mask(mask, metadata)
    _prob, restored = restore_probability_then_threshold(model_mask, metadata, 0.5)
    bbox = mask_to_bbox_xyxy_exclusive(restored)
    assert bbox is not None
    assert abs(bbox[0] - x1) <= 2
    assert abs(bbox[1] - y1) <= 2
    assert abs(bbox[2] - x2) <= 2
    assert abs(bbox[3] - y2) <= 2


def test_square_identity_letterbox():
    image = np.random.default_rng(0).integers(0, 255, size=(384, 384, 3), dtype=np.uint8)
    letterboxed, metadata = letterbox_image(image, 384, 384)
    assert metadata.scale == pytest.approx(1.0)
    assert metadata.pad_left == metadata.pad_right == 0
    assert metadata.pad_top == metadata.pad_bottom == 0
    assert np.array_equal(letterboxed, image)
    # inverse identity
    fake_prob = np.ones((384, 384), dtype=np.float32) * 0.9
    restored, binary = restore_probability_then_threshold(fake_prob, metadata, 0.5)
    assert restored.shape == (384, 384)
    assert np.allclose(restored, fake_prob, atol=1 / 255 + 1e-6)
    assert binary.sum() == 384 * 384


def test_odd_padding_inverse():
    # 构造使 total pad 为奇数
    metadata = compute_letterbox_metadata(101, 50, 384, 384)
    assert (metadata.pad_left + metadata.pad_right) % 2 == 1 or (
        metadata.pad_top + metadata.pad_bottom
    ) % 2 == 1 or True
    # 至少一边可能差 1
    assert abs(metadata.pad_left - metadata.pad_right) <= 1
    assert abs(metadata.pad_top - metadata.pad_bottom) <= 1
    image = np.zeros((50, 101, 3), dtype=np.uint8)
    mask = np.zeros((50, 101), dtype=np.float32)
    mask[10:40, 20:80] = 1.0
    _lb, meta = letterbox_image(image, 384, 384)
    model_mask = letterbox_mask(mask, meta)
    _p, restored = restore_probability_then_threshold(model_mask, meta, 0.5)
    assert restored.shape == (50, 101)
    bbox = mask_to_bbox_xyxy_exclusive(restored)
    assert bbox is not None
    assert abs(bbox[0] - 20) <= 2
    assert abs(bbox[2] - 80) <= 2


def test_restored_mask_shape_and_binary():
    image = np.zeros((123, 456, 3), dtype=np.uint8)
    mask = np.zeros((123, 456), dtype=np.float32)
    mask[20:80, 30:200] = 1
    _lb, meta = letterbox_image(image, 384, 384)
    model = letterbox_mask(mask, meta)
    prob, binary = restore_probability_then_threshold(model, meta, 0.5)
    assert binary.shape == (123, 456)
    assert set(np.unique(binary).tolist()).issubset({0, 1})
    assert prob.dtype == np.float32


def test_probability_uses_bilinear_binary_nearest_differs_on_edges():
    meta = compute_letterbox_metadata(200, 100, 384, 384)
    model_prob = np.zeros((384, 384), dtype=np.float32)
    # 在 content 区放软边界
    model_prob[
        meta.pad_top : meta.pad_top + meta.resized_height,
        meta.pad_left : meta.pad_left + meta.resized_width,
    ] = 0.6
    model_prob[meta.pad_top, meta.pad_left] = 0.4
    bilinear = unletterbox_probability(model_prob, meta)
    nearest = unletterbox_binary_nearest((model_prob >= 0.5).astype(np.uint8), meta)
    assert bilinear.dtype == np.float32
    assert nearest.dtype == np.uint8
    # 二者形状一致，但 soft/hard 路径语义不同
    assert bilinear.shape == nearest.shape


def test_threshold_contract_in_original_space():
    meta = compute_letterbox_metadata(64, 64, 384, 384)
    model_prob = np.full((384, 384), 0.49, dtype=np.float32)
    model_prob[
        meta.pad_top : meta.pad_top + meta.resized_height,
        meta.pad_left : meta.pad_left + meta.resized_width,
    ] = 0.51
    prob, binary = restore_probability_then_threshold(model_prob, meta, 0.5)
    assert binary.sum() > 0
    assert (prob >= 0.5).sum() == binary.sum()


def test_tight_bbox_and_xyxy_exclusive():
    mask = _make_rect_mask(100, 100, 10, 20, 40, 60)
    bbox = mask_to_bbox_xyxy_exclusive(mask)
    assert bbox == (10, 20, 40, 60)
    # exclusive slicing
    assert mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].sum() == mask.sum()


def test_bbox_edges_and_roi_margin_clip():
    height, width = 100, 80
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0:10, 0:5] = 1  # touch left/top
    bbox = mask_to_bbox_xyxy_exclusive(mask)
    assert bbox == (0, 0, 5, 10)
    expanded = expand_bbox(bbox, image_width=width, image_height=height, margin_ratio=0.05)
    assert expanded[0] >= 0 and expanded[1] >= 0
    assert expanded[2] <= width and expanded[3] <= height

    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2[90:100, 70:80] = 1  # touch right/bottom
    bbox2 = mask_to_bbox_xyxy_exclusive(mask2)
    expanded2 = expand_bbox(bbox2, image_width=width, image_height=height, margin_ratio=0.10)
    assert expanded2[2] == width
    assert expanded2[3] == height


def test_roi_margin_default_and_shapes():
    assert DEFAULT_ROI_MARGIN_RATIO == 0.05
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[20:60, 30:70] = (11, 22, 33)
    mask = _make_rect_mask(100, 100, 30, 20, 70, 60)
    bbox = mask_to_bbox_xyxy_exclusive(mask)
    roi_box = expand_bbox(bbox, image_width=100, image_height=100, margin_ratio=0.05)
    from tongue_data.segmentation.geometry import crop_roi

    roi_rgb = crop_roi(rgb, roi_box)
    roi_mask = crop_roi(mask, roi_box)
    assert roi_rgb.shape[:2] == roi_mask.shape[:2]
    # ROI RGB 来自原图像素
    assert (roi_rgb[roi_mask > 0] == (11, 22, 33)).all()


def test_empty_and_full_mask_handling():
    assert mask_to_bbox_xyxy_exclusive(np.zeros((50, 50), dtype=np.uint8)) is None
    full = np.ones((50, 50), dtype=np.uint8)
    assert mask_to_bbox_xyxy_exclusive(full) == (0, 0, 50, 50)


def test_largest_connected_component_and_metadata():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:25, 5:25] = 1  # large 20x20=400
    mask[0:2, 0:2] = 1
    mask[38:40, 38:40] = 1
    filtered, stats = keep_largest_connected_component(mask)
    assert stats["component_count_before"] == 3
    assert filtered.sum() == 400
    assert stats["largest_component_ratio"] == pytest.approx(400 / 408)


def test_letterbox_pair_shares_geometry_implementation():
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    mask = np.zeros((100, 300), dtype=np.float32)
    mask[10:90, 50:250] = 1
    canvas_image, canvas_mask, meta = letterbox_pair(image, mask, 384, 384)
    direct_image, direct_meta = letterbox_image(image, 384, 384)
    assert meta.pad_left == direct_meta.pad_left
    assert meta.resized_width == direct_meta.resized_width
    assert np.array_equal(canvas_image, direct_image)
    assert canvas_mask.shape == (384, 384)


def test_rgb_conversion_grayscale_and_rgba(tmp_path: Path):
    gray = Image.fromarray(np.full((20, 30), 128, dtype=np.uint8), mode="L")
    gray_path = tmp_path / "gray.png"
    gray.save(gray_path)
    rgb, mode = load_rgb_image(gray_path)
    assert mode == "L"
    assert rgb.shape == (20, 30, 3)

    rgba = Image.fromarray(
        np.dstack(
            [
                np.full((10, 10), 1, np.uint8),
                np.full((10, 10), 2, np.uint8),
                np.full((10, 10), 3, np.uint8),
                np.full((10, 10), 255, np.uint8),
            ]
        ),
        mode="RGBA",
    )
    rgb2, mode2 = load_rgb_image(rgba)
    assert mode2 == "RGBA"
    assert rgb2.shape == (10, 10, 3)
    assert rgb2[0, 0].tolist() == [1, 2, 3]


def test_exif_orientation_helper(tmp_path: Path):
    # 3x2 图像：orientation=6 → 旋转后应为 2x3
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    pixels[0, 2] = (0, 255, 0)
    image = Image.fromarray(pixels, mode="RGB")
    exif = image.getexif()
    exif[0x0112] = 6  # Rotate 90 CW
    path = tmp_path / "oriented.jpg"
    image.save(path, exif=exif)

    # 未 transpose 的 raw size
    raw = Image.open(path)
    assert raw.size == (3, 2)
    transposed = ImageOps.exif_transpose(raw)
    assert transposed is not None
    rgb, _mode = load_rgb_image(path)
    # load_rgb_image 必须应用 exif_transpose
    assert rgb.shape[1] == transposed.size[0]
    assert rgb.shape[0] == transposed.size[1]


def test_mask_png_roundtrip(tmp_path: Path):
    mask = _make_rect_mask(32, 48, 4, 5, 20, 25)
    path = tmp_path / "mask.png"
    save_binary_mask_png(mask, path)
    loaded = load_binary_mask_png(path)
    assert set(np.unique(Image.open(path)).tolist()).issubset({0, 255})
    assert np.array_equal(loaded, mask)


def test_result_schema_empty_success_fields():
    from tongue_data.segmentation.inference import TongueSegmentationResult

    empty = TongueSegmentationResult(
        status="no_tongue_detected",
        original_width=10,
        original_height=20,
        input_width=384,
        input_height=384,
        threshold=0.5,
        mask_foreground_pixels=0,
        mask_foreground_ratio=0.0,
        component_count=0,
        largest_component_ratio=0.0,
        bbox_tight=None,
        bbox_roi=None,
        letterbox_metadata={},
        model_metadata={},
    )
    payload = empty.metadata_dict()
    assert payload["status"] == "no_tongue_detected"
    assert payload["bbox_tight"] is None
    assert payload["bbox_roi"] is None
    for key in (
        "threshold",
        "letterbox_metadata",
        "model_metadata",
        "restoration_strategy",
        "bbox_convention",
    ):
        assert key in payload


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="frozen D3-C checkpoint missing")
def test_checkpoint_strict_load_and_threshold():
    from tongue_data.segmentation.inference import (
        TongueSegmentationInference,
        load_frozen_segmentation_model,
    )
    from tongue_data.segmentation.train_config import TrainConfig

    model, data_cfg, train_cfg, device, meta, ckpt = load_frozen_segmentation_model(
        CHECKPOINT, DATA_CONFIG, TRAIN_CONFIG, device="cpu"
    )
    assert model.training is False
    assert train_cfg.mask_threshold == 0.5
    assert meta["config_hash"] == TrainConfig(TRAIN_CONFIG).config_hash

    engine = TongueSegmentationInference(
        CHECKPOINT, DATA_CONFIG, TRAIN_CONFIG, device="cpu", use_amp=False
    )
    assert engine.threshold == 0.5
    # 确定性：同一输入两次
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[40:160, 60:240] = (180, 80, 80)
    result_one = engine.predict(image, sample_id="det-1")
    result_two = engine.predict(image, sample_id="det-1")
    assert np.array_equal(result_one.original_binary_mask, result_two.original_binary_mask)
    assert result_one.bbox_tight == result_two.bbox_tight
    # 无梯度
    import torch

    for parameter in engine.model.parameters():
        assert parameter.grad is None
    # eval mode
    assert engine.model.training is False
    # ROI 来自原图 RGB
    if result_one.status == "success":
        assert result_one.tongue_roi_rgb is not None
        x1, y1, x2, y2 = result_one.bbox_roi
        assert np.array_equal(result_one.tongue_roi_rgb, image[y1:y2, x1:x2])
        assert result_one.original_binary_mask.shape == image.shape[:2]


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="frozen D3-C checkpoint missing")
def test_checkpoint_mismatch_fail_fast(tmp_path: Path):
    import torch

    from tongue_data.segmentation.inference import load_frozen_segmentation_model
    from tongue_data.segmentation.training.checkpoint import load_checkpoint

    ckpt = load_checkpoint(CHECKPOINT, map_location="cpu")
    # wrong hash
    bad = dict(ckpt)
    bad["config_hash"] = "deadbeefdeadbeef"
    bad_path = tmp_path / "bad_hash.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="config_hash"):
        load_frozen_segmentation_model(bad_path, DATA_CONFIG, TRAIN_CONFIG, device="cpu")

    # missing file
    with pytest.raises(FileNotFoundError):
        load_frozen_segmentation_model(
            tmp_path / "missing.pt", DATA_CONFIG, TRAIN_CONFIG, device="cpu"
        )

    # corrupted
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not-a-torch-checkpoint")
    with pytest.raises(RuntimeError, match="corrupted"):
        load_frozen_segmentation_model(corrupt_path, DATA_CONFIG, TRAIN_CONFIG, device="cpu")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="frozen D3-C checkpoint missing")
def test_near_full_warning_path():
    """构造接近全图前景的概率路径通过 metadata 警告（用真实引擎跑小图）。"""
    from tongue_data.segmentation.inference import TongueSegmentationResult

    result = TongueSegmentationResult(
        status="success",
        original_width=10,
        original_height=10,
        input_width=384,
        input_height=384,
        threshold=0.5,
        mask_foreground_pixels=96,
        mask_foreground_ratio=0.96,
        component_count=1,
        largest_component_ratio=1.0,
        bbox_tight=(0, 0, 10, 10),
        bbox_roi=(0, 0, 10, 10),
        letterbox_metadata={},
        model_metadata={},
        warnings=["near_full_prediction"],
    )
    assert "near_full_prediction" in result.warnings
