"""D3-E：原图级舌体分割推理 + unletterbox + ROI。

原则：
- 下游表型 / 颜色分析必须基于 original RGB + original-resolution mask
- 禁止把 384×384 normalized tensor 当作 phenotype 输入
- 不训练、不改 threshold、不改 D3-C weights
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import SegmentationConfig
from .geometry import (
    apply_mask_to_rgb,
    bbox_ratios,
    crop_roi,
    expand_bbox,
    keep_largest_connected_component,
    letterbox_image,
    mask_to_bbox_xyxy_exclusive,
    restore_probability_then_threshold,
)
from .model import build_segmentation_model, count_parameters
from .reproducibility import resolve_device
from .train_config import TrainConfig
from .training.checkpoint import load_checkpoint
from .training.evaluate import verify_checkpoint_integrity
from .transforms import normalize_image

# D3-E v1 固定默认策略（写入 Freeze / Contract）
DEFAULT_ROI_MARGIN_RATIO = 0.05
DEFAULT_KEEP_LARGEST_COMPONENT = True
DEFAULT_NEAR_FULL_RATIO = 0.95
RESTORATION_STRATEGY = (
    "probability_remove_pad_bilinear_then_threshold_original"
)
BBOX_CONVENTION = "xyxy_exclusive"


@dataclass
class TongueSegmentationResult:
    """原图级分割推理结果（内部可含 ndarray；JSON 另存路径）。"""

    status: str
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    threshold: float
    mask_foreground_pixels: int
    mask_foreground_ratio: float
    component_count: int
    largest_component_ratio: float
    bbox_tight: tuple[int, int, int, int] | None
    bbox_roi: tuple[int, int, int, int] | None
    letterbox_metadata: dict[str, Any]
    model_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    sample_id: str | None = None
    original_mode: str | None = None
    # 图像数组（不直接 JSON 序列化）
    original_binary_mask: np.ndarray | None = None
    original_probability_mask: np.ndarray | None = None
    model_probability_mask: np.ndarray | None = None
    model_binary_mask: np.ndarray | None = None
    tongue_roi_rgb: np.ndarray | None = None
    tongue_roi_mask: np.ndarray | None = None
    masked_tongue_rgb: np.ndarray | None = None
    # D4 预留
    bbox_width_ratio: float | None = None
    bbox_height_ratio: float | None = None
    bbox_area_ratio: float | None = None
    touches_image_border: bool | None = None
    mean_foreground_probability: float | None = None
    max_probability: float | None = None
    roi_size: tuple[int, int] | None = None
    restoration_strategy: str = RESTORATION_STRATEGY
    bbox_convention: str = BBOX_CONVENTION
    roi_margin_ratio: float = DEFAULT_ROI_MARGIN_RATIO

    def metadata_dict(self) -> dict[str, Any]:
        """可序列化 metadata（不含大数组）。"""
        return {
            "status": self.status,
            "sample_id": self.sample_id,
            "original_mode": self.original_mode,
            "original_size": [self.original_width, self.original_height],
            "input_size": [self.input_width, self.input_height],
            "threshold": self.threshold,
            "restoration_strategy": self.restoration_strategy,
            "bbox_convention": self.bbox_convention,
            "roi_margin_ratio": self.roi_margin_ratio,
            "mask_foreground_pixels": self.mask_foreground_pixels,
            "mask_foreground_ratio": self.mask_foreground_ratio,
            "component_count": self.component_count,
            "largest_component_ratio": self.largest_component_ratio,
            "bbox_tight": list(self.bbox_tight) if self.bbox_tight else None,
            "bbox_roi": list(self.bbox_roi) if self.bbox_roi else None,
            "roi_size": list(self.roi_size) if self.roi_size else None,
            "bbox_width_ratio": self.bbox_width_ratio,
            "bbox_height_ratio": self.bbox_height_ratio,
            "bbox_area_ratio": self.bbox_area_ratio,
            "touches_image_border": self.touches_image_border,
            "mean_foreground_probability": self.mean_foreground_probability,
            "max_probability": self.max_probability,
            "letterbox_metadata": self.letterbox_metadata,
            "model_metadata": self.model_metadata,
            "warnings": list(self.warnings),
            "arrays": {
                "original_binary_mask": _array_info(self.original_binary_mask),
                "original_probability_mask": _array_info(self.original_probability_mask),
                "tongue_roi_rgb": _array_info(self.tongue_roi_rgb),
                "tongue_roi_mask": _array_info(self.tongue_roi_mask),
            },
        }


def _array_info(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def load_rgb_image(
    source: str | Path | Image.Image | np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    统一加载为 RGB uint8。
    路径 / PIL：exif_transpose → convert RGB。
    ndarray：要求已是 HxWx3 或可广播灰度。
    返回 (rgb, original_mode)。
    """
    if isinstance(source, np.ndarray):
        array = np.asarray(source)
        if array.ndim == 2:
            rgb = np.stack([array, array, array], axis=-1)
            return rgb.astype(np.uint8), "ndarray_gray"
        if array.ndim == 3 and array.shape[2] == 3:
            return array.astype(np.uint8), "ndarray_rgb"
        if array.ndim == 3 and array.shape[2] == 4:
            # RGBA → RGB（丢弃 alpha）
            return array[:, :, :3].astype(np.uint8), "ndarray_rgba"
        raise ValueError(f"unsupported ndarray shape for RGB: {array.shape}")

    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, Image.Image):
        image = source
    else:
        raise TypeError(f"unsupported image type: {type(source)}")

    original_mode = str(image.mode)
    # 必须先 EXIF 旋转，再取尺寸，否则手机竖拍坐标错位
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"failed to convert to RGB, got shape={rgb.shape}")
    return rgb, original_mode


def load_frozen_segmentation_model(
    checkpoint_path: str | Path,
    data_config: SegmentationConfig | str | Path,
    train_config: TrainConfig | str | Path,
    device: str = "auto",
):
    """
    严格加载 frozen D3-C checkpoint。
    校验 architecture / encoder / classes / config_hash；strict=True。
    """
    import torch

    if isinstance(data_config, (str, Path)):
        data_config = SegmentationConfig(data_config)
    if isinstance(train_config, (str, Path)):
        train_config = TrainConfig(train_config)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"corrupted or unreadable checkpoint: {checkpoint_path}: {exc}") from exc

    integrity_errors = verify_checkpoint_integrity(checkpoint, train_config)
    # 额外结构字段
    model_cfg = (checkpoint.get("config") or {}).get("model") or {}
    expected_encoder = str(train_config.model.get("encoder", "resnet34"))
    expected_classes = int(train_config.model.get("classes", 1))
    ckpt_encoder = str(model_cfg.get("encoder", expected_encoder))
    ckpt_classes = int(model_cfg.get("classes", expected_classes))
    if ckpt_encoder != expected_encoder:
        integrity_errors.append(
            f"encoder mismatch: ckpt={ckpt_encoder} config={expected_encoder}"
        )
    if ckpt_classes != expected_classes:
        integrity_errors.append(
            f"classes mismatch: ckpt={ckpt_classes} config={expected_classes}"
        )
    if "model_state_dict" not in checkpoint:
        integrity_errors.append("missing model_state_dict")
    if integrity_errors:
        raise ValueError(
            "checkpoint/config mismatch (fail-fast): " + "; ".join(integrity_errors)
        )

    resolved_device = resolve_device(device if device != "auto" else train_config.device)
    # 推理加载：不重新下载 pretrained；先用 encoder_weights=None 建结构再 load
    build_cfg = dict(train_config.model)
    build_cfg["encoder_weights"] = None
    model = build_segmentation_model(build_cfg)
    # 严格加载，禁止 silent skip
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()

    param_stats = count_parameters(model)
    model_metadata = {
        "checkpoint_path": str(checkpoint_path),
        "config_hash": checkpoint.get("config_hash"),
        "train_config_hash": train_config.config_hash,
        "architecture": str(train_config.model.get("architecture")),
        "encoder": expected_encoder,
        "classes": expected_classes,
        "seed": checkpoint.get("seed"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_dice": checkpoint.get("best_val_dice"),
        "device": str(resolved_device),
        **param_stats,
    }
    return model, data_config, train_config, resolved_device, model_metadata, checkpoint


class TongueSegmentationInference:
    """原图 → mask / bbox / ROI 推理服务。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_config: str | Path | SegmentationConfig,
        train_config: str | Path | TrainConfig,
        device: str = "auto",
        *,
        roi_margin_ratio: float = DEFAULT_ROI_MARGIN_RATIO,
        keep_largest_component: bool = DEFAULT_KEEP_LARGEST_COMPONENT,
        near_full_ratio: float = DEFAULT_NEAR_FULL_RATIO,
        return_model_space: bool = True,
        return_probability: bool = True,
        return_masked_roi: bool = True,
        use_amp: bool | None = None,
    ):
        (
            self.model,
            self.data_config,
            self.train_config,
            self.device,
            self.model_metadata,
            self.checkpoint,
        ) = load_frozen_segmentation_model(
            checkpoint_path, data_config, train_config, device=device
        )
        self.threshold = float(self.train_config.mask_threshold)
        self.roi_margin_ratio = float(roi_margin_ratio)
        self.keep_largest_component = bool(keep_largest_component)
        self.near_full_ratio = float(near_full_ratio)
        self.return_model_space = bool(return_model_space)
        self.return_probability = bool(return_probability)
        self.return_masked_roi = bool(return_masked_roi)
        if use_amp is None:
            use_amp = bool(self.train_config.training.get("amp", True))
        self.use_amp = bool(use_amp) and str(self.device).startswith("cuda")

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        sample_id: str | None = None,
    ) -> TongueSegmentationResult:
        import torch

        original_rgb, original_mode = load_rgb_image(image)
        original_height, original_width = original_rgb.shape[:2]

        letterboxed, letterbox_meta = letterbox_image(
            original_rgb,
            input_height=self.data_config.input_height,
            input_width=self.data_config.input_width,
            image_interpolation=str(
                self.data_config.resize.get("image_interpolation", "bilinear")
            ),
            pad_value=int(self.data_config.resize.get("pad_value_image", 0)),
        )
        # 仅模型输入做 ImageNet normalize；ROI 仍用 original_rgb
        image_tensor = normalize_image(letterboxed, self.data_config)
        batch = torch.from_numpy(np.ascontiguousarray(image_tensor)).unsqueeze(0)
        batch = batch.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits = self.model(batch)
            else:
                logits = self.model(batch)
            # 概率 / 几何统一 float32，避免 fp16 边界问题
            probability = torch.sigmoid(logits.float())
            model_probability = (
                probability[0, 0].detach().cpu().numpy().astype(np.float32)
            )

        model_binary = (model_probability >= self.threshold).astype(np.uint8)
        original_probability, original_binary = restore_probability_then_threshold(
            model_probability, letterbox_meta, threshold=self.threshold
        )

        warnings: list[str] = []
        component_count = 0
        largest_component_ratio = 0.0
        if self.keep_largest_component:
            original_binary, component_stats = keep_largest_connected_component(
                original_binary
            )
            component_count = int(component_stats["component_count_before"])
            largest_component_ratio = float(component_stats["largest_component_ratio"])
        else:
            # 仍统计连通域数量供 metadata
            _filtered, component_stats = keep_largest_connected_component(original_binary)
            component_count = int(component_stats["component_count_before"])
            largest_component_ratio = float(component_stats["largest_component_ratio"])
            del _filtered

        foreground_pixels = int(original_binary.sum())
        foreground_ratio = float(foreground_pixels / original_binary.size)
        if foreground_ratio > self.near_full_ratio:
            warnings.append("near_full_prediction")

        max_probability = float(original_probability.max()) if original_probability.size else 0.0
        if foreground_pixels > 0:
            mean_foreground_probability = float(
                original_probability[original_binary > 0].mean()
            )
        else:
            mean_foreground_probability = None

        if foreground_pixels == 0:
            return TongueSegmentationResult(
                status="no_tongue_detected",
                sample_id=sample_id,
                original_mode=original_mode,
                original_width=original_width,
                original_height=original_height,
                input_width=letterbox_meta.input_width,
                input_height=letterbox_meta.input_height,
                threshold=self.threshold,
                mask_foreground_pixels=0,
                mask_foreground_ratio=0.0,
                component_count=component_count,
                largest_component_ratio=largest_component_ratio,
                bbox_tight=None,
                bbox_roi=None,
                letterbox_metadata=letterbox_meta.to_dict(),
                model_metadata=dict(self.model_metadata),
                warnings=warnings,
                original_binary_mask=original_binary,
                original_probability_mask=original_probability
                if self.return_probability
                else None,
                model_probability_mask=model_probability
                if self.return_model_space
                else None,
                model_binary_mask=model_binary if self.return_model_space else None,
                mean_foreground_probability=mean_foreground_probability,
                max_probability=max_probability,
                roi_margin_ratio=self.roi_margin_ratio,
            )

        bbox_tight = mask_to_bbox_xyxy_exclusive(original_binary)
        assert bbox_tight is not None
        bbox_roi = expand_bbox(
            bbox_tight,
            image_width=original_width,
            image_height=original_height,
            margin_ratio=self.roi_margin_ratio,
        )
        tongue_roi_rgb = crop_roi(original_rgb, bbox_roi)
        tongue_roi_mask = crop_roi(original_binary, bbox_roi)
        if tongue_roi_rgb.shape[:2] != tongue_roi_mask.shape[:2]:
            raise RuntimeError("ROI image/mask shape mismatch")
        masked_roi = (
            apply_mask_to_rgb(tongue_roi_rgb, tongue_roi_mask)
            if self.return_masked_roi
            else None
        )
        ratios = bbox_ratios(
            bbox_tight, image_width=original_width, image_height=original_height
        )

        return TongueSegmentationResult(
            status="success",
            sample_id=sample_id,
            original_mode=original_mode,
            original_width=original_width,
            original_height=original_height,
            input_width=letterbox_meta.input_width,
            input_height=letterbox_meta.input_height,
            threshold=self.threshold,
            mask_foreground_pixels=foreground_pixels,
            mask_foreground_ratio=foreground_ratio,
            component_count=component_count,
            largest_component_ratio=largest_component_ratio,
            bbox_tight=bbox_tight,
            bbox_roi=bbox_roi,
            letterbox_metadata=letterbox_meta.to_dict(),
            model_metadata=dict(self.model_metadata),
            warnings=warnings,
            original_binary_mask=original_binary,
            original_probability_mask=original_probability
            if self.return_probability
            else None,
            model_probability_mask=model_probability if self.return_model_space else None,
            model_binary_mask=model_binary if self.return_model_space else None,
            tongue_roi_rgb=tongue_roi_rgb,
            tongue_roi_mask=tongue_roi_mask,
            masked_tongue_rgb=masked_roi,
            mean_foreground_probability=mean_foreground_probability,
            max_probability=max_probability,
            roi_size=(int(tongue_roi_rgb.shape[1]), int(tongue_roi_rgb.shape[0])),
            roi_margin_ratio=self.roi_margin_ratio,
            **ratios,
        )


def save_binary_mask_png(mask: np.ndarray, path: str | Path) -> None:
    """内部 0/1 → PNG 0/255。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(path)


def load_binary_mask_png(path: str | Path) -> np.ndarray:
    """PNG >0 → 内部 0/1。"""
    array = np.asarray(Image.open(path))
    return (array > 0).astype(np.uint8)


def save_probability_png(probability: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip(np.asarray(probability) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)


def make_overlay(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """调试用半透明 overlay；不得作为 phenotype 输入。"""
    base = rgb.astype(np.float32)
    overlay = base.copy()
    mask = np.asarray(binary_mask) > 0
    for channel_index, channel_value in enumerate(color):
        overlay[..., channel_index] = np.where(
            mask,
            base[..., channel_index] * (1 - alpha) + channel_value * alpha,
            base[..., channel_index],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_inference_outputs(
    result: TongueSegmentationResult,
    output_dir: str | Path,
    *,
    original_rgb: np.ndarray | None = None,
    save_overlay: bool = True,
    save_probability: bool = True,
) -> dict[str, str]:
    """写入 sample 目录：metadata + mask/roi PNG。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(result.metadata_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path)

    if result.original_binary_mask is not None:
        mask_path = output_dir / "mask.png"
        save_binary_mask_png(result.original_binary_mask, mask_path)
        paths["mask"] = str(mask_path)

    if (
        save_probability
        and result.original_probability_mask is not None
    ):
        prob_path = output_dir / "probability.png"
        save_probability_png(result.original_probability_mask, prob_path)
        paths["probability"] = str(prob_path)

    if result.tongue_roi_rgb is not None:
        roi_path = output_dir / "roi.png"
        Image.fromarray(result.tongue_roi_rgb).save(roi_path)
        paths["roi"] = str(roi_path)
    if result.tongue_roi_mask is not None:
        roi_mask_path = output_dir / "roi_mask.png"
        save_binary_mask_png(result.tongue_roi_mask, roi_mask_path)
        paths["roi_mask"] = str(roi_mask_path)
    if result.masked_tongue_rgb is not None:
        masked_path = output_dir / "roi_masked.png"
        Image.fromarray(result.masked_tongue_rgb).save(masked_path)
        paths["roi_masked"] = str(masked_path)

    if (
        save_overlay
        and original_rgb is not None
        and result.original_binary_mask is not None
    ):
        overlay = make_overlay(original_rgb, result.original_binary_mask)
        overlay_path = output_dir / "overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        paths["overlay"] = str(overlay_path)

    return paths


def format_console_summary(result: TongueSegmentationResult) -> str:
    """CLI 友好摘要，不打印巨大数组。"""
    lines = [
        f"status: {result.status}",
        f"original_size: {result.original_width}×{result.original_height}",
        f"model_input: {result.input_width}×{result.input_height}",
        f"foreground_ratio: {result.mask_foreground_ratio:.6f}",
        f"bbox_tight: {list(result.bbox_tight) if result.bbox_tight else None}",
        f"bbox_roi: {list(result.bbox_roi) if result.bbox_roi else None}",
        f"roi_size: {list(result.roi_size) if result.roi_size else None}",
        f"component_count: {result.component_count}",
        f"largest_component_ratio: {result.largest_component_ratio:.6f}",
        f"warnings: {result.warnings}",
    ]
    return "\n".join(lines)
