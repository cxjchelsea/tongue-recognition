"""Image/Mask 同步几何变换与预处理（letterbox + 保守增广）。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import SegmentationConfig
from .geometry import LetterboxMetadata, letterbox_image, letterbox_mask

# 向后兼容：D3-A 代码使用 GeometryMeta 名称
GeometryMeta = LetterboxMetadata


def _pil_resample(name: str) -> Image.Resampling:
    key = str(name).lower()
    if key in {"nearest"}:
        return Image.Resampling.NEAREST
    if key in {"bilinear", "linear"}:
        return Image.Resampling.BILINEAR
    if key in {"bicubic"}:
        return Image.Resampling.BICUBIC
    raise ValueError(f"unsupported interpolation: {name}")


def letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_height: int,
    target_width: int,
    image_interpolation: str = "bilinear",
    mask_interpolation: str = "nearest",
    pad_value_image: int = 0,
    pad_value_mask: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, GeometryMeta]:
    """保持长宽比缩放 + 居中 padding；image/mask 同步几何（复用 D3-E geometry）。"""
    if mask_interpolation.lower() != "nearest":
        raise ValueError("mask_interpolation must be nearest")

    original_height, original_width = image.shape[:2]
    if mask.shape[:2] != (original_height, original_width):
        raise ValueError(
            f"image/mask shape mismatch before letterbox: "
            f"image={image.shape[:2]} mask={mask.shape[:2]}"
        )

    canvas_image, metadata = letterbox_image(
        image,
        input_height=target_height,
        input_width=target_width,
        image_interpolation=image_interpolation,
        pad_value=pad_value_image,
    )
    canvas_mask = letterbox_mask(mask, metadata, pad_value=pad_value_mask)
    return canvas_image, canvas_mask, metadata


def _rotate_pair(image: np.ndarray, mask: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    image_rot = np.asarray(image_pil.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)))
    mask_rot = np.asarray(mask_pil.rotate(degrees, resample=Image.Resampling.NEAREST, fillcolor=0))
    return image_rot.astype(np.uint8), (mask_rot > 0).astype(np.float32)


def _scale_pair(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    image_scaled = np.asarray(
        image_pil.resize((new_width, new_height), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    mask_scaled = np.asarray(
        mask_pil.resize((new_width, new_height), Image.Resampling.NEAREST), dtype=np.uint8
    )
    mask_scaled = (mask_scaled > 0).astype(np.float32)

    # 裁回或 pad 回原尺寸（中心对齐）
    canvas_image = np.zeros_like(image)
    canvas_mask = np.zeros_like(mask, dtype=np.float32)
    if new_height >= height and new_width >= width:
        top = (new_height - height) // 2
        left = (new_width - width) // 2
        canvas_image = image_scaled[top : top + height, left : left + width]
        canvas_mask = mask_scaled[top : top + height, left : left + width]
    else:
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        canvas_image[top : top + new_height, left : left + new_width] = image_scaled
        canvas_mask[top : top + new_height, left : left + new_width] = mask_scaled
    return canvas_image, canvas_mask


def apply_train_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    config: SegmentationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """保守 train 增广；几何变换同步作用于 image/mask。"""
    train_cfg = config.augmentation.get("train", {})

    flip_cfg = train_cfg.get("horizontal_flip", {})
    if flip_cfg.get("enabled", False) and rng.random() < float(flip_cfg.get("probability", 0.5)):
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    rot_cfg = train_cfg.get("rotation", {})
    if rot_cfg.get("enabled", False):
        max_degrees = float(rot_cfg.get("degrees", 10))
        degrees = float(rng.uniform(-max_degrees, max_degrees))
        image, mask = _rotate_pair(image, mask, degrees)

    scale_cfg = train_cfg.get("scale", {})
    if scale_cfg.get("enabled", False):
        scale = float(rng.uniform(float(scale_cfg.get("min", 0.9)), float(scale_cfg.get("max", 1.1))))
        image, mask = _scale_pair(image, mask, scale)

    bc_cfg = train_cfg.get("brightness_contrast", {})
    if bc_cfg.get("enabled", False) and rng.random() < float(bc_cfg.get("probability", 0.3)):
        brightness = 1.0 + float(rng.uniform(-float(bc_cfg.get("brightness_limit", 0.1)), float(bc_cfg.get("brightness_limit", 0.1))))
        contrast = 1.0 + float(rng.uniform(-float(bc_cfg.get("contrast_limit", 0.1)), float(bc_cfg.get("contrast_limit", 0.1))))
        image_float = image.astype(np.float32)
        mean = image_float.mean(axis=(0, 1), keepdims=True)
        image_float = (image_float - mean) * contrast + mean
        image_float = image_float * brightness
        image = np.clip(image_float, 0, 255).astype(np.uint8)

    return image, mask


def normalize_image(image: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """ImageNet normalize → float32 CHW。"""
    mean = np.asarray(config.normalization.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(config.normalization.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    image_float = image.astype(np.float32) / 255.0
    image_float = (image_float - mean) / std
    return np.transpose(image_float, (2, 0, 1)).astype(np.float32)


def preprocess_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: SegmentationConfig,
    split: str,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, GeometryMeta]:
    """完整预处理：可选 train 增广 → letterbox → normalize。"""
    split_name = str(split)
    if split_name == "train":
        if rng is None:
            rng = np.random.default_rng(config.seed)
        image, mask = apply_train_augmentation(image, mask, config, rng)
    else:
        # val/test：禁止随机增强
        val_enabled = bool(config.augmentation.get("val", {}).get("enabled", False))
        test_enabled = bool(config.augmentation.get("test", {}).get("enabled", False))
        if split_name == "val" and val_enabled:
            raise ValueError("val augmentation must be disabled in D3-A")
        if split_name == "test" and test_enabled:
            raise ValueError("test augmentation must be disabled in D3-A")

    image_lb, mask_lb, meta = letterbox_pair(
        image,
        mask,
        target_height=config.input_height,
        target_width=config.input_width,
        image_interpolation=str(config.resize.get("image_interpolation", "bilinear")),
        mask_interpolation=str(config.resize.get("mask_interpolation", "nearest")),
        pad_value_image=int(config.resize.get("pad_value_image", 0)),
        pad_value_mask=float(config.resize.get("pad_value_mask", 0)),
    )
    # 确保 resize 后仍为 binary
    unique_values = set(np.unique(mask_lb).tolist())
    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(f"mask not binary after resize: {unique_values}")

    image_tensor = normalize_image(image_lb, config)
    mask_tensor = mask_lb[None, ...].astype(np.float32)  # [1,H,W]
    return image_tensor, mask_tensor, meta
