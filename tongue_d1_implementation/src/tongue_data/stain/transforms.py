"""Stain ROI 预处理：masked RGB + letterbox；禁止色相增强。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import StainDataConfig


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_tongue_mask(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    fill_value: int = 0,
) -> np.ndarray:
    """mask 外填充固定值；不修改输入数组。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8).copy()
    mask = np.asarray(roi_mask) > 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"roi/mask shape mismatch: {rgb.shape[:2]} vs {mask.shape}")
    rgb[~mask] = int(fill_value)
    return rgb


def letterbox_rgb(
    image: np.ndarray,
    size: int,
    fill_value: int = 0,
) -> np.ndarray:
    """保持长宽比 letterbox 到 size×size。"""
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = float(size) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.full((size, size, 3), int(fill_value), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def normalize_imagenet(image_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 CHW ImageNet normalize。"""
    image = image_uint8.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def geometric_augment(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    horizontal_flip: bool = True,
    rotation_degrees: float = 10.0,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    translate_frac: float = 0.05,
) -> np.ndarray:
    """仅几何增强；禁止 hue/saturation/color jitter。"""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if horizontal_flip and rng.random() < 0.5:
        pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rotation_degrees and rotation_degrees > 0:
        degrees = float(rng.uniform(-rotation_degrees, rotation_degrees))
        pil = pil.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    width, height = pil.size
    scale = float(rng.uniform(scale_min, scale_max))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil = pil.resize((new_width, new_height), Image.Resampling.BILINEAR)
    # pad/crop 回原尺寸
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    offset_x = (width - new_width) // 2
    offset_y = (height - new_height) // 2
    if translate_frac > 0:
        offset_x += int(rng.uniform(-translate_frac, translate_frac) * width)
        offset_y += int(rng.uniform(-translate_frac, translate_frac) * height)
    canvas.paste(pil, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_masked_roi(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    config: StainDataConfig,
    *,
    split: str,
    rng: np.random.Generator | None = None,
    augment_cfg: dict | None = None,
    return_pre_norm_rgb: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """masked ROI → letterbox → optional geom aug → normalize。"""
    masked = apply_tongue_mask(roi_rgb, roi_mask, fill_value=config.mask_fill)
    letterboxed = letterbox_rgb(masked, config.input_size, fill_value=config.mask_fill)
    if split == "train" and augment_cfg is not None:
        # 硬禁止颜色增强
        if augment_cfg.get("color_jitter") or augment_cfg.get("brightness_contrast"):
            raise ValueError("color augmentation is forbidden for stain detection")
        if rng is None:
            rng = np.random.default_rng(0)
        letterboxed = geometric_augment(
            letterboxed,
            rng,
            horizontal_flip=bool(augment_cfg.get("horizontal_flip", True)),
            rotation_degrees=float(augment_cfg.get("rotation_degrees", 10)),
            scale_min=float(augment_cfg.get("scale_min", 0.9)),
            scale_max=float(augment_cfg.get("scale_max", 1.1)),
            translate_frac=float(augment_cfg.get("translate_frac", 0.05)),
        )
        # 再 letterbox 回固定尺寸（几何后可能轻微越界）
        letterboxed = letterbox_rgb(letterboxed, config.input_size, fill_value=config.mask_fill)
    tensor = normalize_imagenet(letterboxed)
    if return_pre_norm_rgb:
        return tensor, letterboxed
    return tensor
