"""Mask 读取与归一化：统一 background=0 / tongue=1。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_image_rgb(path: str) -> np.ndarray:
    """读取 RGB uint8 图像，shape=[H,W,3]。"""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_mask_raw(path: str) -> np.ndarray:
    """读取原始 mask（可能是 bool / 0-1 / 0-255 / 多通道）。"""
    mask = np.asarray(Image.open(path))
    return mask


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    统一前景规则：mask > 0 → 1，否则 0。
    禁止依赖 mask == 255。
    返回 float32 [H, W]。
    """
    if mask.ndim == 3:
        # 多通道：任一通道 > 0 即前景
        foreground = np.any(mask > 0, axis=-1)
    else:
        foreground = mask > 0
    return foreground.astype(np.float32)


def foreground_ratio(mask_binary: np.ndarray) -> float:
    """计算前景占比；mask 应为 {0,1}。"""
    if mask_binary.size == 0:
        return 0.0
    return float(mask_binary.mean())


def unique_pixel_values(mask: np.ndarray, limit: int = 32) -> list:
    """审计用：返回有限个唯一像素值。"""
    flat = mask.reshape(-1)
    values = np.unique(flat)
    if len(values) > limit:
        return values[:limit].tolist() + ["..."]
    # bool → int 便于 JSON
    return [int(value) if not isinstance(value, (np.bool_, bool)) else int(value) for value in values]
