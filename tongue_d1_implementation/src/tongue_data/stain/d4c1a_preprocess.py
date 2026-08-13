"""D4-C.1-A：只读 counterfactual 输入表示（不改正式 runtime contract）。"""
from __future__ import annotations

import numpy as np

from .config import StainDataConfig
from .transforms import (
    apply_tongue_mask,
    letterbox_rgb,
    normalize_imagenet,
    preprocess_masked_roi,
)


def mask_binary(roi_mask: np.ndarray) -> np.ndarray:
    """统一 mask 语义：>0 为前景。"""
    return np.asarray(roi_mask) > 0


def border_mean_color(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """舌头边界一带的均值色（用于 mean-fill counterfactual）。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = mask_binary(roi_mask)
    if not mask.any():
        return np.array([128, 128, 128], dtype=np.uint8)
    # 简单膨胀边界环
    from scipy import ndimage

    eroded = ndimage.binary_erosion(mask, iterations=2)
    border = mask & (~eroded)
    if not border.any():
        border = mask
    means = rgb[border].mean(axis=0)
    return np.clip(np.round(means), 0, 255).astype(np.uint8)


def apply_fill(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    生成不同 fill 的 ROI（未 letterbox）。
    mode: black | gray | mean_fill | bbox | context
    """
    rgb = np.asarray(roi_rgb, dtype=np.uint8).copy()
    mask = mask_binary(roi_mask)
    if mode == "black":
        return apply_tongue_mask(rgb, mask.astype(np.uint8) * 255, fill_value=0)
    if mode == "gray":
        return apply_tongue_mask(rgb, mask.astype(np.uint8) * 255, fill_value=127)
    if mode == "mean_fill":
        color = border_mean_color(rgb, mask)
        out = rgb.copy()
        out[~mask] = color
        return out
    if mode == "bbox":
        # 不 mask：保留 ROI 内背景
        return rgb
    if mode == "context":
        # 与 bbox 相同输入；context 扩张在外层裁剪时处理，此处原样返回
        return rgb
    raise ValueError(f"unknown fill mode: {mode}")


def letterbox_meta(height: int, width: int, size: int) -> dict[str, float]:
    """letterbox 几何元数据。"""
    scale = float(size) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    content_area = float(new_width * new_height)
    canvas_area = float(size * size)
    return {
        "scale": scale,
        "new_width": float(new_width),
        "new_height": float(new_height),
        "pad_top": float(top),
        "pad_left": float(left),
        "padding_ratio": 1.0 - content_area / canvas_area,
        "content_area_ratio": content_area / canvas_area,
    }


def preprocess_counterfactual(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    config: StainDataConfig,
    *,
    mode: str = "black",
    return_pre_norm_rgb: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """frozen-model counterfactual preprocessing（split=val，无 augment）。"""
    if mode == "black":
        # 正式 contract 路径
        return preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            config,
            split="val",
            rng=None,
            augment_cfg=None,
            return_pre_norm_rgb=return_pre_norm_rgb,
        )
    filled = apply_fill(roi_rgb, roi_mask, mode=mode)
    # bbox/context：fill_value 用于 letterbox pad；用 0 保持 pad 语义可比
    letterboxed = letterbox_rgb(filled, config.input_size, fill_value=0)
    tensor = normalize_imagenet(letterboxed)
    if return_pre_norm_rgb:
        return tensor, letterboxed
    return tensor


def compute_fill_padding_ratios(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    config: StainDataConfig,
) -> dict[str, float]:
    """black-fill letterbox 输入上的 fill / padding / black 比例。"""
    _tensor, letterboxed = preprocess_counterfactual(
        roi_rgb, roi_mask, config, mode="black", return_pre_norm_rgb=True
    )
    del _tensor
    height, width = np.asarray(roi_rgb).shape[:2]
    meta = letterbox_meta(height, width, config.input_size)
    black = (letterboxed[..., 0] == 0) & (letterboxed[..., 1] == 0) & (
        letterboxed[..., 2] == 0
    )
    # letterbox 后的前景：缩放 mask
    mask = mask_binary(roi_mask).astype(np.uint8) * 255
    mask_lb = letterbox_rgb(
        np.stack([mask, mask, mask], axis=-1),
        config.input_size,
        fill_value=0,
    )[..., 0]
    fore = mask_lb > 0
    top = int(meta["pad_top"])
    left = int(meta["pad_left"])
    new_h = int(meta["new_height"])
    new_w = int(meta["new_width"])
    padding = np.ones((config.input_size, config.input_size), dtype=bool)
    padding[top : top + new_h, left : left + new_w] = False
    content = ~padding
    mask_fill_in_content = content & (~fore)
    return {
        "black_pixel_ratio": float(black.mean()),
        "padding_ratio": float(meta["padding_ratio"]),
        "mask_fill_ratio": float(mask_fill_in_content.mean())
        if content.any()
        else float("nan"),
        "foreground_in_canvas_ratio": float(fore.mean()),
        "black_in_padding_ratio": float(black[padding].mean()) if padding.any() else 0.0,
        "black_in_fill_ratio": float(black[mask_fill_in_content].mean())
        if mask_fill_in_content.any()
        else 0.0,
    }


def assert_train_runtime_tensor_equiv(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    config: StainDataConfig,
    *,
    atol: float = 1e-6,
) -> bool:
    """训练 val 路径与 runtime black 路径张量一致。"""
    train_like = preprocess_masked_roi(
        roi_rgb, roi_mask, config, split="val", augment_cfg=None
    )
    runtime_like = preprocess_counterfactual(
        roi_rgb, roi_mask, config, mode="black"
    )
    return bool(np.allclose(train_like, runtime_like, atol=atol))
