"""D4-B 信号特征提取：blur / exposure / illumination / border / resolution。

所有运算在副本上进行，禁止原地修改 original RGB。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .features import InputGuardFeatures, features_from_segmentation_result

# 固定分析尺度（blur）；配置可覆盖
DEFAULT_BLUR_LONG_SIDE = 256
DEFAULT_SHADOW_CLIP_LUMA = 8.0
DEFAULT_HIGHLIGHT_CLIP_LUMA = 247.0
DEFAULT_DARK_LUMA = 40.0
DEFAULT_BRIGHT_LUMA = 220.0
DEFAULT_ILLUM_GRID = 3
DEFAULT_MIN_GRID_TONGUE_PIXELS = 16


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance；输入 uint8/float RGB，输出 float32。"""
    array = np.asarray(rgb, dtype=np.float32)
    return (
        0.2126 * array[..., 0]
        + 0.7152 * array[..., 1]
        + 0.0722 * array[..., 2]
    )


def resize_long_side_gray(rgb: np.ndarray, long_side: int) -> np.ndarray:
    """保持长宽比缩放后转灰度（副本）。"""
    if rgb.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    width, height = image.size
    scale = float(long_side) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    gray = np.asarray(resized.convert("L"), dtype=np.float32)
    return gray


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Laplacian 方差（越高越清晰）。"""
    array = np.asarray(gray, dtype=np.float32)
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0
    # 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # 简易卷积（valid）
    padded = np.pad(array, 1, mode="edge")
    accum = (
        kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 1] * padded[2:, 1:-1]
    )
    return float(accum.var())


def tenengrad_energy(gray: np.ndarray) -> float:
    """Sobel 梯度能量均值（越高越清晰）。"""
    array = np.asarray(gray, dtype=np.float32)
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    padded = np.pad(array, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    grad_x = (
        sobel_x[0, 0] * padded[:-2, :-2]
        + sobel_x[0, 1] * padded[:-2, 1:-1]
        + sobel_x[0, 2] * padded[:-2, 2:]
        + sobel_x[1, 0] * padded[1:-1, :-2]
        + sobel_x[1, 1] * center
        + sobel_x[1, 2] * padded[1:-1, 2:]
        + sobel_x[2, 0] * padded[2:, :-2]
        + sobel_x[2, 1] * padded[2:, 1:-1]
        + sobel_x[2, 2] * padded[2:, 2:]
    )
    grad_y = (
        sobel_y[0, 0] * padded[:-2, :-2]
        + sobel_y[0, 1] * padded[:-2, 1:-1]
        + sobel_y[0, 2] * padded[:-2, 2:]
        + sobel_y[1, 0] * padded[1:-1, :-2]
        + sobel_y[1, 1] * center
        + sobel_y[1, 2] * padded[1:-1, 2:]
        + sobel_y[2, 0] * padded[2:, :-2]
        + sobel_y[2, 1] * padded[2:, 1:-1]
        + sobel_y[2, 2] * padded[2:, 2:]
    )
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return float(magnitude.mean())


def compute_border_touch_stats(binary_mask: np.ndarray) -> dict[str, Any]:
    """基于 original-resolution mask 的边界接触统计。"""
    mask = np.asarray(binary_mask) > 0
    height, width = mask.shape
    if height == 0 or width == 0 or not np.any(mask):
        return {
            "left_touch_ratio": None,
            "right_touch_ratio": None,
            "top_touch_ratio": None,
            "bottom_touch_ratio": None,
            "border_touch_ratio": None,
            "mask_border_touch_pixels": None,
            "touches_left": False,
            "touches_right": False,
            "touches_top": False,
            "touches_bottom": False,
        }
    left_line = mask[:, 0]
    right_line = mask[:, -1]
    top_line = mask[0, :]
    bottom_line = mask[-1, :]
    left_count = int(left_line.sum())
    right_count = int(right_line.sum())
    top_count = int(top_line.sum())
    bottom_count = int(bottom_line.sum())
    border_pixels = left_count + right_count + top_count + bottom_count
    perimeter_capacity = 2 * (height + width)
    return {
        "left_touch_ratio": float(left_count / height),
        "right_touch_ratio": float(right_count / height),
        "top_touch_ratio": float(top_count / width),
        "bottom_touch_ratio": float(bottom_count / width),
        "border_touch_ratio": float(border_pixels / max(perimeter_capacity, 1)),
        "mask_border_touch_pixels": border_pixels,
        "touches_left": left_count > 0,
        "touches_right": right_count > 0,
        "touches_top": top_count > 0,
        "touches_bottom": bottom_count > 0,
    }


def compute_focus_features(
    image_rgb: np.ndarray,
    roi_rgb: np.ndarray | None,
    *,
    long_side: int = DEFAULT_BLUR_LONG_SIDE,
) -> dict[str, float | None]:
    image_gray = resize_long_side_gray(image_rgb, long_side)
    image_blur = variance_of_laplacian(image_gray)
    image_grad = tenengrad_energy(image_gray)
    if roi_rgb is None or np.asarray(roi_rgb).size == 0:
        return {
            "blur_score": float(image_blur),
            "roi_blur_score": None,
            "image_gradient_energy": float(image_grad),
            "roi_gradient_energy": None,
            "blur_analysis_long_side": float(long_side),
        }
    roi_gray = resize_long_side_gray(roi_rgb, long_side)
    return {
        "blur_score": float(image_blur),
        "roi_blur_score": float(variance_of_laplacian(roi_gray)),
        "image_gradient_energy": float(image_grad),
        "roi_gradient_energy": float(tenengrad_energy(roi_gray)),
        "blur_analysis_long_side": float(long_side),
    }


def compute_exposure_features(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray | None,
    *,
    shadow_clip_luma: float = DEFAULT_SHADOW_CLIP_LUMA,
    highlight_clip_luma: float = DEFAULT_HIGHLIGHT_CLIP_LUMA,
    dark_luma: float = DEFAULT_DARK_LUMA,
    bright_luma: float = DEFAULT_BRIGHT_LUMA,
) -> dict[str, float | None]:
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    if rgb.size == 0:
        return {key: None for key in (
            "mean_luminance", "roi_luminance_std", "roi_luminance_p01",
            "roi_luminance_p05", "roi_luminance_p50", "roi_luminance_p95",
            "roi_luminance_p99", "dark_pixel_ratio", "bright_pixel_ratio",
            "shadow_clip_ratio", "highlight_clip_ratio",
        )}
    luma = rgb_to_luminance(rgb)
    if roi_mask is not None:
        mask = np.asarray(roi_mask) > 0
        if mask.shape == luma.shape and np.any(mask):
            values = luma[mask]
        else:
            values = luma.reshape(-1)
    else:
        values = luma.reshape(-1)
    if values.size == 0:
        return {key: None for key in (
            "mean_luminance", "roi_luminance_std", "roi_luminance_p01",
            "roi_luminance_p05", "roi_luminance_p50", "roi_luminance_p95",
            "roi_luminance_p99", "dark_pixel_ratio", "bright_pixel_ratio",
            "shadow_clip_ratio", "highlight_clip_ratio",
        )}
    return {
        "mean_luminance": float(values.mean()),
        "roi_luminance_std": float(values.std()),
        "roi_luminance_p01": float(np.percentile(values, 1)),
        "roi_luminance_p05": float(np.percentile(values, 5)),
        "roi_luminance_p50": float(np.percentile(values, 50)),
        "roi_luminance_p95": float(np.percentile(values, 95)),
        "roi_luminance_p99": float(np.percentile(values, 99)),
        "dark_pixel_ratio": float((values <= dark_luma).mean()),
        "bright_pixel_ratio": float((values >= bright_luma).mean()),
        "shadow_clip_ratio": float((values <= shadow_clip_luma).mean()),
        "highlight_clip_ratio": float((values >= highlight_clip_luma).mean()),
    }


def compute_illumination_features(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray | None,
    *,
    grid_size: int = DEFAULT_ILLUM_GRID,
    min_cell_pixels: int = DEFAULT_MIN_GRID_TONGUE_PIXELS,
) -> dict[str, Any]:
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    if rgb.size == 0:
        return {
            "illumination_uniformity_score": None,
            "valid_grid_cells": None,
            "max_min_luminance_difference": None,
            "relative_luminance_range": None,
            "left_right_difference": None,
            "top_bottom_difference": None,
            "spatial_luminance_cv": None,
        }
    luma = rgb_to_luminance(rgb)
    if roi_mask is None:
        mask = np.ones(luma.shape, dtype=bool)
    else:
        mask = np.asarray(roi_mask) > 0
        if mask.shape != luma.shape:
            mask = np.ones(luma.shape, dtype=bool)

    height, width = luma.shape
    cell_means: list[float] = []
    grid_means = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for row_index in range(grid_size):
        row_start = int(round(height * row_index / grid_size))
        row_end = int(round(height * (row_index + 1) / grid_size))
        for col_index in range(grid_size):
            col_start = int(round(width * col_index / grid_size))
            col_end = int(round(width * (col_index + 1) / grid_size))
            cell_mask = mask[row_start:row_end, col_start:col_end]
            cell_luma = luma[row_start:row_end, col_start:col_end]
            tongue_pixels = int(cell_mask.sum())
            if tongue_pixels < min_cell_pixels:
                continue
            mean_value = float(cell_luma[cell_mask].mean())
            cell_means.append(mean_value)
            grid_means[row_index, col_index] = mean_value

    if len(cell_means) < 2:
        return {
            "illumination_uniformity_score": None,
            "valid_grid_cells": int(len(cell_means)),
            "max_min_luminance_difference": None,
            "relative_luminance_range": None,
            "left_right_difference": None,
            "top_bottom_difference": None,
            "spatial_luminance_cv": None,
        }

    values = np.asarray(cell_means, dtype=np.float32)
    max_min_diff = float(values.max() - values.min())
    mean_value = float(values.mean()) + 1e-6
    relative_range = float(max_min_diff / mean_value)
    spatial_cv = float(values.std() / mean_value)

    # left/right：用有效列均值
    col_means = []
    for col_index in range(grid_size):
        column = grid_means[:, col_index]
        valid = column[~np.isnan(column)]
        if valid.size:
            col_means.append(float(valid.mean()))
    left_right = None
    if len(col_means) >= 2:
        left_right = abs(col_means[0] - col_means[-1])

    row_means = []
    for row_index in range(grid_size):
        row = grid_means[row_index, :]
        valid = row[~np.isnan(row)]
        if valid.size:
            row_means.append(float(valid.mean()))
    top_bottom = None
    if len(row_means) >= 2:
        top_bottom = abs(row_means[0] - row_means[-1])

    return {
        # 分数：越大越不均匀（便于 higher-is-worse 阈值）
        "illumination_uniformity_score": relative_range,
        "valid_grid_cells": int(len(cell_means)),
        "max_min_luminance_difference": max_min_diff,
        "relative_luminance_range": relative_range,
        "left_right_difference": float(left_right) if left_right is not None else None,
        "top_bottom_difference": float(top_bottom) if top_bottom is not None else None,
        "spatial_luminance_cv": spatial_cv,
    }


def enrich_features_with_signals(
    original_rgb: np.ndarray,
    segmentation_result: Any,
    *,
    blur_long_side: int = DEFAULT_BLUR_LONG_SIDE,
    shadow_clip_luma: float = DEFAULT_SHADOW_CLIP_LUMA,
    highlight_clip_luma: float = DEFAULT_HIGHLIGHT_CLIP_LUMA,
) -> InputGuardFeatures:
    """D3-E adapter + D4-B 信号特征；不修改 original_rgb。"""
    # 确保使用副本做任何派生
    image = np.asarray(original_rgb)
    if image is original_rgb:
        image_view = image  # 只读视图；派生用新数组
    else:
        image_view = image

    features = features_from_segmentation_result(segmentation_result)
    status = getattr(segmentation_result, "status", None)

    # tight bbox 像素尺寸
    if features.bbox_tight is not None:
        x1, y1, x2, y2 = features.bbox_tight
        features.tight_bbox_width_px = int(x2 - x1)
        features.tight_bbox_height_px = int(y2 - y1)
        features.effective_short_side_px = int(
            min(features.tight_bbox_width_px, features.tight_bbox_height_px)
        )

    binary_mask = getattr(segmentation_result, "original_binary_mask", None)
    if binary_mask is not None:
        border = compute_border_touch_stats(binary_mask)
        for key, value in border.items():
            if hasattr(features, key):
                setattr(features, key, value)

    if status == "no_tongue_detected":
        features.available_feature_names = tuple(
            name for name, ok in features.availability_map().items() if ok
        )
        return features

    roi_rgb = getattr(segmentation_result, "tongue_roi_rgb", None)
    roi_mask = getattr(segmentation_result, "tongue_roi_mask", None)
    # 若无预裁 ROI，用 tight bbox 从原图裁（副本）
    if roi_rgb is None and features.bbox_tight is not None:
        x1, y1, x2, y2 = features.bbox_tight
        roi_rgb = np.ascontiguousarray(image_view[y1:y2, x1:x2].copy())
        if binary_mask is not None:
            roi_mask = np.ascontiguousarray(binary_mask[y1:y2, x1:x2].copy())

    focus = compute_focus_features(
        image_view, roi_rgb, long_side=blur_long_side
    )
    for key, value in focus.items():
        if hasattr(features, key):
            setattr(features, key, value)

    if roi_rgb is not None:
        exposure = compute_exposure_features(
            roi_rgb,
            roi_mask,
            shadow_clip_luma=shadow_clip_luma,
            highlight_clip_luma=highlight_clip_luma,
        )
        for key, value in exposure.items():
            if hasattr(features, key):
                setattr(features, key, value)
        illum = compute_illumination_features(roi_rgb, roi_mask)
        for key, value in illum.items():
            if hasattr(features, key):
                setattr(features, key, value)

    # color_cast 明确保持 null
    features.color_cast_score = None
    features.available_feature_names = tuple(
        name for name, ok in features.availability_map().items() if ok
    )
    return features
