"""D4-C.1-A：ROI 颜色 / 几何 / 锐度特征（只读诊断）。"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_roi_color_features(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """舌头 ROI 内颜色统计。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = np.asarray(roi_mask) > 0
    out: dict[str, float] = {}
    if not mask.any():
        keys = [
            "mean_r",
            "mean_g",
            "mean_b",
            "median_r",
            "median_g",
            "median_b",
            "mean_l",
            "mean_a",
            "mean_b_lab",
            "mean_h",
            "mean_s",
            "mean_v",
            "luminance_mean",
            "luminance_std",
            "luminance_p05",
            "luminance_p50",
            "luminance_p95",
            "rg_ratio",
            "bg_ratio",
            "rb_ratio",
        ]
        return {key: float("nan") for key in keys}

    pixels = rgb[mask].astype(np.float64)
    out["mean_r"], out["mean_g"], out["mean_b"] = pixels.mean(axis=0)
    out["median_r"], out["median_g"], out["median_b"] = np.median(pixels, axis=0)
    # OpenCV Lab：L[0,100] 被缩放到 0-255；a/b 也有偏移，转回近似标准
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab_pixels = lab[mask]
    out["mean_l"] = float(lab_pixels[:, 0].mean() * (100.0 / 255.0))
    out["mean_a"] = float(lab_pixels[:, 1].mean() - 128.0)
    out["mean_b_lab"] = float(lab_pixels[:, 2].mean() - 128.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv_pixels = hsv[mask]
    out["mean_h"] = float(hsv_pixels[:, 0].mean())
    out["mean_s"] = float(hsv_pixels[:, 1].mean())
    out["mean_v"] = float(hsv_pixels[:, 2].mean())
    luminance = (
        0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    )
    lum = _safe_stats(luminance)
    out["luminance_mean"] = lum["mean"]
    out["luminance_std"] = lum["std"]
    out["luminance_p05"] = lum["p05"]
    out["luminance_p50"] = lum["p50"]
    out["luminance_p95"] = lum["p95"]
    mean_g = max(out["mean_g"], 1e-6)
    out["rg_ratio"] = float(out["mean_r"] / mean_g)
    out["bg_ratio"] = float(out["mean_b"] / mean_g)
    out["rb_ratio"] = float(out["mean_r"] / max(out["mean_b"], 1e-6))
    return out


def compute_roi_geometry_features(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    *,
    original_width: float | None = None,
    original_height: float | None = None,
) -> dict[str, float]:
    """ROI / mask 几何特征。"""
    rgb = np.asarray(roi_rgb)
    mask = (np.asarray(roi_mask) > 0).astype(np.uint8)
    height, width = rgb.shape[:2]
    out: dict[str, float] = {
        "roi_width": float(width),
        "roi_height": float(height),
        "roi_aspect_ratio": float(width / max(height, 1)),
        "tongue_pixel_count": float(mask.sum()),
        "foreground_ratio": float(mask.mean()) if mask.size else float("nan"),
    }
    if original_width and original_height:
        out["roi_width_ratio"] = float(width / original_width)
        out["roi_height_ratio"] = float(height / original_height)
        out["bbox_area_ratio"] = float((width * height) / (original_width * original_height))
    else:
        out["roi_width_ratio"] = float("nan")
        out["roi_height_ratio"] = float("nan")
        out["bbox_area_ratio"] = float("nan")

    if not mask.any():
        for key in (
            "centroid_y_norm",
            "centroid_x_norm",
            "perimeter_area_ratio",
            "compactness",
            "extent",
            "solidity",
        ):
            out[key] = float("nan")
        return out

    ys, xs = np.where(mask > 0)
    out["centroid_y_norm"] = float(ys.mean() / max(height, 1))
    out["centroid_x_norm"] = float(xs.mean() / max(width, 1))
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        for key in ("perimeter_area_ratio", "compactness", "extent", "solidity"):
            out[key] = float("nan")
        return out
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    out["perimeter_area_ratio"] = float(perimeter / max(area, 1e-6))
    out["compactness"] = float((perimeter ** 2) / max(4.0 * np.pi * area, 1e-6))
    x, y, w_box, h_box = cv2.boundingRect(contour)
    out["extent"] = float(area / max(w_box * h_box, 1e-6))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    out["solidity"] = float(area / max(hull_area, 1e-6))
    return out


def compute_sharpness_features(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """锐度 / 噪声 proxy。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = np.asarray(roi_mask) > 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = gx * gx + gy * gy
    if mask.any():
        lap_vals = lap[mask]
        ten_vals = tenengrad[mask]
        local = gray.astype(np.float64)
        noise_proxy = float(np.mean(np.abs(local[mask] - cv2.GaussianBlur(local, (5, 5), 0)[mask])))
    else:
        lap_vals = lap.ravel()
        ten_vals = tenengrad.ravel()
        noise_proxy = float("nan")
    return {
        "laplacian_var": float(lap_vals.var()) if lap_vals.size else float("nan"),
        "tenengrad_mean": float(ten_vals.mean()) if ten_vals.size else float("nan"),
        "noise_proxy": noise_proxy,
        "roi_short_side": float(min(rgb.shape[0], rgb.shape[1])),
        "roi_pixel_count": float(rgb.shape[0] * rgb.shape[1]),
    }


def compute_all_roi_features(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    *,
    original_width: float | None = None,
    original_height: float | None = None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(compute_roi_color_features(roi_rgb, roi_mask))
    features.update(
        compute_roi_geometry_features(
            roi_rgb,
            roi_mask,
            original_width=original_width,
            original_height=original_height,
        )
    )
    features.update(compute_sharpness_features(roi_rgb, roi_mask))
    return features


def group_quantile_table(frame, group_col: str, value_cols: list[str]) -> dict[str, Any]:
    """按组汇总分位数。"""
    import pandas as pd

    out: dict[str, Any] = {}
    for group_name, subset in frame.groupby(group_col):
        block = {"n": int(len(subset))}
        for column in value_cols:
            values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                block[column] = None
            else:
                block[column] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p05": float(np.percentile(values, 5)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                }
        out[str(group_name)] = block
    return out
