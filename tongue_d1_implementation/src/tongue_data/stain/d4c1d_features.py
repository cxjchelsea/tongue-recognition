"""D4-C.1-D：Stained source acquisition / local 特征提取（只读诊断，无 CNN）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .d4c1a_features import (
    compute_roi_color_features,
    compute_roi_geometry_features,
    compute_sharpness_features,
)


# 预注册 feature families（禁止 fishing）
COLOR_FEATURES = [
    "RGB_mean_r",
    "RGB_mean_g",
    "RGB_mean_b",
    "RGB_median_r",
    "RGB_median_g",
    "RGB_median_b",
    "rg_ratio",
    "bg_ratio",
    "Lab_L_mean",
    "Lab_a_mean",
    "Lab_b_mean",
    "Lab_L_median",
    "Lab_a_median",
    "Lab_b_median",
    "HSV_h_mean",
    "HSV_s_mean",
    "HSV_v_mean",
    "luminance_mean",
    "luminance_std",
    "luminance_p05",
    "luminance_p50",
    "luminance_p95",
]

RESOLUTION_FEATURES = [
    "original_width",
    "original_height",
    "original_pixel_count",
    "ROI_width",
    "ROI_height",
    "ROI_short_side",
    "ROI_pixel_count",
    "bbox_area_ratio",
]

GEOMETRY_FEATURES = [
    "ROI_aspect_ratio",
    "foreground_ratio",
    "black_fill_ratio",
    "padding_ratio",
    "bbox_area_ratio",
]

QUALITY_FEATURES = [
    "blur_laplacian",
    "tenengrad",
    "clipping_dark_ratio",
    "clipping_bright_ratio",
    "original_pixel_count",
    "ROI_short_side",
]

LOCAL_FEATURES = [
    "local_chroma_var",
    "spatial_color_var",
    "patch_lab_var",
    "center_edge_delta_l",
    "center_edge_delta_a",
    "center_edge_delta_b",
    "sat_p95",
    "sat_extrema_ratio",
]

FORBIDDEN_CLASSIFIER_COLS = {
    "sample_id",
    "md5",
    "source_image_path",
    "roi_rgb_path",
    "roi_mask_path",
    "path",
    "filename",
    "split",
    "dataset",
    "stain_label",
    "label",
    "canonical_label",
    "folder_batch",
    "file_extension",
}


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {key: float("nan") for key in ("mean", "std", "p05", "p25", "p50", "p75", "p95")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_clipping_features(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = np.asarray(roi_mask) > 0
    if not mask.any():
        return {"clipping_dark_ratio": float("nan"), "clipping_bright_ratio": float("nan")}
    pixels = rgb[mask]
    dark = float((pixels.max(axis=1) <= 5).mean())
    bright = float((pixels.min(axis=1) >= 250).mean())
    return {"clipping_dark_ratio": dark, "clipping_bright_ratio": bright}


def compute_local_heterogeneity(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """局部颜色异质性 proxy（非 CNN）。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = (np.asarray(roi_mask) > 0).astype(np.uint8)
    out = {key: float("nan") for key in LOCAL_FEATURES}
    if not mask.any():
        return out
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    # chroma ≈ sqrt(a^2+b^2)（OpenCV a/b 偏移 128）
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    out["local_chroma_var"] = float(chroma[mask > 0].var())
    # 空间颜色方差：RGB 通道方差均值
    pixels = rgb[mask > 0].astype(np.float64)
    out["spatial_color_var"] = float(pixels.var(axis=0).mean())
    # patch Lab 方差：非重叠 16x16
    patch = 16
    height, width = rgb.shape[:2]
    patch_vars = []
    for row in range(0, height - patch + 1, patch):
        for col in range(0, width - patch + 1, patch):
            sub_mask = mask[row : row + patch, col : col + patch]
            if sub_mask.mean() < 0.5:
                continue
            sub_lab = lab[row : row + patch, col : col + patch][sub_mask > 0]
            if len(sub_lab) < 8:
                continue
            patch_vars.append(float(sub_lab.var(axis=0).mean()))
    out["patch_lab_var"] = float(np.mean(patch_vars)) if patch_vars else float("nan")
    # center vs edge
    ys, xs = np.where(mask > 0)
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    thresh = float(np.percentile(dist, 40))
    center = dist <= thresh
    edge = dist >= float(np.percentile(dist, 70))
    lab_pixels = lab[mask > 0]
    if center.any() and edge.any():
        out["center_edge_delta_l"] = float(
            lab_pixels[center, 0].mean() - lab_pixels[edge, 0].mean()
        )
        out["center_edge_delta_a"] = float(
            lab_pixels[center, 1].mean() - lab_pixels[edge, 1].mean()
        )
        out["center_edge_delta_b"] = float(
            lab_pixels[center, 2].mean() - lab_pixels[edge, 2].mean()
        )
    sat = hsv[mask > 0, 1]
    out["sat_p95"] = float(np.percentile(sat, 95))
    out["sat_extrema_ratio"] = float((sat >= np.percentile(sat, 90)).mean())
    return out


def compute_padding_ratio(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """padding / black fill：mask 外黑色占比。"""
    rgb = np.asarray(roi_rgb, dtype=np.uint8)
    mask = np.asarray(roi_mask) > 0
    outside = ~mask
    if outside.any():
        black = (rgb.max(axis=2) <= 2) & outside
        padding = float(black.mean())
        black_fill = float(black.sum() / max(outside.sum(), 1))
    else:
        padding = 0.0
        black_fill = 0.0
    return {
        "padding_ratio": padding,
        "black_fill_ratio": black_fill if outside.any() else 0.0,
    }


def read_exif_summary(image_path: str | Path) -> dict[str, Any]:
    """EXIF 可选；缺失则 null。"""
    keys = {
        "exif_make": None,
        "exif_model": None,
        "exif_software": None,
        "exif_iso": None,
        "exif_exposure": None,
        "exif_white_balance": None,
        "exif_flash": None,
    }
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return keys
            # 常用 TIFF tags
            mapping = {
                271: "exif_make",
                272: "exif_model",
                305: "exif_software",
                34855: "exif_iso",
                33434: "exif_exposure",
                41987: "exif_white_balance",
                37385: "exif_flash",
            }
            for tag_id, name in mapping.items():
                if tag_id in exif:
                    value = exif.get(tag_id)
                    keys[name] = str(value) if value is not None else None
    except Exception:
        return keys
    return keys


def extract_folder_batch(source_path: str) -> str:
    """从路径提取粗 batch（倒数第二级目录）。"""
    parts = Path(source_path).parts
    if len(parts) >= 2:
        return str(parts[-2])
    return "unknown"


def compute_dhash(roi_rgb: np.ndarray, hash_size: int = 8) -> str:
    """简单 difference hash（near-duplicate audit）。"""
    gray = cv2.cvtColor(np.asarray(roi_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if value else "0" for value in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def extract_sample_features(
    *,
    sample_id: str,
    split: str,
    stain_label: int,
    source_image_path: str,
    md5: str,
    roi_rgb_path: str,
    roi_mask_path: str,
    original_width: float | None,
    original_height: float | None,
    foreground_ratio_manifest: float | None = None,
) -> dict[str, Any]:
    """单样本完整 confounding feature 行。"""
    roi_rgb = np.asarray(Image.open(roi_rgb_path).convert("RGB"), dtype=np.uint8)
    roi_mask = (np.asarray(Image.open(roi_mask_path)) > 0).astype(np.uint8)

    # 若 manifest 缺原图尺寸，尝试从源图读取
    width = original_width
    height = original_height
    file_size = None
    file_extension = Path(source_image_path).suffix.lower() if source_image_path else None
    if source_image_path and Path(source_image_path).exists():
        file_size = int(Path(source_image_path).stat().st_size)
        if not width or not height:
            with Image.open(source_image_path) as source_image:
                width, height = source_image.size
    else:
        # 回退：用 ROI 尺寸
        if not width:
            width = float(roi_rgb.shape[1])
        if not height:
            height = float(roi_rgb.shape[0])

    color = compute_roi_color_features(roi_rgb, roi_mask)
    geom = compute_roi_geometry_features(
        roi_rgb,
        roi_mask,
        original_width=float(width) if width else None,
        original_height=float(height) if height else None,
    )
    sharp = compute_sharpness_features(roi_rgb, roi_mask)
    clip = compute_clipping_features(roi_rgb, roi_mask)
    pad = compute_padding_ratio(roi_rgb, roi_mask)
    local = compute_local_heterogeneity(roi_rgb, roi_mask)
    exif = read_exif_summary(source_image_path) if source_image_path else {}

    # Lab median
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    mask = roi_mask > 0
    if mask.any():
        lab_pixels = lab[mask]
        lab_l_med = float(np.median(lab_pixels[:, 0]) * (100.0 / 255.0))
        lab_a_med = float(np.median(lab_pixels[:, 1]) - 128.0)
        lab_b_med = float(np.median(lab_pixels[:, 2]) - 128.0)
    else:
        lab_l_med = lab_a_med = lab_b_med = float("nan")

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "stain_label": int(stain_label),
        "source_image_path": source_image_path,
        "md5": md5,
        "original_width": float(width),
        "original_height": float(height),
        "original_pixel_count": float(width) * float(height),
        "file_extension": file_extension,
        "file_size": file_size,
        "ROI_width": float(geom["roi_width"]),
        "ROI_height": float(geom["roi_height"]),
        "ROI_short_side": float(sharp["roi_short_side"]),
        "ROI_pixel_count": float(sharp["roi_pixel_count"]),
        "ROI_aspect_ratio": float(geom["roi_aspect_ratio"]),
        "bbox_area_ratio": float(geom.get("bbox_area_ratio", float("nan"))),
        "foreground_ratio": float(
            foreground_ratio_manifest
            if foreground_ratio_manifest is not None
            else geom["foreground_ratio"]
        ),
        "black_fill_ratio": pad["black_fill_ratio"],
        "padding_ratio": pad["padding_ratio"],
        "blur_laplacian": float(sharp["laplacian_var"]),
        "tenengrad": float(sharp["tenengrad_mean"]),
        "luminance_mean": float(color["luminance_mean"]),
        "luminance_std": float(color["luminance_std"]),
        "luminance_p05": float(color["luminance_p05"]),
        "luminance_p50": float(color["luminance_p50"]),
        "luminance_p95": float(color["luminance_p95"]),
        "RGB_mean_r": float(color["mean_r"]),
        "RGB_mean_g": float(color["mean_g"]),
        "RGB_mean_b": float(color["mean_b"]),
        "RGB_median_r": float(color["median_r"]),
        "RGB_median_g": float(color["median_g"]),
        "RGB_median_b": float(color["median_b"]),
        "rg_ratio": float(color["rg_ratio"]),
        "bg_ratio": float(color["bg_ratio"]),
        "Lab_L_mean": float(color["mean_l"]),
        "Lab_a_mean": float(color["mean_a"]),
        "Lab_b_mean": float(color["mean_b_lab"]),
        "Lab_L_median": lab_l_med,
        "Lab_a_median": lab_a_med,
        "Lab_b_median": lab_b_med,
        "HSV_h_mean": float(color["mean_h"]),
        "HSV_s_mean": float(color["mean_s"]),
        "HSV_v_mean": float(color["mean_v"]),
        "clipping_dark_ratio": clip["clipping_dark_ratio"],
        "clipping_bright_ratio": clip["clipping_bright_ratio"],
        "folder_batch": extract_folder_batch(source_image_path or ""),
        "dhash": compute_dhash(roi_rgb),
        "roi_rgb_path": roi_rgb_path,
        "roi_mask_path": roi_mask_path,
    }
    row.update(local)
    row.update(exif)
    return row


def all_acquisition_features() -> list[str]:
    """去重后的 acquisition feature 全集。"""
    seen: list[str] = []
    for name in COLOR_FEATURES + RESOLUTION_FEATURES + GEOMETRY_FEATURES + QUALITY_FEATURES:
        if name not in seen:
            seen.append(name)
    return seen


def summarize_feature(values: np.ndarray) -> dict[str, float]:
    return _percentile_stats(values)
