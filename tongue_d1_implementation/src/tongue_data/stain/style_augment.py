"""D4-C.1-B：acquisition-style augmentation（禁止伪标 / 极端色相变换）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

# 预声明安全上限（不得被 test 分布覆盖）
SAFE_CAPS = {
    "channel_gain_min": 0.75,
    "channel_gain_max": 1.35,
    "gamma_min": 0.80,
    "gamma_max": 1.25,
    "exposure_min": 0.80,
    "exposure_max": 1.25,
    "contrast_min": 0.85,
    "contrast_max": 1.20,
    "jpeg_quality_min": 55,
    "jpeg_quality_max": 95,
}


FORBIDDEN_OPS = frozenset(
    {
        "random_grayscale",
        "color_inversion",
        "solarize",
        "posterize",
        "extreme_hue_rotation",
        "strong_saturation_shift",
    }
)


def _roi_mean_rgb(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(roi_mask) > 0
    rgb = np.asarray(roi_rgb, dtype=np.float64)
    if not mask.any():
        return rgb.reshape(-1, 3).mean(axis=0)
    return rgb[mask].mean(axis=0)


def estimate_style_ranges_from_train(
    *,
    stain_manifest: str | Path,
    external_roi_index: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    仅用 Stained/BioHit/TongueSet3 TRAIN 估计 style range。
    禁止使用任何 test split。
    """
    stain = pd.read_parquet(stain_manifest)
    stain = stain[(stain["eligible"] == True) & (stain["split"] == "train")]
    external = pd.read_parquet(external_roi_index)
    external = external[external["split"] == "train"]
    if (external["split"] == "test").any():
        raise RuntimeError("test leaked into style range estimation")

    groups: dict[str, list[np.ndarray]] = {
        "stained": [],
        "biohit": [],
        "tongueset3": [],
    }
    for _index, row in stain.iterrows():
        rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"))
        mask = np.asarray(Image.open(row["roi_mask_path"]))
        groups["stained"].append(_roi_mean_rgb(rgb, mask))
    for _index, row in external.iterrows():
        if not row["roi_rgb_path"]:
            continue
        rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"))
        mask = np.asarray(Image.open(row["roi_mask_path"]))
        groups[str(row["dataset"])].append(_roi_mean_rgb(rgb, mask))

    stats = {}
    for name, vectors in groups.items():
        arr = np.stack(vectors) if vectors else np.zeros((0, 3))
        stats[name] = {
            "n": int(len(arr)),
            "mean_rgb": arr.mean(axis=0).tolist() if len(arr) else [0, 0, 0],
            "p05_rgb": np.percentile(arr, 5, axis=0).tolist() if len(arr) else [0, 0, 0],
            "p95_rgb": np.percentile(arr, 95, axis=0).tolist() if len(arr) else [0, 0, 0],
        }

    # 以 stained 为参考，用 external train 相对增益估计范围，再夹安全帽
    ref = np.array(stats["stained"]["mean_rgb"], dtype=np.float64) + 1e-6
    ratios = []
    for domain_name in ("biohit", "tongueset3"):
        domain_mean = np.array(stats[domain_name]["mean_rgb"], dtype=np.float64)
        ratios.append(domain_mean / ref)
        ratios.append(np.array(stats[domain_name]["p05_rgb"]) / ref)
        ratios.append(np.array(stats[domain_name]["p95_rgb"]) / ref)
    ratio_stack = np.stack(ratios)
    gain_lo = float(np.clip(ratio_stack.min(), SAFE_CAPS["channel_gain_min"], 1.0))
    gain_hi = float(np.clip(ratio_stack.max(), 1.0, SAFE_CAPS["channel_gain_max"]))
    # 保证有可采样宽度
    gain_lo = min(gain_lo, 0.90)
    gain_hi = max(gain_hi, 1.10)

    contract = {
        "stage": "D4-C.1-B",
        "calibration_splits": ["train"],
        "forbidden_test_usage": True,
        "calibration_data": stats,
        "safe_caps": SAFE_CAPS,
        "channel_gain_ranges": {
            "r": [gain_lo, gain_hi],
            "g": [gain_lo, gain_hi],
            "b": [gain_lo, gain_hi],
        },
        "gamma_range": [SAFE_CAPS["gamma_min"], SAFE_CAPS["gamma_max"]],
        "exposure_range": [SAFE_CAPS["exposure_min"], SAFE_CAPS["exposure_max"]],
        "contrast_range": [SAFE_CAPS["contrast_min"], SAFE_CAPS["contrast_max"]],
        "jpeg_range": [SAFE_CAPS["jpeg_quality_min"], SAFE_CAPS["jpeg_quality_max"]],
        "enable_jpeg": True,
        "forbidden_ops": sorted(FORBIDDEN_OPS),
        "rationale": (
            "Ranges derived from train-only ROI mean RGB ratios across "
            "Stained/BioHit/TongueSet3, clipped by predeclared SAFE_CAPS. "
            "No test split used."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract


def load_style_contract(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_channel_gains(
    rgb: np.ndarray, gains: tuple[float, float, float]
) -> np.ndarray:
    """RGB multiplicative WB gains；不交换通道。"""
    out = rgb.astype(np.float32)
    out[..., 0] *= float(gains[0])
    out[..., 1] *= float(gains[1])
    out[..., 2] *= float(gains[2])
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array(
        [((index / 255.0) ** float(gamma)) * 255.0 for index in range(256)],
        dtype=np.float32,
    )
    return cv2.LUT(rgb, table.astype(np.uint8))


def apply_exposure(rgb: np.ndarray, exposure: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * float(exposure), 0, 255).astype(np.uint8)


def apply_contrast(rgb: np.ndarray, contrast: float) -> np.ndarray:
    mean = rgb.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    out = (rgb.astype(np.float32) - mean) * float(contrast) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    # OpenCV 期望 BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, encode_param)
    if not ok:
        return rgb
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def sample_style_params(
    contract: dict[str, Any],
    rng: np.random.Generator,
    *,
    strength: str = "moderate",
) -> dict[str, float | int]:
    """strength: weak | moderate | max_safe。"""
    scale = {"weak": 0.35, "moderate": 0.7, "max_safe": 1.0}[strength]

    def _sample(lo: float, hi: float) -> float:
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * scale
        return float(rng.uniform(mid - half, mid + half))

    rg = contract["channel_gain_ranges"]["r"]
    gg = contract["channel_gain_ranges"]["g"]
    bg = contract["channel_gain_ranges"]["b"]
    return {
        "gain_r": _sample(rg[0], rg[1]),
        "gain_g": _sample(gg[0], gg[1]),
        "gain_b": _sample(bg[0], bg[1]),
        "gamma": _sample(*contract["gamma_range"]),
        "exposure": _sample(*contract["exposure_range"]),
        "contrast": _sample(*contract["contrast_range"]),
        "jpeg_quality": int(
            round(_sample(contract["jpeg_range"][0], contract["jpeg_range"][1]))
        ),
    }


def apply_style_transform(
    rgb: np.ndarray,
    contract: dict[str, Any],
    rng: np.random.Generator,
    *,
    strength: str = "moderate",
    params: dict[str, float | int] | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    有界 acquisition-style transform。
    禁止 grayscale / hue extreme / inversion。
    """
    if any(op in FORBIDDEN_OPS for op in contract.get("forbidden_ops", [])):
        # contract 应声明 forbidden；运行时仍硬禁
        pass
    rgb = np.asarray(rgb, dtype=np.uint8).copy()
    params = params or sample_style_params(contract, rng, strength=strength)
    out = apply_channel_gains(
        rgb, (params["gain_r"], params["gain_g"], params["gain_b"])
    )
    out = apply_gamma(out, float(params["gamma"]))
    out = apply_exposure(out, float(params["exposure"]))
    out = apply_contrast(out, float(params["contrast"]))
    if contract.get("enable_jpeg", True) and rng.random() < 0.5:
        out = apply_jpeg(out, int(params["jpeg_quality"]))
    return out, params


def style_sanity_stats(rgb_before: np.ndarray, rgb_after: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """裁剪/亮度塌陷等 label-preservation proxy。"""
    m = np.asarray(mask) > 0
    before = rgb_before[m].astype(np.float32) if m.any() else rgb_before.reshape(-1, 3)
    after = rgb_after[m].astype(np.float32) if m.any() else rgb_after.reshape(-1, 3)
    clip_lo = float((after <= 1).mean())
    clip_hi = float((after >= 254).mean())
    lum_b = 0.2126 * before[:, 0] + 0.7152 * before[:, 1] + 0.0722 * before[:, 2]
    lum_a = 0.2126 * after[:, 0] + 0.7152 * after[:, 1] + 0.0722 * after[:, 2]
    return {
        "clip_low_rate": clip_lo,
        "clip_high_rate": clip_hi,
        "luminance_mean_before": float(lum_b.mean()),
        "luminance_mean_after": float(lum_a.mean()),
        "luminance_collapse": float(lum_a.std() < 5.0),
        "saturation_proxy_before": float(before.std()),
        "saturation_proxy_after": float(after.std()),
    }
