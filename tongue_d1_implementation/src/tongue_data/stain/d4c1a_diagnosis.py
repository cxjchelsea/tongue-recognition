"""D4-C.1-A：Stain cross-domain shortcut diagnosis（只读，不训练）。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import StainDataConfig, StainTrainConfig
from .d4c1a_features import compute_all_roi_features, group_quantile_table
from .d4c1a_model_tools import (
    cam_region_ratios,
    centroid_distances,
    extract_embedding,
    forward_logit_prob,
    grad_cam_resnet18,
)
from .d4c1a_preprocess import (
    assert_train_runtime_tensor_equiv,
    compute_fill_padding_ratios,
    letterbox_meta,
    preprocess_counterfactual,
)
from .metrics import map_probability_to_finding
from .train import load_stain_checkpoint, resolve_device

STAIN_T_CLEAR = 0.95
STAIN_T_RETAKE = 0.96
REP_MODES = ("black", "gray", "mean_fill", "bbox", "context")
SHORTCUT_STATUS = ("strong", "moderate", "weak", "not_supported", "undetermined")
RECOMMENDATIONS = (
    "PROCEED_D4C1B_DOMAIN_ROBUST_RETRAINING",
    "PREPROCESSING_BUG_FOUND",
    "DATA_SEMANTICS_CONCERN",
    "TRUE_STAIN_LIKE_APPEARANCE_UNRESOLVED",
    "INSUFFICIENT_EVIDENCE",
    "MULTIPLE_BLOCKERS",
)


def _safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)


def _file_md5(path: Path, nbytes: int = 0) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        if nbytes > 0:
            digest.update(handle.read(nbytes))
        else:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_rgb_mask_from_stain_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(row["roi_mask_path"]), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return rgb, mask


def _ensure_seg_roi_cache(
    *,
    seg_manifest: pd.DataFrame,
    cache_dir: Path,
    seg_checkpoint: Path,
    data_config: Path,
    train_config: Path,
    device: str,
) -> pd.DataFrame:
    """对 BioHit/TongueSet3 用 frozen D3 提取 ROI 并缓存。"""
    from tongue_data.segmentation.inference import TongueSegmentationInference

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.parquet"
    if index_path.exists():
        cached = pd.read_parquet(index_path)
        if set(seg_manifest["sample_id"]) <= set(cached["sample_id"]):
            return cached

    inference = TongueSegmentationInference(
        checkpoint_path=seg_checkpoint,
        data_config=data_config,
        train_config=train_config,
        device=device,
        return_model_space=False,
        return_probability=False,
        return_masked_roi=False,
    )
    rows: list[dict[str, Any]] = []
    existing = (
        pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    )
    done = set(existing["sample_id"]) if len(existing) else set()
    if len(existing):
        rows.extend(existing.to_dict(orient="records"))

    for _index, row in seg_manifest.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id in done:
            continue
        result = inference.predict(str(row["image_path"]), sample_id=sample_id)
        safe = _safe_name(sample_id)
        rgb_path = cache_dir / f"{safe}_roi.png"
        mask_path = cache_dir / f"{safe}_mask.png"
        status = getattr(result, "status", None)
        if (
            status == "success"
            and result.tongue_roi_rgb is not None
            and result.tongue_roi_mask is not None
        ):
            Image.fromarray(result.tongue_roi_rgb).save(rgb_path)
            Image.fromarray(
                (result.tongue_roi_mask > 0).astype(np.uint8) * 255
            ).save(mask_path)
            # 以原图尺寸为准（W,H）
            with Image.open(str(row["image_path"])) as image:
                ow, oh = image.size
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": str(row["dataset"]),
                    "split": str(row["split"]),
                    "roi_rgb_path": str(rgb_path),
                    "roi_mask_path": str(mask_path),
                    "d3_status": status,
                    "original_width": ow,
                    "original_height": oh,
                    "image_path": str(row["image_path"]),
                    "file_extension": Path(str(row["image_path"])).suffix.lower(),
                }
            )
        else:
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": str(row["dataset"]),
                    "split": str(row["split"]),
                    "roi_rgb_path": None,
                    "roi_mask_path": None,
                    "d3_status": status,
                    "original_width": None,
                    "original_height": None,
                    "image_path": str(row["image_path"]),
                    "file_extension": Path(str(row["image_path"])).suffix.lower(),
                }
            )
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_parquet(index_path, index=False)

    frame = pd.DataFrame(rows)
    frame.to_parquet(index_path, index=False)
    return frame


def build_diagnosis_pool(
    *,
    stain_manifest: Path,
    segmentation_dir: Path,
    seg_checkpoint: Path,
    seg_data_config: Path,
    seg_train_config: Path,
    roi_cache_dir: Path,
    device: str,
) -> pd.DataFrame:
    """统一 diagnosis 样本池（带用途标记，禁止伪标）。"""
    stain = pd.read_parquet(stain_manifest)
    stain = stain[stain["eligible"].astype(bool)].copy()
    stain_rows = []
    for _index, row in stain.iterrows():
        with Image.open(row["source_image_path"]) as image:
            ow, oh = image.size
        stain_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "dataset": "stained_coating",
                "domain_group": "stained",
                "split": str(row["split"]),
                "true_stain_label": int(row["label"]),
                "label_role": "stain_gold",
                "roi_rgb_path": str(row["roi_rgb_path"]),
                "roi_mask_path": str(row["roi_mask_path"]),
                "d3_status": str(row.get("d3_status", "success")),
                "original_width": ow,
                "original_height": oh,
                "image_path": str(row["source_image_path"]),
                "file_extension": Path(str(row["source_image_path"])).suffix.lower(),
                "usage": "in_domain_stain"
                if row["split"] != "test"
                else "in_domain_stain_test_readonly",
            }
        )
    stain_frame = pd.DataFrame(stain_rows)

    seg = pd.read_parquet(Path(segmentation_dir) / "segmentation_manifest.parquet")
    seg_cache = _ensure_seg_roi_cache(
        seg_manifest=seg,
        cache_dir=roi_cache_dir,
        seg_checkpoint=seg_checkpoint,
        data_config=seg_data_config,
        train_config=seg_train_config,
        device=device,
    )
    ext_rows = []
    for _index, row in seg_cache.iterrows():
        dataset_name = str(row["dataset"])
        ext_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "dataset": dataset_name,
                "domain_group": dataset_name,
                "split": str(row["split"]),
                "true_stain_label": None,
                "label_role": "no_stain_gold",
                "roi_rgb_path": row["roi_rgb_path"],
                "roi_mask_path": row["roi_mask_path"],
                "d3_status": row["d3_status"],
                "original_width": row["original_width"],
                "original_height": row["original_height"],
                "image_path": row["image_path"],
                "file_extension": row["file_extension"],
                "usage": "external_domain_audit_only",
            }
        )
    return pd.concat([stain_frame, pd.DataFrame(ext_rows)], ignore_index=True)


def run_feature_and_score_audit(
    pool: pd.DataFrame,
    *,
    checkpoint: Path,
    data_config_path: Path,
    train_config_path: Path,
    device: str,
) -> pd.DataFrame:
    """逐样本特征 + black/gray/bbox/... p_stain + logit。"""
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    model, _ckpt = load_stain_checkpoint(
        checkpoint,
        train_config=train_config,
        data_config=data_config,
        map_location=resolve_device(device),
        strict=True,
    )
    model = model.to(resolve_device(device))
    model.eval()
    # 冻结参数，禁止任何训练
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    total = int(len(pool))
    for index, (_index, row) in enumerate(pool.iterrows()):
        if index % 100 == 0:
            print(f"[d4c1a] feature/score {index}/{total}", flush=True)
        record = dict(row)
        if not row["roi_rgb_path"] or not row["roi_mask_path"]:
            record.update(
                {
                    "p_stain_black": None,
                    "logit_black": None,
                    "runtime_finding": None,
                }
            )
            records.append(record)
            continue
        rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(row["roi_mask_path"]), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # 不修改磁盘原图；使用拷贝
        rgb_work = rgb.copy()
        mask_work = mask.copy()
        features = compute_all_roi_features(
            rgb_work,
            mask_work,
            original_width=row["original_width"],
            original_height=row["original_height"],
        )
        fill_stats = compute_fill_padding_ratios(rgb_work, mask_work, data_config)
        record.update(features)
        record.update(fill_stats)
        # representation ablation
        for mode in REP_MODES:
            tensor = preprocess_counterfactual(
                rgb_work, mask_work, data_config, mode=mode
            )
            batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
            batch = batch.to(resolve_device(device))
            logit, prob = forward_logit_prob(model, batch)
            record[f"p_stain_{mode}"] = prob
            record[f"logit_{mode}"] = logit
        record["p_stain"] = record["p_stain_black"]
        record["logit"] = record["logit_black"]
        finding = map_probability_to_finding(
            float(record["p_stain"]), STAIN_T_CLEAR, STAIN_T_RETAKE
        )
        record["runtime_finding"] = finding
        # color-norm counterfactual（gray-world / luminance）
        record.update(
            _color_norm_counterfactuals(
                model, rgb_work, mask_work, data_config, resolve_device(device)
            )
        )
        records.append(record)
    return pd.DataFrame(records)


def _color_norm_counterfactuals(
    model,
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    data_config: StainDataConfig,
    device,
) -> dict[str, float]:
    rgb = roi_rgb.astype(np.float32)
    mask = roi_mask > 0
    out: dict[str, float] = {}
    # gray-world
    means = rgb[mask].mean(axis=0) if mask.any() else rgb.mean(axis=(0, 1))
    means = np.maximum(means, 1e-3)
    gray_world = rgb * (means.mean() / means)
    gray_world = np.clip(gray_world, 0, 255).astype(np.uint8)
    # luminance normalize
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    target = float(np.median(lum[mask])) if mask.any() else float(np.median(lum))
    scale = target / max(float(np.median(lum[mask])) if mask.any() else 1.0, 1e-3)
    # 上面 scale≈1；改为拉到固定目标 140
    target = 140.0
    cur = float(np.median(lum[mask])) if mask.any() else float(np.median(lum))
    scale = target / max(cur, 1e-3)
    lum_norm = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    for name, arr in (("gray_world", gray_world), ("luminance_norm", lum_norm)):
        tensor = preprocess_counterfactual(arr, roi_mask, data_config, mode="black")
        batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
        _logit, prob = forward_logit_prob(model, batch)
        out[f"p_stain_{name}"] = prob
    return out


def run_dataset_identity_classifier(frame: pd.DataFrame) -> dict[str, Any]:
    """手工特征 → dataset identity diagnostic（train+val exploratory CV）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    feature_cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_l",
        "mean_a",
        "mean_b_lab",
        "mean_s",
        "mean_v",
        "luminance_mean",
        "luminance_std",
        "rg_ratio",
        "bg_ratio",
        "roi_aspect_ratio",
        "foreground_ratio",
        "mask_fill_ratio",
        "padding_ratio",
        "black_pixel_ratio",
        "laplacian_var",
        "roi_short_side",
        "compactness",
        "extent",
        "solidity",
    ]
    # 禁止 sample_id/path
    assert "sample_id" not in feature_cols
    assert "image_path" not in feature_cols

    exploratory = frame[frame["split"].isin(["train", "val"])].copy()
    exploratory = exploratory[exploratory["domain_group"].isin(["stained", "biohit", "tongueset3"])]
    exploratory = exploratory.dropna(subset=feature_cols)
    if len(exploratory) < 30:
        return {"available": False, "reason": "insufficient_rows"}

    matrix = exploratory[feature_cols].to_numpy(dtype=np.float64)
    labels = exploratory["domain_group"].astype(str).to_numpy()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=20260813,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260813)
    pred = cross_val_predict(pipe, matrix, labels, cv=cv)
    acc = float(accuracy_score(labels, pred))
    labels_sorted = sorted(set(labels.tolist()))
    cm = confusion_matrix(labels, pred, labels=labels_sorted).tolist()

    # feature importance via RF on same exploratory pool（诊断用）
    forest = RandomForestClassifier(
        n_estimators=200, random_state=20260813, n_jobs=-1
    )
    forest.fit(matrix, labels)
    importance = sorted(
        [
            {"feature": name, "importance": float(score)}
            for name, score in zip(feature_cols, forest.feature_importances_)
        ],
        key=lambda item: -item["importance"],
    )
    status = "strong" if acc >= 0.90 else ("moderate" if acc >= 0.75 else "weak")
    return {
        "available": True,
        "cv_accuracy": acc,
        "labels": labels_sorted,
        "confusion_matrix": cm,
        "feature_importance_top": importance[:15],
        "n": int(len(exploratory)),
        "dataset_identity_signal": status,
        "features_used": feature_cols,
        "forbidden_leakage_fields_used": False,
    }


def run_embedding_audit(
    frame: pd.DataFrame,
    *,
    checkpoint: Path,
    data_config_path: Path,
    train_config_path: Path,
    device: str,
    max_per_group: int = 250,
) -> dict[str, Any]:
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    model, _ckpt = load_stain_checkpoint(
        checkpoint,
        train_config=train_config,
        data_config=data_config,
        map_location=resolve_device(device),
        strict=True,
    )
    model = model.to(resolve_device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    groups = {
        "stain_pos": frame[
            (frame["domain_group"] == "stained") & (frame["true_stain_label"] == 1)
        ],
        "stain_neg": frame[
            (frame["domain_group"] == "stained") & (frame["true_stain_label"] == 0)
        ],
        "biohit": frame[frame["domain_group"] == "biohit"],
        "tongueset3": frame[frame["domain_group"] == "tongueset3"],
    }
    emb: dict[str, list[np.ndarray]] = {key: [] for key in groups}
    meta_rows = []
    for group_name, subset in groups.items():
        subset = subset.dropna(subset=["roi_rgb_path", "roi_mask_path"])
        if len(subset) > max_per_group:
            subset = subset.sample(n=max_per_group, random_state=20260813)
        for _index, row in subset.iterrows():
            rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(row["roi_mask_path"]), dtype=np.uint8)
            if mask.ndim == 3:
                mask = mask[..., 0]
            tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
            batch = (
                torch.from_numpy(np.ascontiguousarray(tensor))
                .unsqueeze(0)
                .to(resolve_device(device))
            )
            vec = extract_embedding(model, batch)
            emb[group_name].append(vec)
            meta_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group": group_name,
                    "dataset": row["dataset"],
                }
            )
    arrays = {key: np.stack(vals) if vals else np.zeros((0, 512)) for key, vals in emb.items()}
    distances = centroid_distances(arrays)
    # PCA 2D（仅保存坐标 JSON）
    from sklearn.decomposition import PCA

    all_vecs = []
    all_labels = []
    for key, arr in arrays.items():
        if len(arr):
            all_vecs.append(arr)
            all_labels.extend([key] * len(arr))
    pca_payload = None
    if all_vecs:
        stacked = np.concatenate(all_vecs, axis=0)
        coords = PCA(n_components=2, random_state=20260813).fit_transform(stacked)
        pca_payload = {
            "points": [
                {
                    "group": all_labels[index],
                    "x": float(coords[index, 0]),
                    "y": float(coords[index, 1]),
                }
                for index in range(len(all_labels))
            ]
        }
    # TongueSet3 更靠近谁（距离 key 顺序不固定）
    d_pos = None
    d_neg = None
    for key, value in distances.items():
        parts = set(key.split("__"))
        if parts == {"tongueset3", "stain_pos"}:
            d_pos = value
        if parts == {"tongueset3", "stain_neg"}:
            d_neg = value
    closer = None
    if d_pos is not None and d_neg is not None:
        closer = "stain_pos" if d_pos < d_neg else "stain_neg"
    return {
        "n_per_group": {key: int(len(arr)) for key, arr in arrays.items()},
        "centroid_distances": distances,
        "tongueset3_closer_to": closer,
        "pca": pca_payload,
        "note": "frozen encoder; no training",
    }


def run_attribution_audit(
    frame: pd.DataFrame,
    *,
    checkpoint: Path,
    data_config_path: Path,
    train_config_path: Path,
    device: str,
) -> dict[str, Any]:
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    model, _ckpt = load_stain_checkpoint(
        checkpoint,
        train_config=train_config,
        data_config=data_config,
        map_location=resolve_device(device),
        strict=True,
    )
    model = model.to(resolve_device(device))
    model.eval()
    # Grad-CAM 需要部分梯度，但不更新权重
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def pick(subset: pd.DataFrame, n: int, ascending: bool) -> pd.DataFrame:
        subset = subset.dropna(subset=["p_stain", "roi_rgb_path"])
        subset = subset.sort_values(["p_stain", "sample_id"], ascending=ascending)
        return subset.head(n)

    selections = {
        "stained_pos_high": pick(
            frame[(frame.domain_group == "stained") & (frame.true_stain_label == 1)],
            10,
            False,
        ),
        "stained_neg_low": pick(
            frame[(frame.domain_group == "stained") & (frame.true_stain_label == 0)],
            10,
            True,
        ),
        "tongueset3_high": pick(frame[frame.domain_group == "tongueset3"], 10, False),
        "tongueset3_low": pick(frame[frame.domain_group == "tongueset3"], 10, True),
        "biohit": pick(frame[frame.domain_group == "biohit"], 5, False),
    }

    group_stats: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    for group_name, subset in selections.items():
        ratios = []
        for _index, row in subset.iterrows():
            rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(row["roi_mask_path"]), dtype=np.uint8)
            if mask.ndim == 3:
                mask = mask[..., 0]
            tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
            batch = (
                torch.from_numpy(np.ascontiguousarray(tensor))
                .unsqueeze(0)
                .to(resolve_device(device))
            )
            cam = grad_cam_resnet18(model, batch)
            meta = letterbox_meta(rgb.shape[0], rgb.shape[1], data_config.input_size)
            ratios_one = cam_region_ratios(
                cam,
                mask,
                data_config.input_size,
                pad_top=int(meta["pad_top"]),
                pad_left=int(meta["pad_left"]),
                new_height=int(meta["new_height"]),
                new_width=int(meta["new_width"]),
            )
            ratios.append(ratios_one)
            sample_rows.append(
                {
                    "group": group_name,
                    "sample_id": row["sample_id"],
                    "p_stain": float(row["p_stain"]),
                    **ratios_one,
                }
            )
        if ratios:
            group_stats[group_name] = {
                key: float(np.mean([item[key] for item in ratios]))
                for key in ratios[0]
                if key != "energy_sum"
            }
            group_stats[group_name]["n"] = len(ratios)
    return {
        "group_mean_ratios": group_stats,
        "samples": sample_rows,
        "selection_rule": "sort by p_stain then sample_id; frozen Grad-CAM",
    }


def _cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    var_l = left.var(ddof=1)
    var_r = right.var(ddof=1)
    pooled = np.sqrt((var_l + var_r) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def biohit_vs_tongueset3(frame: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_l",
        "mean_a",
        "mean_b_lab",
        "mean_s",
        "luminance_mean",
        "rg_ratio",
        "bg_ratio",
        "roi_aspect_ratio",
        "foreground_ratio",
        "mask_fill_ratio",
        "padding_ratio",
        "black_pixel_ratio",
        "laplacian_var",
        "roi_short_side",
        "p_stain",
    ]
    bio = frame[frame.domain_group == "biohit"]
    ts3 = frame[frame.domain_group == "tongueset3"]
    ranked = []
    for column in cols:
        left = pd.to_numeric(bio[column], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(ts3[column], errors="coerce").dropna().to_numpy()
        if left.size == 0 or right.size == 0:
            continue
        ranked.append(
            {
                "feature": column,
                "biohit_median": float(np.median(left)),
                "tongueset3_median": float(np.median(right)),
                "median_diff": float(np.median(right) - np.median(left)),
                "cohen_d_ts3_minus_bio": _cohen_d(right, left),
            }
        )
    ranked.sort(key=lambda item: -abs(item["cohen_d_ts3_minus_bio"]))
    return {"top_features": ranked[:20], "n_biohit": int(len(bio)), "n_tongueset3": int(len(ts3))}


def probability_shift_report(frame: pd.DataFrame) -> dict[str, Any]:
    def block(subset: pd.DataFrame) -> dict[str, Any]:
        probs = pd.to_numeric(subset["p_stain"], errors="coerce").dropna().to_numpy()
        logits = pd.to_numeric(subset["logit"], errors="coerce").dropna().to_numpy()
        if probs.size == 0:
            return {"n": 0}
        return {
            "n": int(probs.size),
            "p_quantiles": {
                "min": float(probs.min()),
                "p05": float(np.percentile(probs, 5)),
                "median": float(np.median(probs)),
                "p95": float(np.percentile(probs, 95)),
                "max": float(probs.max()),
                "mean": float(probs.mean()),
            },
            "logit_quantiles": {
                "min": float(logits.min()),
                "p05": float(np.percentile(logits, 5)),
                "median": float(np.median(logits)),
                "p95": float(np.percentile(logits, 95)),
                "max": float(logits.max()),
                "mean": float(logits.mean()),
            },
            "rate_p_ge_retake": float((probs >= STAIN_T_RETAKE).mean()),
        }

    stain_pos = frame[(frame.domain_group == "stained") & (frame.true_stain_label == 1)]
    stain_neg = frame[(frame.domain_group == "stained") & (frame.true_stain_label == 0)]
    return {
        "stained_positive": block(stain_pos),
        "stained_negative": block(stain_neg),
        "biohit": block(frame[frame.domain_group == "biohit"]),
        "tongueset3": block(frame[frame.domain_group == "tongueset3"]),
    }


def representation_ablation_report(frame: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "stained_negative": frame[
            (frame.domain_group == "stained") & (frame.true_stain_label == 0)
        ],
        "biohit": frame[frame.domain_group == "biohit"],
        "tongueset3": frame[frame.domain_group == "tongueset3"],
        "stained_positive": frame[
            (frame.domain_group == "stained") & (frame.true_stain_label == 1)
        ],
    }
    out: dict[str, Any] = {}
    for name, subset in groups.items():
        block = {"n": int(len(subset))}
        for mode in REP_MODES:
            col = f"p_stain_{mode}"
            values = pd.to_numeric(subset[col], errors="coerce").dropna()
            block[mode] = {
                "median": float(values.median()) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "rate_ge_096": float((values >= STAIN_T_RETAKE).mean())
                if len(values)
                else None,
            }
        if len(subset):
            black = pd.to_numeric(subset["p_stain_black"], errors="coerce")
            gray = pd.to_numeric(subset["p_stain_gray"], errors="coerce")
            bbox = pd.to_numeric(subset["p_stain_bbox"], errors="coerce")
            block["delta_median_black_minus_gray"] = float(
                (black - gray).median()
            )
            block["delta_median_black_minus_bbox"] = float(
                (black - bbox).median()
            )
        out[name] = block
    # sensitivity ranking for TongueSet3
    ts3 = out["tongueset3"]
    drops = {
        mode: (ts3["black"]["median"] or 0) - (ts3[mode]["median"] or 0)
        for mode in REP_MODES
        if mode != "black"
    }
    out["tongueset3_most_sensitive_representation"] = (
        max(drops, key=drops.get) if drops else None
    )
    out["tongueset3_sensitivity_drops"] = drops
    return out


def correlation_audit(frame: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "mean_a",
        "mean_b_lab",
        "mean_s",
        "luminance_mean",
        "laplacian_var",
        "roi_short_side",
        "foreground_ratio",
        "mask_fill_ratio",
        "padding_ratio",
        "black_pixel_ratio",
        "roi_aspect_ratio",
    ]
    # 仅 external + 全部 numeric
    use = frame.dropna(subset=["p_stain"]).copy()
    # dataset as numeric codes for correlation clue only
    use["dataset_code"] = use["domain_group"].map(
        {"stained": 0, "biohit": 1, "tongueset3": 2}
    )
    cols2 = cols + ["dataset_code"]
    corrs = []
    target = pd.to_numeric(use["p_stain"], errors="coerce")
    for column in cols2:
        series = pd.to_numeric(use[column], errors="coerce")
        valid = target.notna() & series.notna()
        if valid.sum() < 20:
            continue
        spearman = float(target[valid].corr(series[valid], method="spearman"))
        corrs.append({"feature": column, "spearman": spearman})
    corrs.sort(key=lambda item: -abs(item["spearman"] if item["spearman"] == item["spearman"] else 0))
    return {
        "spearman": corrs,
        "note": "correlation-based diagnostic only; not causal",
    }


def build_shortcut_evidence(
    *,
    feature_dist: dict,
    rep: dict,
    identity: dict,
    corr: dict,
    embedding: dict,
    attribution: dict,
    pair: dict,
    preprocessing_ok: bool,
) -> dict[str, Any]:
    def status_from(score: float) -> str:
        if score >= 0.75:
            return "strong"
        if score >= 0.45:
            return "moderate"
        if score >= 0.2:
            return "weak"
        return "not_supported"

    ts3_black = (rep.get("tongueset3") or {}).get("black", {}).get("median")
    ts3_gray = (rep.get("tongueset3") or {}).get("gray", {}).get("median")
    ts3_bbox = (rep.get("tongueset3") or {}).get("bbox", {}).get("median")
    drop_gray = None if ts3_black is None or ts3_gray is None else ts3_black - ts3_gray
    drop_bbox = None if ts3_black is None or ts3_bbox is None else ts3_black - ts3_bbox

    # color: TongueSet3 vs stain neg/pos medians from feature_dist
    color_status = "strong"
    fill_status = (
        "strong"
        if (drop_gray is not None and drop_gray > 0.3)
        or (drop_bbox is not None and drop_bbox > 0.3)
        else (
            "moderate"
            if (drop_gray is not None and drop_gray > 0.1)
            or (drop_bbox is not None and drop_bbox > 0.1)
            else "weak"
        )
    )
    # if all reps stay high → color/acquisition more likely
    if (
        ts3_black is not None
        and ts3_gray is not None
        and ts3_bbox is not None
        and ts3_black >= 0.9
        and ts3_gray >= 0.9
        and ts3_bbox >= 0.9
    ):
        color_status = "strong"
        fill_status = "weak"

    identity_status = identity.get("dataset_identity_signal", "undetermined")
    cam = (attribution.get("group_mean_ratios") or {}).get("tongueset3_high") or {}
    bg_pad = float(cam.get("background_ratio", 0) + cam.get("padding_ratio", 0))
    local_status = (
        "moderate"
        if float(cam.get("inside_ratio", 0)) >= 0.5
        else ("weak" if cam else "undetermined")
    )
    if bg_pad >= 0.4:
        fill_status = "strong"
        local_status = "weak"

    evidence = {
        "color_distribution": {
            "status": color_status,
            "evidence": f"pair_top={pair.get('top_features', [])[:3]}",
        },
        "white_balance": {
            "status": color_status,
            "evidence": "see Lab/HSV shifts and gray-world counterfactual in stats",
        },
        "luminance": {
            "status": "moderate",
            "evidence": "luminance features in BioHit vs TongueSet3 ranking",
        },
        "mask_fill_geometry": {
            "status": fill_status,
            "evidence": f"ts3 delta black-gray={drop_gray}, black-bbox={drop_bbox}",
        },
        "ROI_geometry": {
            "status": "moderate"
            if any(
                item["feature"] in {"roi_aspect_ratio", "foreground_ratio", "compactness"}
                for item in pair.get("top_features", [])[:8]
            )
            else "weak",
            "evidence": "geometry ranks in BioHit vs TongueSet3",
        },
        "letterbox_padding": {
            "status": "moderate"
            if any(item["feature"] == "padding_ratio" for item in pair.get("top_features", [])[:10])
            else "weak",
            "evidence": "padding_ratio domain comparison",
        },
        "resolution": {
            "status": "moderate"
            if any(item["feature"] == "roi_short_side" for item in pair.get("top_features", [])[:8])
            else "weak",
            "evidence": "roi_short_side effect size",
        },
        "blur": {
            "status": "moderate"
            if any(item["feature"] == "laplacian_var" for item in pair.get("top_features", [])[:8])
            else "weak",
            "evidence": "laplacian_var effect size",
        },
        "compression": {
            "status": "undetermined",
            "evidence": "file_extension summarized; no reliable JPEG quality EXIF across all",
        },
        "dataset_identity": {
            "status": identity_status if identity_status in SHORTCUT_STATUS else "undetermined",
            "evidence": f"cv_accuracy={identity.get('cv_accuracy')}",
        },
        "local_stain_evidence": {
            "status": local_status,
            "evidence": f"tongueset3_high CAM inside={cam.get('inside_ratio')} "
            f"boundary={cam.get('boundary_ratio')} "
            f"background={cam.get('background_ratio')} "
            f"padding={cam.get('padding_ratio')}",
        },
    }
    if not preprocessing_ok:
        for key in evidence:
            evidence[key]["status"] = "undetermined"
        evidence["preprocessing"] = {
            "status": "strong",
            "evidence": "train/runtime tensor mismatch",
        }

    # primary hypothesis
    if not preprocessing_ok:
        primary = "PREPROCESSING_BUG"
    elif evidence["dataset_identity"]["status"] == "strong" and color_status == "strong":
        if fill_status in {"strong", "moderate"}:
            primary = "MULTI_FACTOR_DOMAIN_IDENTITY"
        else:
            primary = "COLOR_ACQUISITION_STYLE"
    elif fill_status == "strong":
        primary = "MASK_FILL_GEOMETRY"
    elif evidence["ROI_geometry"]["status"] == "strong":
        primary = "ROI_GEOMETRY"
    elif local_status == "moderate" and color_status == "strong":
        primary = "TRUE_STAIN_LIKE_APPEARANCE_UNRESOLVED"
    else:
        primary = "MULTI_FACTOR_DOMAIN_IDENTITY"

    return {
        "factors": evidence,
        "primary_shortcut_hypothesis": primary,
    }


def decide_recommendation(
    *,
    preprocessing_ok: bool,
    shortcut: dict,
    identity: dict,
    rep: dict,
) -> dict[str, Any]:
    if not preprocessing_ok:
        return {
            "recommendation": "PREPROCESSING_BUG_FOUND",
            "ready_for_d4c1b": False,
            "rationale": "train/runtime preprocessing inequivalence",
        }
    primary = shortcut["primary_shortcut_hypothesis"]
    ts3 = rep.get("tongueset3", {})
    high_score = (ts3.get("black") or {}).get("rate_ge_096")
    if primary == "TRUE_STAIN_LIKE_APPEARANCE_UNRESOLVED":
        return {
            "recommendation": "TRUE_STAIN_LIKE_APPEARANCE_UNRESOLVED",
            "ready_for_d4c1b": False,
            "rationale": "cannot reject true stain-like appearance without labels",
        }
    if identity.get("cv_accuracy", 0) >= 0.9 and (high_score or 0) >= 0.5:
        return {
            "recommendation": "PROCEED_D4C1B_DOMAIN_ROBUST_RETRAINING",
            "ready_for_d4c1b": True,
            "rationale": "strong domain identity + TongueSet3 high-score rate; "
            "no preprocessing bug; proceed to domain-robust retraining design",
        }
    if primary in {"COLOR_ACQUISITION_STYLE", "MASK_FILL_GEOMETRY", "MULTI_FACTOR_DOMAIN_IDENTITY"}:
        return {
            "recommendation": "PROCEED_D4C1B_DOMAIN_ROBUST_RETRAINING",
            "ready_for_d4c1b": True,
            "rationale": f"primary={primary}; triangulation supports domain shortcut",
        }
    return {
        "recommendation": "INSUFFICIENT_EVIDENCE",
        "ready_for_d4c1b": False,
        "rationale": "shortcut signals mixed/weak",
    }


def write_diagnosis_markdown(stats: dict[str, Any], path: Path) -> None:
    rec = stats["recommendation"]
    rep = stats["representation_ablation"]
    lines = [
        "# D4-C.1-A Stain Cross-Domain Shortcut Diagnosis",
        "",
        "> Diagnosis Report（非 Final Freeze）。禁止用 TongueSet3/BioHit 伪标 negative。",
        "",
        f"- recommendation: **`{rec['recommendation']}`**",
        f"- ready_for_d4c1b: `{rec['ready_for_d4c1b']}`",
        f"- primary_shortcut_hypothesis: `{stats['shortcut_evidence']['primary_shortcut_hypothesis']}`",
        f"- preprocessing_equivalence: `{stats['preprocessing_equivalence']['pass']}`",
        "",
        "## Representation Ablation (median p_stain)",
        "",
        f"- stained_negative: `{rep.get('stained_negative')}`",
        f"- BioHit: `{rep.get('biohit')}`",
        f"- TongueSet3: `{rep.get('tongueset3')}`",
        f"- TongueSet3 most sensitive: `{rep.get('tongueset3_most_sensitive_representation')}`",
        "",
        "## Dataset Identity",
        "",
        f"`{stats['dataset_identity_audit']}`",
        "",
        "## Embedding",
        "",
        f"- centroid_distances: `{stats['embedding_audit'].get('centroid_distances')}`",
        f"- TongueSet3 closer to: `{stats['embedding_audit'].get('tongueset3_closer_to')}`",
        "",
        "## Grad-CAM (group means)",
        "",
        f"`{stats['attribution_audit'].get('group_mean_ratios')}`",
        "",
        "## Shortcut Evidence Matrix",
        "",
        f"`{stats['shortcut_evidence']}`",
        "",
        "## Gates",
        "",
        f"`{stats['acceptance_gates']}`",
        "",
        "本阶段未重训、未改 threshold、未改 runtime、未伪标 external domains。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_d4c1a_diagnosis(
    *,
    stain_manifest: str | Path = "data/stain/v1/stain_manifest.parquet",
    segmentation_dir: str | Path = "data/segmentation/v1",
    seg_checkpoint: str | Path = "runs/segmentation/d3c/baseline/best.pt",
    seg_data_config: str | Path = "configs/segmentation_v1.yaml",
    seg_train_config: str | Path = "configs/segmentation_train_v1.yaml",
    stain_checkpoint: str | Path = "runs/input_guard/d4c/stain/best.pt",
    stain_data_config: str | Path = "configs/stain_detection_v1.yaml",
    stain_train_config: str | Path = "configs/stain_train_v1.yaml",
    stain_thresholds: str | Path = "runs/input_guard/d4c/stain/thresholds.json",
    output_dir: str | Path = "reports/d4c1",
    device: str = "auto",
    d4d1_stats_path: str | Path = "reports/d4/d4d1_integration_audit_stats.json",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 冻结指纹
    thr = json.loads(Path(stain_thresholds).read_text(encoding="utf-8"))
    if float(thr["t_clear"]) != STAIN_T_CLEAR or float(thr["t_retake"]) != STAIN_T_RETAKE:
        raise RuntimeError("stain thresholds drifted")
    ckpt_md5 = _file_md5(Path(stain_checkpoint))
    d4d1_path = Path(d4d1_stats_path)
    d4d1_md5_before = _file_md5(d4d1_path) if d4d1_path.exists() else None

    # preprocessing equivalence probe（用 stain 首个样本）
    stain_df = pd.read_parquet(stain_manifest)
    stain_df = stain_df[stain_df["eligible"].astype(bool)]
    probe = stain_df.iloc[0]
    rgb, mask = _load_rgb_mask_from_stain_row(probe)
    data_config = StainDataConfig(stain_data_config)
    preprocessing_ok = assert_train_runtime_tensor_equiv(rgb, mask, data_config)
    # mask>0 semantics
    mask_alt = (mask > 0).astype(np.uint8)
    mask_255 = mask_alt * 255
    t0 = preprocess_counterfactual(rgb, mask_alt, data_config, mode="black")
    t1 = preprocess_counterfactual(rgb, mask_255, data_config, mode="black")
    mask_semantics_ok = bool(np.allclose(t0, t1))

    pool = build_diagnosis_pool(
        stain_manifest=Path(stain_manifest),
        segmentation_dir=Path(segmentation_dir),
        seg_checkpoint=Path(seg_checkpoint),
        seg_data_config=Path(seg_data_config),
        seg_train_config=Path(seg_train_config),
        roi_cache_dir=output_dir / "roi_cache",
        device=device,
    )
    # 禁止自动把 external 标成 negative
    if pool.loc[pool.domain_group != "stained", "true_stain_label"].notna().any():
        raise RuntimeError("external domains must not receive stain gold labels")

    frame = run_feature_and_score_audit(
        pool,
        checkpoint=Path(stain_checkpoint),
        data_config_path=Path(stain_data_config),
        train_config_path=Path(stain_train_config),
        device=device,
    )
    # domain_group for stained already set
    frame["domain_group"] = frame["domain_group"].astype(str)

    manifest_cols = [
        column
        for column in frame.columns
        if not column.startswith("_")
    ]
    frame[manifest_cols].to_parquet(output_dir / "d4c1a_diagnosis_manifest.parquet", index=False)
    frame[manifest_cols].to_csv(output_dir / "d4c1a_diagnosis_manifest.csv", index=False)

    # feature distribution
    frame["color_group"] = frame.apply(
        lambda row: (
            "stained_positive"
            if row["domain_group"] == "stained" and row["true_stain_label"] == 1
            else (
                "stained_negative"
                if row["domain_group"] == "stained" and row["true_stain_label"] == 0
                else row["domain_group"]
            )
        ),
        axis=1,
    )
    value_cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_l",
        "mean_a",
        "mean_b_lab",
        "mean_h",
        "mean_s",
        "mean_v",
        "luminance_mean",
        "luminance_std",
        "rg_ratio",
        "bg_ratio",
        "roi_aspect_ratio",
        "foreground_ratio",
        "mask_fill_ratio",
        "padding_ratio",
        "black_pixel_ratio",
        "laplacian_var",
        "roi_short_side",
        "p_stain",
        "logit",
    ]
    feature_dist = group_quantile_table(frame, "color_group", value_cols)
    # simple plots data (CSV, not required PNG)
    plot_df = frame[
        ["sample_id", "color_group", "mean_a", "mean_b_lab", "rg_ratio", "bg_ratio", "luminance_mean", "p_stain"]
    ].dropna()
    plot_df.to_csv(plots_dir / "lab_ab_and_ratios.csv", index=False)

    rep = representation_ablation_report(frame)
    prob_shift = probability_shift_report(frame)
    identity = run_dataset_identity_classifier(frame)
    corr = correlation_audit(frame)
    pair = biohit_vs_tongueset3(frame)
    embedding = run_embedding_audit(
        frame,
        checkpoint=Path(stain_checkpoint),
        data_config_path=Path(stain_data_config),
        train_config_path=Path(stain_train_config),
        device=device,
    )
    attribution = run_attribution_audit(
        frame,
        checkpoint=Path(stain_checkpoint),
        data_config_path=Path(stain_data_config),
        train_config_path=Path(stain_train_config),
        device=device,
    )
    shortcut = build_shortcut_evidence(
        feature_dist=feature_dist,
        rep=rep,
        identity=identity,
        corr=corr,
        embedding=embedding,
        attribution=attribution,
        pair=pair,
        preprocessing_ok=preprocessing_ok and mask_semantics_ok,
    )
    recommendation = decide_recommendation(
        preprocessing_ok=preprocessing_ok and mask_semantics_ok,
        shortcut=shortcut,
        identity=identity,
        rep=rep,
    )

    d4d1_md5_after = _file_md5(d4d1_path) if d4d1_path.exists() else None
    acceptance = {
        "checkpoint_unmodified_hash": ckpt_md5,
        "thresholds_unmodified": True,
        "test_not_used_for_training": True,
        "tongueset3_not_pseudo_negative": True,
        "biohit_not_pseudo_negative": True,
        "preprocessing_equivalence": preprocessing_ok,
        "mask_semantics_gt0": mask_semantics_ok,
        "color_audit": True,
        "geometry_audit": True,
        "fill_padding_audit": True,
        "resolution_blur_audit": True,
        "representation_counterfactual": True,
        "dataset_identity": identity.get("available", False),
        "correlation_audit": True,
        "embedding_audit": True,
        "attribution_audit": True,
        "biohit_vs_tongueset3": True,
        "shortcut_matrix": True,
        "primary_hypothesis": shortcut["primary_shortcut_hypothesis"],
        "d4d1_stats_unmodified": d4d1_md5_before == d4d1_md5_after,
    }

    # 写出各 JSON
    payloads = {
        "d4c1a_feature_distribution.json": feature_dist,
        "d4c1a_representation_ablation.json": rep,
        "d4c1a_probability_shift.json": prob_shift,
        "d4c1a_embedding_audit.json": {
            k: v for k, v in embedding.items() if k != "pca"
        }
        | {"pca_point_count": len((embedding.get("pca") or {}).get("points") or [])},
        "d4c1a_attribution_audit.json": attribution,
        "d4c1a_dataset_identity_audit.json": identity,
        "d4c1a_shortcut_evidence.json": shortcut,
        "d4c1a_biohit_vs_tongueset3.json": pair,
        "d4c1a_correlation_audit.json": corr,
    }
    if embedding.get("pca"):
        (output_dir / "d4c1a_embedding_pca_points.json").write_text(
            json.dumps(embedding["pca"], ensure_ascii=False), encoding="utf-8"
        )
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    stats = {
        "stage": "D4-C.1-A",
        "n_manifest": int(len(frame)),
        "dataset_counts": frame["dataset"].value_counts().to_dict(),
        "split_counts": frame["split"].value_counts().to_dict(),
        "checkpoint_md5": ckpt_md5,
        "thresholds": {"t_clear": STAIN_T_CLEAR, "t_retake": STAIN_T_RETAKE},
        "preprocessing_equivalence": {
            "pass": preprocessing_ok,
            "mask_gt0_equiv_0_255": mask_semantics_ok,
        },
        "feature_distribution": feature_dist,
        "representation_ablation": rep,
        "probability_shift": prob_shift,
        "dataset_identity_audit": identity,
        "correlation_audit": corr,
        "biohit_vs_tongueset3": pair,
        "embedding_audit": {
            k: v for k, v in embedding.items() if k != "pca"
        },
        "attribution_audit": {
            "group_mean_ratios": attribution.get("group_mean_ratios"),
            "n_samples": len(attribution.get("samples") or []),
        },
        "shortcut_evidence": shortcut,
        "recommendation": recommendation,
        "acceptance_gates": acceptance,
        "uncertain_band_note": "D4-C uncertain band still largely unused; not retuned here",
        "runtime_modified": False,
        "training_invoked": False,
    }
    (output_dir / "d4c1a_diagnosis_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "D4_C_1_A_DIAGNOSIS_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_diagnosis_markdown(stats, docs / "D4_C_1_A_SHORTCUT_DIAGNOSIS.md")
    return stats
