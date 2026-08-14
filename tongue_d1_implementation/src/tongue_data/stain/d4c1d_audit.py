"""D4-C.1-D：Stained dataset confounding & label validity audit（无 CNN 训练）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp, wasserstein_distance

from .d4c1d_features import (
    COLOR_FEATURES,
    FORBIDDEN_CLASSIFIER_COLS,
    GEOMETRY_FEATURES,
    LOCAL_FEATURES,
    QUALITY_FEATURES,
    RESOLUTION_FEATURES,
    all_acquisition_features,
    extract_sample_features,
    summarize_feature,
)


SEED = 20260814
CALIPER = 1.5  # standardized Euclidean caliper
MATCH_COVARIATES = [
    "luminance_mean",
    "Lab_a_mean",
    "Lab_b_mean",
    "ROI_short_side",
    "ROI_aspect_ratio",
    "foreground_ratio",
    "original_pixel_count",
    "blur_laplacian",
]

VALID_LEVELS = {"NONE", "LOW", "MODERATE", "STRONG", "SEVERE"}
VALID_ACTIONS = {
    "REBALANCE_EXISTING_DATA",
    "MATCH_AND_RETRAIN",
    "SUPPLEMENT_MISSING_ACQUISITION_CELLS",
    "RECOLLECT_STAIN_DATASET",
    "LABEL_REVIEW_REQUIRED",
    "INSUFFICIENT_EVIDENCE",
}
VALID_EVIDENCE = {"strong", "moderate", "weak", "not_supported", "unavailable"}


def _md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_source_confounding_manifest(
    *,
    stain_manifest: str | Path = "data/stain/v1/stain_manifest.parquet",
    output_path: str | Path = "reports/d4c1d/source_confounding_manifest.parquet",
    splits: tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    """仅 TRAIN+VAL；默认不读 source TEST 图像特征。"""
    frame = pd.read_parquet(stain_manifest)
    frame = frame[
        (frame["eligible"] == True)
        & (frame["split"].isin(list(splits)))
        & frame["roi_rgb_path"].notna()
    ].copy()
    if frame["label"].isna().any():
        raise ValueError("stain gold label required")
    # 禁止 coating.color
    if "coating.color" in frame.columns or "coating_color" in frame.columns:
        raise ValueError("must not use coating.color as stain label")

    rows = []
    for _index, row in frame.sort_values("sample_id").iterrows():
        rows.append(
            extract_sample_features(
                sample_id=str(row["sample_id"]),
                split=str(row["split"]),
                stain_label=int(row["label"]),
                source_image_path=str(row["source_image_path"]),
                md5=str(row["md5"]),
                roi_rgb_path=str(row["roi_rgb_path"]),
                roi_mask_path=str(row["roi_mask_path"]),
                original_width=float(row["width"]) if pd.notna(row.get("width")) else None,
                original_height=float(row["height"]) if pd.notna(row.get("height")) else None,
                foreground_ratio_manifest=(
                    float(row["foreground_ratio"])
                    if pd.notna(row.get("foreground_ratio"))
                    else None
                ),
            )
        )
    out = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
        / max(len(pos) + len(neg) - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def robust_median_diff(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    mad = np.median(np.abs(np.concatenate([pos, neg]) - np.median(np.concatenate([pos, neg]))))
    if mad < 1e-12:
        return 0.0
    return float((np.median(pos) - np.median(neg)) / (1.4826 * mad))


def univariate_feature_effects(manifest: pd.DataFrame) -> dict[str, Any]:
    features = all_acquisition_features() + LOCAL_FEATURES
    pos = manifest[manifest["stain_label"] == 1]
    neg = manifest[manifest["stain_label"] == 0]
    rows = []
    for name in features:
        if name not in manifest.columns:
            continue
        pos_vals = pd.to_numeric(pos[name], errors="coerce").to_numpy()
        neg_vals = pd.to_numeric(neg[name], errors="coerce").to_numpy()
        pos_finite = pos_vals[np.isfinite(pos_vals)]
        neg_finite = neg_vals[np.isfinite(neg_vals)]
        if len(pos_finite) < 5 or len(neg_finite) < 5:
            continue
        ks_stat, ks_p = ks_2samp(pos_finite, neg_finite)
        try:
            w_dist = float(wasserstein_distance(pos_finite, neg_finite))
        except Exception:
            w_dist = float("nan")
        rows.append(
            {
                "feature": name,
                "positive": summarize_feature(pos_finite),
                "negative": summarize_feature(neg_finite),
                "cohens_d": cohens_d(pos_finite, neg_finite),
                "robust_median_diff": robust_median_diff(pos_finite, neg_finite),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "wasserstein": w_dist,
                "abs_cohens_d": abs(cohens_d(pos_finite, neg_finite)),
            }
        )
    rows.sort(key=lambda item: item["abs_cohens_d"], reverse=True)
    return {
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "effects": rows,
        "top10": [
            {
                "feature": item["feature"],
                "cohens_d": item["cohens_d"],
                "ks_statistic": item["ks_statistic"],
                "wasserstein": item["wasserstein"],
            }
            for item in rows[:10]
        ],
        "note": "association only; not causal",
    }


def _matrix_from_features(
    frame: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    usable = [name for name in feature_names if name in frame.columns]
    forbidden_hit = [name for name in usable if name in FORBIDDEN_CLASSIFIER_COLS]
    if forbidden_hit:
        raise RuntimeError(f"forbidden columns in classifier: {forbidden_hit}")
    matrix = frame[usable].apply(pd.to_numeric, errors="coerce")
    # 填中位数，保持可复现
    for column in usable:
        median = float(matrix[column].median())
        matrix[column] = matrix[column].fillna(median)
    labels = frame["stain_label"].astype(int).to_numpy()
    return matrix.to_numpy(dtype=np.float64), labels, usable


def run_diagnostic_classifier(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    model_name: str = "logistic",
    n_splits: int = 5,
) -> dict[str, Any]:
    """Stratified CV；禁止 sample_id/path/CNN embedding。"""
    matrix, labels, usable = _matrix_from_features(frame, feature_names)
    if len(np.unique(labels)) < 2:
        raise ValueError("need both classes")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    # 无 fold overlap 检查在 tests 中做；此处保证 shuffle+seed
    if model_name == "logistic":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=SEED,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    elif model_name == "random_forest":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=SEED,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(model_name)

    # 检查 fold 无重叠
    for train_idx, test_idx in cv.split(matrix, labels):
        if set(train_idx).intersection(test_idx):
            raise RuntimeError("CV fold overlap detected")

    probs = cross_val_predict(
        model, matrix, labels, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    preds = (probs >= 0.5).astype(int)
    auroc = float(roc_auc_score(labels, probs))
    prauc = float(average_precision_score(labels, probs))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds))
    cm = confusion_matrix(labels, preds).tolist()

    # 全量拟合做 importance（仅诊断）
    model.fit(matrix, labels)
    importance = {}
    if model_name == "logistic":
        coefs = model.named_steps["clf"].coef_.ravel()
        importance = {
            name: float(coef)
            for name, coef in sorted(
                zip(usable, coefs), key=lambda item: abs(item[1]), reverse=True
            )
        }
    else:
        importances = model.named_steps["clf"].feature_importances_
        importance = {
            name: float(value)
            for name, value in sorted(
                zip(usable, importances), key=lambda item: item[1], reverse=True
            )
        }

    gate = (
        "STRONG_CONFOUNDING_SIGNAL"
        if auroc >= 0.90
        else ("MODERATE_CONFOUNDING_SIGNAL" if auroc >= 0.80 else "WEAK_OR_UNRESOLVED")
    )
    return {
        "model": model_name,
        "n_samples": int(len(labels)),
        "n_features": len(usable),
        "features": usable,
        "auroc": auroc,
        "pr_auc": prauc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "confusion_matrix": cm,
        "feature_importance": importance,
        "top_features": list(importance.keys())[:15],
        "oof_probabilities": probs.tolist(),
        "gate": gate,
        "seed": SEED,
        "n_splits": n_splits,
    }


def propensity_overlap_audit(
    frame: pd.DataFrame, oof_probs: np.ndarray
) -> dict[str, Any]:
    labels = frame["stain_label"].astype(int).to_numpy()
    pos = oof_probs[labels == 1]
    neg = oof_probs[labels == 0]
    pos_min, pos_max = float(pos.min()), float(pos.max())
    neg_min, neg_max = float(neg.min()), float(neg.max())
    overlap_low = max(pos_min, neg_min)
    overlap_high = min(pos_max, neg_max)
    has_overlap = overlap_high > overlap_low
    if has_overlap:
        in_support = (oof_probs >= overlap_low) & (oof_probs <= overlap_high)
    else:
        in_support = np.zeros(len(oof_probs), dtype=bool)
    return {
        "positive_propensity": {
            "mean": float(pos.mean()),
            "median": float(np.median(pos)),
            "p05": float(np.percentile(pos, 5)),
            "p95": float(np.percentile(pos, 95)),
        },
        "negative_propensity": {
            "mean": float(neg.mean()),
            "median": float(np.median(neg)),
            "p05": float(np.percentile(neg, 5)),
            "p95": float(np.percentile(neg, 95)),
        },
        "overlap_range": [overlap_low, overlap_high] if has_overlap else None,
        "samples_in_common_support": int(in_support.sum()),
        "samples_outside_common_support": int((~in_support).sum()),
        "positive_support_rate": float(in_support[labels == 1].mean()),
        "negative_support_rate": float(in_support[labels == 0].mean()),
        "common_support_rate": float(in_support.mean()),
    }


def nearest_neighbor_matching(
    frame: pd.DataFrame,
    *,
    covariates: list[str] | None = None,
    caliper: float = CALIPER,
) -> dict[str, Any]:
    """positive→negative nearest neighbor；standardized Euclidean + caliper。"""
    covariates = covariates or MATCH_COVARIATES
    usable = [name for name in covariates if name in frame.columns]
    # 硬禁 stain prediction
    assert "p_stain" not in usable
    assert "stain_pred" not in usable

    data = frame.copy()
    matrix = data[usable].apply(pd.to_numeric, errors="coerce")
    for column in usable:
        matrix[column] = matrix[column].fillna(float(matrix[column].median()))
    values = matrix.to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = (values - mean) / std

    pos_idx = np.where(data["stain_label"].to_numpy() == 1)[0]
    neg_idx = np.where(data["stain_label"].to_numpy() == 0)[0]
    pairs = []
    used_neg: set[int] = set()
    unmatched_pos = []

    for pos_i in pos_idx:
        distances = np.linalg.norm(scaled[neg_idx] - scaled[pos_i], axis=1)
        order = np.argsort(distances)
        matched = False
        for rank in order:
            neg_i = int(neg_idx[rank])
            dist = float(distances[rank])
            if dist > caliper:
                break
            if neg_i in used_neg:
                continue
            # per-feature standardized differences
            feat_diff = {
                name: float(scaled[pos_i, col] - scaled[neg_i, col])
                for col, name in enumerate(usable)
            }
            pairs.append(
                {
                    "positive_sample_id": data.iloc[pos_i]["sample_id"],
                    "negative_sample_id": data.iloc[neg_i]["sample_id"],
                    "matching_distance": dist,
                    "feature_std_diff": feat_diff,
                }
            )
            used_neg.add(neg_i)
            matched = True
            break
        if not matched:
            unmatched_pos.append(data.iloc[pos_i]["sample_id"])

    distances = [item["matching_distance"] for item in pairs]
    return {
        "covariates": usable,
        "caliper": caliper,
        "n_positive": int(len(pos_idx)),
        "n_negative": int(len(neg_idx)),
        "n_matched_pairs": int(len(pairs)),
        "positive_match_rate": float(len(pairs) / max(len(pos_idx), 1)),
        "negative_match_utilization": float(len(used_neg) / max(len(neg_idx), 1)),
        "median_match_distance": float(np.median(distances)) if distances else None,
        "p95_match_distance": float(np.percentile(distances, 95)) if distances else None,
        "unmatched_positive_count": int(len(unmatched_pos)),
        "pairs": pairs,
        "unmatched_positive_ids": unmatched_pos[:50],
    }


def stratification_by_bins(
    frame: pd.DataFrame, column: str, n_bins: int = 4
) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce")
    labels_q = pd.qcut(values, q=n_bins, duplicates="drop")
    rows = []
    for bin_name, subset in frame.groupby(labels_q, observed=True):
        rate = float(subset["stain_label"].mean())
        rows.append(
            {
                "bin": str(bin_name),
                "n": int(len(subset)),
                "positive_rate": rate,
                "n_positive": int((subset["stain_label"] == 1).sum()),
            }
        )
    rates = [item["positive_rate"] for item in rows]
    monotonic = False
    if len(rates) >= 3:
        diffs = np.diff(rates)
        monotonic = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))
    span = float(max(rates) - min(rates)) if rates else 0.0
    return {
        "feature": column,
        "bins": rows,
        "positive_rate_span": span,
        "approximately_monotonic": monotonic,
        "confounding_flag": bool(span >= 0.25 or (monotonic and span >= 0.15)),
    }


def metadata_confounding_audit(frame: pd.DataFrame) -> dict[str, Any]:
    def by_key(column: str) -> dict[str, Any]:
        if column not in frame.columns:
            return {"available": False}
        table = []
        for key, subset in frame.groupby(frame[column].astype(str)):
            table.append(
                {
                    "key": str(key),
                    "n": int(len(subset)),
                    "positive_rate": float(subset["stain_label"].mean()),
                    "n_positive": int((subset["stain_label"] == 1).sum()),
                    "n_negative": int((subset["stain_label"] == 0).sum()),
                }
            )
        table.sort(key=lambda item: item["n"], reverse=True)
        extreme = [
            item
            for item in table
            if item["n"] >= 20 and (item["positive_rate"] >= 0.95 or item["positive_rate"] <= 0.05)
        ]
        return {
            "available": True,
            "groups": table[:30],
            "extreme_groups": extreme,
            "batch_label_confounding": bool(len(extreme) > 0),
        }

    return {
        "folder_batch": by_key("folder_batch"),
        "file_extension": by_key("file_extension"),
        "exif_make": by_key("exif_make"),
        "exif_model": by_key("exif_model"),
    }


def near_duplicate_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """dHash 碰撞：同 hash 不同 label → inconsistency；同 label → collection clue。"""
    grouped = frame.groupby("dhash")
    cross_label = 0
    same_label_multi = 0
    examples = []
    for hash_value, subset in grouped:
        if len(subset) < 2:
            continue
        labels = set(subset["stain_label"].tolist())
        if len(labels) > 1:
            cross_label += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "dhash": hash_value,
                        "sample_ids": subset["sample_id"].tolist()[:6],
                        "labels": subset["stain_label"].tolist()[:6],
                        "type": "cross_label",
                    }
                )
        else:
            same_label_multi += 1
    return {
        "n_unique_dhash": int(frame["dhash"].nunique()),
        "cross_label_hash_groups": int(cross_label),
        "same_label_multi_hash_groups": int(same_label_multi),
        "near_duplicate_concern": bool(cross_label > 0 or same_label_multi > 50),
        "examples": examples,
        "action": "report_only_no_deletion",
    }


def build_review_candidates(
    frame: pd.DataFrame,
    propensity: np.ndarray,
    matching: dict[str, Any],
    *,
    per_group: int = 15,
) -> pd.DataFrame:
    data = frame.copy()
    data["acquisition_propensity"] = propensity
    pos = data[data.stain_label == 1].sort_values("sample_id")
    neg = data[data.stain_label == 0].sort_values("sample_id")
    rows = []

    def take(subset: pd.DataFrame, group_name: str, ascending: bool):
        ordered = subset.sort_values(
            ["acquisition_propensity", "sample_id"], ascending=[ascending, True]
        ).head(per_group)
        for _index, row in ordered.iterrows():
            rows.append(
                {
                    "review_group": group_name,
                    "sample_id": row["sample_id"],
                    "stain_label": int(row["stain_label"]),
                    "acquisition_propensity": float(row["acquisition_propensity"]),
                    "source_image_path": row["source_image_path"],
                    "split": row["split"],
                    "luminance_mean": row.get("luminance_mean"),
                    "ROI_short_side": row.get("ROI_short_side"),
                }
            )

    take(pos, "A_high_propensity_positive", ascending=False)
    take(pos, "B_low_propensity_positive", ascending=True)
    take(neg, "C_high_propensity_negative", ascending=False)
    take(neg, "D_low_propensity_negative", ascending=True)

    # E matched pairs
    for pair in matching.get("pairs", [])[:per_group]:
        rows.append(
            {
                "review_group": "E_matched_pair_positive",
                "sample_id": pair["positive_sample_id"],
                "stain_label": 1,
                "acquisition_propensity": None,
                "source_image_path": None,
                "split": None,
                "matched_with": pair["negative_sample_id"],
                "matching_distance": pair["matching_distance"],
            }
        )
        rows.append(
            {
                "review_group": "E_matched_pair_negative",
                "sample_id": pair["negative_sample_id"],
                "stain_label": 0,
                "acquisition_propensity": None,
                "source_image_path": None,
                "split": None,
                "matched_with": pair["positive_sample_id"],
                "matching_distance": pair["matching_distance"],
            }
        )
    # F unmatched
    for sample_id in matching.get("unmatched_positive_ids", [])[:per_group]:
        row = data[data.sample_id == sample_id].iloc[0]
        rows.append(
            {
                "review_group": "F_unmatched_positive",
                "sample_id": sample_id,
                "stain_label": 1,
                "acquisition_propensity": float(row["acquisition_propensity"]),
                "source_image_path": row["source_image_path"],
                "split": row["split"],
            }
        )
    return pd.DataFrame(rows)


def level_from_signal(
    *,
    auroc: float,
    common_support_rate: float,
    match_rate: float,
    top_effect: float,
    batch_confounding: bool,
) -> str:
    if auroc >= 0.95 and common_support_rate < 0.25 and match_rate < 0.35:
        return "SEVERE"
    if auroc >= 0.90 or (auroc >= 0.85 and common_support_rate < 0.4):
        return "STRONG"
    if auroc >= 0.80 or top_effect >= 1.0 or batch_confounding:
        return "MODERATE"
    if auroc >= 0.70 or top_effect >= 0.5:
        return "LOW"
    return "NONE"


def decide_recommendation(
    *,
    level: str,
    auroc: float,
    matched_auroc: float | None,
    match_rate: float,
    n_matched: int,
    common_support_rate: float,
    batch_confounding: bool,
    folder_label_perfect_split: bool = False,
) -> dict[str, Any]:
    confirmed = level in {"MODERATE", "STRONG", "SEVERE"}
    delta = None if matched_auroc is None else float(auroc - matched_auroc)

    # 采集 batch/目录与 label 完全绑定 + 匹配率极低 → 必须重采
    if folder_label_perfect_split and match_rate < 0.25 and auroc >= 0.90:
        action = "RECOLLECT_STAIN_DATASET"
        rescuable = "false"
        level = "SEVERE"
    elif auroc >= 0.95 and match_rate < 0.20:
        action = "RECOLLECT_STAIN_DATASET"
        rescuable = "false"
        level = "SEVERE" if level != "SEVERE" else level
    elif (
        n_matched >= 200
        and matched_auroc is not None
        and matched_auroc < 0.75
        and delta is not None
        and delta >= 0.15
    ):
        action = "MATCH_AND_RETRAIN"
        rescuable = "partial" if matched_auroc >= 0.65 else "true"
    elif common_support_rate >= 0.45 and match_rate >= 0.50 and n_matched >= 150:
        action = "REBALANCE_EXISTING_DATA"
        rescuable = "partial"
    elif batch_confounding and match_rate >= 0.30 and common_support_rate >= 0.3:
        action = "SUPPLEMENT_MISSING_ACQUISITION_CELLS"
        rescuable = "partial"
    elif level in {"STRONG", "SEVERE"}:
        action = "RECOLLECT_STAIN_DATASET"
        rescuable = "false"
    elif level == "MODERATE":
        action = "SUPPLEMENT_MISSING_ACQUISITION_CELLS"
        rescuable = "partial"
    else:
        action = "INSUFFICIENT_EVIDENCE"
        rescuable = "true"

    assert action in VALID_ACTIONS
    return {
        "SOURCE_CONFOUNDING_CONFIRMED": bool(confirmed or level == "SEVERE"),
        "SOURCE_CONFOUNDING_LEVEL": level,
        "EXISTING_DATA_RESCUABLE": rescuable,
        "RECOMMENDED_DATA_ACTION": action,
        "acquisition_auroc_before_match": auroc,
        "acquisition_auroc_after_match": matched_auroc,
        "auroc_delta": delta,
        "folder_label_perfect_split": bool(folder_label_perfect_split),
    }


def evidence_strength(flag: bool, *, strong_if: bool = False, moderate_if: bool = False) -> str:
    if strong_if:
        return "strong"
    if moderate_if:
        return "moderate"
    if flag:
        return "weak"
    return "not_supported"


def run_full_d4c1d_audit(
    *,
    stain_manifest: str | Path = "data/stain/v1/stain_manifest.parquet",
    reports_dir: str | Path = "reports/d4c1d",
    docs_dir: str | Path = "docs",
    rebuild_manifest: bool = True,
) -> dict[str, Any]:
    reports_dir = Path(reports_dir)
    docs_dir = Path(docs_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 冻结文件未被本阶段修改的断言（hash 记录）
    frozen_paths = {
        "v1_ckpt": Path("runs/input_guard/d4c/stain/best.pt"),
        "v1_thr": Path("runs/input_guard/d4c/stain/thresholds.json"),
        "v2_ckpt": Path("runs/input_guard/d4c1b/stain_v2/best.pt"),
        "policy": Path("configs/input_guard_v1.yaml"),
    }
    frozen_hashes_before = {
        key: _md5_file(path) if path.exists() else None
        for key, path in frozen_paths.items()
    }

    manifest_path = reports_dir / "source_confounding_manifest.parquet"
    if rebuild_manifest or not manifest_path.exists():
        manifest = build_source_confounding_manifest(
            stain_manifest=stain_manifest, output_path=manifest_path
        )
    else:
        manifest = pd.read_parquet(manifest_path)

    # 确认无 test
    if (manifest["split"] == "test").any():
        raise RuntimeError("source TEST must not enter default audit")

    effects = univariate_feature_effects(manifest)
    (reports_dir / "feature_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    family = {
        "color_only": run_diagnostic_classifier(manifest, COLOR_FEATURES, model_name="logistic"),
        "resolution_only": run_diagnostic_classifier(
            manifest, RESOLUTION_FEATURES, model_name="logistic"
        ),
        "geometry_only": run_diagnostic_classifier(
            manifest, GEOMETRY_FEATURES, model_name="logistic"
        ),
        "quality_only": run_diagnostic_classifier(
            manifest, QUALITY_FEATURES, model_name="logistic"
        ),
    }
    # 去掉超大 oof 列表写盘时可保留
    family_slim = {}
    for key, value in family.items():
        slim = dict(value)
        slim.pop("oof_probabilities", None)
        family_slim[key] = slim
    (reports_dir / "feature_family_classifiers.json").write_text(
        json.dumps(family_slim, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_acq = run_diagnostic_classifier(
        manifest, all_acquisition_features(), model_name="logistic"
    )
    rf = run_diagnostic_classifier(
        manifest, all_acquisition_features(), model_name="random_forest"
    )
    all_slim = dict(all_acq)
    probs = np.asarray(all_slim.pop("oof_probabilities"), dtype=np.float64)
    rf_slim = dict(rf)
    rf_slim.pop("oof_probabilities", None)
    (reports_dir / "acquisition_classifier.json").write_text(
        json.dumps(
            {"logistic": all_slim, "random_forest": rf_slim},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    propensity = propensity_overlap_audit(manifest, probs)
    (reports_dir / "propensity_overlap.json").write_text(
        json.dumps(propensity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matching = nearest_neighbor_matching(manifest)
    pairs_frame = pd.DataFrame(
        [
            {
                "positive_sample_id": item["positive_sample_id"],
                "negative_sample_id": item["negative_sample_id"],
                "matching_distance": item["matching_distance"],
                **{f"diff_{key}": value for key, value in item["feature_std_diff"].items()},
            }
            for item in matching["pairs"]
        ]
    )
    pairs_frame.to_csv(reports_dir / "matched_pairs.csv", index=False)
    matching_slim = {key: value for key, value in matching.items() if key != "pairs"}
    matching_slim["n_pairs_written"] = int(len(pairs_frame))
    (reports_dir / "matching_audit.json").write_text(
        json.dumps(matching_slim, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # matched subset
    matched_ids = set()
    for item in matching["pairs"]:
        matched_ids.add(item["positive_sample_id"])
        matched_ids.add(item["negative_sample_id"])
    matched_subset = manifest[manifest["sample_id"].isin(matched_ids)].copy()
    matched_clf = None
    if len(matched_subset) >= 40 and matched_subset["stain_label"].nunique() == 2:
        matched_clf = run_diagnostic_classifier(
            matched_subset, all_acquisition_features(), model_name="logistic"
        )
        matched_slim = dict(matched_clf)
        matched_slim.pop("oof_probabilities", None)
    else:
        matched_slim = {"skipped": True, "reason": "matched subset too small"}
    matched_audit = {
        "n_matched_samples": int(len(matched_subset)),
        "n_matched_pairs": matching["n_matched_pairs"],
        "before_auroc": all_acq["auroc"],
        "after_auroc": matched_slim.get("auroc"),
        "auroc_delta": (
            None
            if matched_slim.get("auroc") is None
            else float(all_acq["auroc"] - matched_slim["auroc"])
        ),
        "matched_classifier": matched_slim,
    }
    (reports_dir / "matched_subset_audit.json").write_text(
        json.dumps(matched_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    res_strat = stratification_by_bins(manifest, "original_pixel_count")
    lum_strat = stratification_by_bins(manifest, "luminance_mean")
    lab_a_strat = stratification_by_bins(manifest, "Lab_a_mean")
    (reports_dir / "resolution_stratification.json").write_text(
        json.dumps(res_strat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "color_stratification.json").write_text(
        json.dumps(
            {"luminance": lum_strat, "Lab_a": lab_a_strat},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = metadata_confounding_audit(manifest)
    (reports_dir / "metadata_confounding.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    near = near_duplicate_audit(manifest)
    (reports_dir / "near_duplicate_audit.json").write_text(
        json.dumps(near, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    local_clf = run_diagnostic_classifier(manifest, LOCAL_FEATURES, model_name="logistic")
    local_slim = dict(local_clf)
    local_slim.pop("oof_probabilities", None)
    global_vs_local = {
        "global_acquisition_auroc": all_acq["auroc"],
        "local_heterogeneity_auroc": local_slim["auroc"],
        "local_gain": float(local_slim["auroc"] - all_acq["auroc"]),
        "interpretation": (
            "global_dominates"
            if all_acq["auroc"] - local_slim["auroc"] >= 0.05
            else (
                "local_comparable_or_stronger"
                if local_slim["auroc"] >= all_acq["auroc"]
                else "mixed"
            )
        ),
        "local_classifier": local_slim,
    }
    (reports_dir / "global_vs_local.json").write_text(
        json.dumps(global_vs_local, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    review = build_review_candidates(manifest, probs, matching)
    # 回填 path
    id_to_path = dict(zip(manifest["sample_id"], manifest["source_image_path"]))
    if "source_image_path" in review.columns:
        review["source_image_path"] = review.apply(
            lambda row: row["source_image_path"]
            if pd.notna(row.get("source_image_path"))
            else id_to_path.get(row["sample_id"]),
            axis=1,
        )
    review.to_csv(reports_dir / "label_review_candidates.csv", index=False)

    top_effect = float(effects["top10"][0]["cohens_d"]) if effects["top10"] else 0.0
    batch_flag = bool(meta.get("folder_batch", {}).get("batch_label_confounding"))
    folder_groups = meta.get("folder_batch", {}).get("extreme_groups", [])
    folder_label_perfect_split = bool(
        len(folder_groups) >= 2
        and any(item.get("positive_rate", 0) >= 0.999 for item in folder_groups)
        and any(item.get("positive_rate", 1) <= 0.001 for item in folder_groups)
    )
    level = level_from_signal(
        auroc=all_acq["auroc"],
        common_support_rate=propensity["common_support_rate"],
        match_rate=matching["positive_match_rate"],
        top_effect=abs(top_effect),
        batch_confounding=batch_flag,
    )
    decision = decide_recommendation(
        level=level,
        auroc=all_acq["auroc"],
        matched_auroc=matched_audit.get("after_auroc"),
        match_rate=matching["positive_match_rate"],
        n_matched=matching["n_matched_pairs"],
        common_support_rate=propensity["common_support_rate"],
        batch_confounding=batch_flag,
        folder_label_perfect_split=folder_label_perfect_split,
    )
    level = decision["SOURCE_CONFOUNDING_LEVEL"]

    def matrix_item(strength: str, summary: str) -> dict[str, str]:
        assert strength in VALID_EVIDENCE
        return {"strength": strength, "evidence_summary": summary}

    evidence_matrix = {
        "color_label_confounding": matrix_item(
            "strong"
            if family["color_only"]["auroc"] >= 0.90
            else ("moderate" if family["color_only"]["auroc"] >= 0.80 else "weak"),
            f"color-only AUROC={family['color_only']['auroc']:.3f}",
        ),
        "luminance_label_confounding": matrix_item(
            "strong" if lum_strat["confounding_flag"] else "moderate",
            f"luminance bin positive_rate_span={lum_strat['positive_rate_span']:.3f}",
        ),
        "resolution_label_confounding": matrix_item(
            "strong"
            if family["resolution_only"]["auroc"] >= 0.90 or res_strat["confounding_flag"]
            else ("moderate" if family["resolution_only"]["auroc"] >= 0.80 else "weak"),
            f"resolution-only AUROC={family['resolution_only']['auroc']:.3f}; "
            f"span={res_strat['positive_rate_span']:.3f}",
        ),
        "blur_label_confounding": matrix_item(
            "moderate" if family["quality_only"]["auroc"] >= 0.80 else "weak",
            f"quality-only AUROC={family['quality_only']['auroc']:.3f}",
        ),
        "geometry_label_confounding": matrix_item(
            "moderate" if family["geometry_only"]["auroc"] >= 0.80 else "weak",
            f"geometry-only AUROC={family['geometry_only']['auroc']:.3f}",
        ),
        "batch_label_confounding": matrix_item(
            "strong" if batch_flag else "not_supported",
            f"extreme folder batches={len(meta.get('folder_batch', {}).get('extreme_groups', []))}",
        ),
        "format_label_confounding": matrix_item(
            "moderate"
            if meta.get("file_extension", {}).get("batch_label_confounding")
            else "weak",
            "file_extension group rates",
        ),
        "camera_label_confounding": matrix_item(
            "unavailable"
            if not any(
                pd.notna(manifest.get(col)).any()
                for col in ("exif_make", "exif_model")
                if col in manifest.columns
            )
            else (
                "moderate"
                if meta.get("exif_model", {}).get("batch_label_confounding")
                else "weak"
            ),
            "EXIF make/model audit",
        ),
        "near_duplicate_concern": matrix_item(
            "moderate" if near["near_duplicate_concern"] else "not_supported",
            f"cross_label_hash_groups={near['cross_label_hash_groups']}",
        ),
        "common_support": matrix_item(
            "strong"
            if propensity["common_support_rate"] < 0.3
            else ("moderate" if propensity["common_support_rate"] < 0.5 else "weak"),
            f"common_support_rate={propensity['common_support_rate']:.3f}",
        ),
        "matchability": matrix_item(
            "strong"
            if matching["positive_match_rate"] < 0.4
            else ("moderate" if matching["positive_match_rate"] < 0.7 else "weak"),
            f"positive_match_rate={matching['positive_match_rate']:.3f}; "
            f"pairs={matching['n_matched_pairs']}",
        ),
        "local_signal_support": matrix_item(
            "not_supported"
            if global_vs_local["interpretation"] == "global_dominates"
            else "weak",
            f"local AUROC={local_slim['auroc']:.3f} vs global={all_acq['auroc']:.3f}",
        ),
    }
    (reports_dir / "confounding_evidence_matrix.json").write_text(
        json.dumps(evidence_matrix, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    frozen_hashes_after = {
        key: _md5_file(path) if path.exists() else None
        for key, path in frozen_paths.items()
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen artifacts modified during D4-C.1-D audit")

    stats = {
        "stage": "D4-C.1-D",
        "n_audit_samples": int(len(manifest)),
        "n_positive": int((manifest.stain_label == 1).sum()),
        "n_negative": int((manifest.stain_label == 0).sum()),
        "splits": sorted(manifest["split"].unique().tolist()),
        "top10_effects": effects["top10"],
        "family_aurocs": {key: value["auroc"] for key, value in family_slim.items()},
        "all_acquisition": {
            "auroc": all_acq["auroc"],
            "pr_auc": all_acq["pr_auc"],
            "gate": all_acq["gate"],
            "top_features": all_acq["top_features"][:10],
        },
        "propensity": propensity,
        "matching": matching_slim,
        "matched_subset_audit": {
            "n_matched_samples": matched_audit["n_matched_samples"],
            "before_auroc": matched_audit["before_auroc"],
            "after_auroc": matched_audit["after_auroc"],
            "auroc_delta": matched_audit["auroc_delta"],
        },
        "resolution_stratification": res_strat,
        "color_stratification": {"luminance": lum_strat, "Lab_a": lab_a_strat},
        "metadata": {
            "batch_label_confounding": batch_flag,
            "format_extreme": meta.get("file_extension", {}).get("batch_label_confounding"),
        },
        "near_duplicate": near,
        "global_vs_local": {
            "global_auroc": global_vs_local["global_acquisition_auroc"],
            "local_auroc": global_vs_local["local_heterogeneity_auroc"],
            "interpretation": global_vs_local["interpretation"],
        },
        "evidence_matrix": evidence_matrix,
        "decision": decision,
        "frozen_artifacts_unchanged": True,
        "cnn_training_performed": False,
        "policy_modified": False,
        "source_test_used": False,
        "known_external_audit_run": False,
    }
    (docs_dir / "D4_C_1_D_AUDIT_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_md = _render_report(stats, decision)
    (docs_dir / "D4_C_1_D_DATASET_CONFOUNDING_AUDIT.md").write_text(
        report_md, encoding="utf-8"
    )
    return stats


def _render_report(stats: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = [
        "# D4-C.1-D Dataset Confounding & Label Validity Audit",
        "",
        "本阶段为**数据诊断**，未训练 ResNet / 未改 threshold / 未改 policy。",
        "",
        "## Decision",
        "",
        f"- SOURCE_CONFOUNDING_CONFIRMED: `{decision['SOURCE_CONFOUNDING_CONFIRMED']}`",
        f"- SOURCE_CONFOUNDING_LEVEL: `{decision['SOURCE_CONFOUNDING_LEVEL']}`",
        f"- EXISTING_DATA_RESCUABLE: `{decision['EXISTING_DATA_RESCUABLE']}`",
        f"- RECOMMENDED_DATA_ACTION: `{decision['RECOMMENDED_DATA_ACTION']}`",
        "",
        "## Cohort",
        "",
        f"- n_audit (train+val): {stats['n_audit_samples']}",
        f"- positive: {stats['n_positive']}",
        f"- negative: {stats['n_negative']}",
        "",
        "## Acquisition-only classifiers",
        "",
        f"- color-only AUROC: {stats['family_aurocs']['color_only']:.4f}",
        f"- resolution-only AUROC: {stats['family_aurocs']['resolution_only']:.4f}",
        f"- geometry-only AUROC: {stats['family_aurocs']['geometry_only']:.4f}",
        f"- quality-only AUROC: {stats['family_aurocs']['quality_only']:.4f}",
        f"- all-acquisition AUROC/PR-AUC: "
        f"{stats['all_acquisition']['auroc']:.4f} / {stats['all_acquisition']['pr_auc']:.4f}",
        f"- gate: `{stats['all_acquisition']['gate']}`",
        "",
        "## Matching",
        "",
        f"- positive match rate: {stats['matching']['positive_match_rate']:.4f}",
        f"- matched pairs: {stats['matching']['n_matched_pairs']}",
        f"- median / p95 distance: "
        f"{stats['matching']['median_match_distance']} / {stats['matching']['p95_match_distance']}",
        f"- acquisition AUROC before→after: "
        f"{stats['matched_subset_audit']['before_auroc']} → "
        f"{stats['matched_subset_audit']['after_auroc']} "
        f"(Δ={stats['matched_subset_audit']['auroc_delta']})",
        "",
        "## Notes",
        "",
        "- Confounding = association, not causation.",
        "- Fixing confounding ≠ forcing all images to one white-balance.",
        "- Goal: statistically decouple acquisition condition from stain label.",
        "- STOP here; no v4 train / recollect / policy change without confirmation.",
        "",
    ]
    return "\n".join(lines)
