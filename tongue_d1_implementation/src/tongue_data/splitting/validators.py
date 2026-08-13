"""D2-B/C split 泄漏与契约校验。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .policy import SplitPolicy


def compute_leakage_counts(
    split_assignments: pd.DataFrame,
    split_supervision: pd.DataFrame | None = None,
    decisions: pd.DataFrame | None = None,
) -> dict:
    """计算六类泄漏计数（0 为合格）。"""
    counts = {
        "patient_leakage": 0,
        "md5_leakage": 0,
        "sample_leakage": 0,
        "group_leakage": 0,
        "pseudo_leakage": 0,
        "external_holdout_leakage": 0,
    }
    details = {key: [] for key in counts}

    regular = split_assignments[
        split_assignments["split"].astype(str).isin(["train", "val", "test"])
    ].copy()

    # 1) sample：同 sample 多 split
    for sample_id, group in split_assignments.groupby("sample_id"):
        splits = set(group["split"].astype(str))
        if len(splits) > 1:
            counts["sample_leakage"] += 1
            details["sample_leakage"].append({"sample_id": str(sample_id), "splits": sorted(splits)})

    # 2) group：同 group 多 split
    for group_id, group in split_assignments.groupby("split_group_id"):
        splits = set(group["split"].astype(str))
        if len(splits) > 1:
            counts["group_leakage"] += 1
            details["group_leakage"].append(
                {"split_group_id": str(group_id), "splits": sorted(splits)}
            )

    # 3) patient：TongueDx 同 patient 跨 regular split
    tonguedx = regular[regular["dataset"].astype(str) == "tonguedx"]
    tonguedx = tonguedx[tonguedx["patient_id"].notna()]
    for patient_id, group in tonguedx.groupby(tonguedx["patient_id"].astype(str)):
        splits = set(group["split"].astype(str))
        if len(splits) > 1:
            counts["patient_leakage"] += 1
            details["patient_leakage"].append(
                {"patient_id": str(patient_id), "splits": sorted(splits)}
            )

    # 4) MD5：同 MD5 跨 regular split
    for md5_value, group in regular.groupby(regular["md5"].astype(str)):
        splits = set(group["split"].astype(str))
        if len(splits) > 1:
            counts["md5_leakage"] += 1
            details["md5_leakage"].append({"md5": str(md5_value), "splits": sorted(splits)})

    # 5) duplicate alias → canonical 不得跨 split（以 decisions 映射）
    if decisions is not None and len(decisions):
        canon_split = dict(
            zip(
                split_assignments["sample_id"].astype(str),
                split_assignments["split"].astype(str),
            )
        )
        for group_id, group in decisions.groupby("duplicate_group_id"):
            canon_ids = set(group["canonical_sample_id"].astype(str))
            splits = {canon_split[cid] for cid in canon_ids if cid in canon_split}
            if len(splits) > 1:
                counts["md5_leakage"] += 1
                details["md5_leakage"].append(
                    {"duplicate_group_id": str(group_id), "splits": sorted(splits)}
                )

    # 6) pseudo leakage：非 train sample 仍 effective_for_train
    if split_supervision is not None and len(split_supervision):
        pseudo = split_supervision[split_supervision["supervision_pool"].astype(str) == "pseudo"]
        bad = pseudo[
            (pseudo["effective_for_train"].astype(bool))
            & (pseudo["sample_split"].astype(str) != "train")
        ]
        counts["pseudo_leakage"] = int(len(bad))
        if len(bad):
            details["pseudo_leakage"] = bad[["sample_id", "sample_split"]].head(50).to_dict("records")
        # val/test 不得用 pseudo 作评估监督
        eval_bad = pseudo[
            (pseudo["effective_for_val"].astype(bool)) | (pseudo["effective_for_test"].astype(bool))
        ]
        if len(eval_bad):
            counts["pseudo_leakage"] += int(len(eval_bad))

    # 7) DSCT / external_holdout isolation
    dsct = split_assignments[split_assignments["dataset"].astype(str) == "dsct"]
    if len(dsct):
        non_holdout = dsct[dsct["split"].astype(str) != "external_holdout"]
        counts["external_holdout_leakage"] = int(len(non_holdout))
        if len(non_holdout):
            details["external_holdout_leakage"] = (
                non_holdout[["sample_id", "split"]].head(50).to_dict("records")
            )

    return {"counts": counts, "details": details}


def validate_split(
    split_dir: str | Path,
    processed_dir: str | Path | None = None,
    policy_path: str | Path | None = None,
):
    """validate-split：硬失败任何 leakage > 0。"""
    root = Path(split_dir)
    errors, warnings = [], []
    required = [
        "split_groups.parquet",
        "sample_group_assignments.parquet",
        "split_assignments.parquet",
        "split_supervision_assignments.parquet",
    ]
    for name in required:
        if not (root / name).exists():
            errors.append(f"missing: {name}")
    if errors:
        return errors, warnings

    split_assignments = pd.read_parquet(root / "split_assignments.parquet")
    split_supervision = pd.read_parquet(root / "split_supervision_assignments.parquet")
    decisions = None
    if processed_dir is not None:
        decisions_path = Path(processed_dir) / "dedup_decisions.parquet"
        if decisions_path.exists():
            decisions = pd.read_parquet(decisions_path)

    if policy_path is not None:
        SplitPolicy(policy_path)  # 确保可读

    # 每个 sample 恰好一个 split
    if split_assignments["sample_id"].duplicated().any():
        errors.append("split_assignments: duplicate sample_id")

    allowed_splits = {"train", "val", "test", "external_holdout"}
    bad_split = set(split_assignments["split"].astype(str)) - allowed_splits
    if bad_split:
        errors.append(f"unknown split values: {sorted(bad_split)}")

    # Stained 不得进 coating.color 训练角色（guard）
    if len(split_supervision):
        stained = split_supervision[
            (split_supervision["dataset"].astype(str) == "stained_coating")
            & (split_supervision["canonical_task"].astype(str) == "coating.color")
            & (
                split_supervision["effective_for_train"].astype(bool)
                | split_supervision["effective_for_val"].astype(bool)
                | split_supervision["effective_for_test"].astype(bool)
            )
        ]
        if len(stained):
            errors.append("stained_coating must remain quality-only; coating.color effective>0")

    leakage = compute_leakage_counts(split_assignments, split_supervision, decisions)
    for key, value in leakage["counts"].items():
        if int(value) > 0:
            errors.append(f"{key}={value}")

    # 写出 leakage_report 到同级 reports 若存在旁路文件则由 builder 负责；此处仅校验
    meta_path = root / "split_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("leakage_counts"):
            for key, value in meta["leakage_counts"].items():
                if int(value) > 0 and f"{key}=" not in " ".join(errors):
                    errors.append(f"metadata {key}={value}")

    return errors, warnings
