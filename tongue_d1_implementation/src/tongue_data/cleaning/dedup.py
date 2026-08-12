"""Duplicate group 与 canonical 选择（deterministic）。"""
from __future__ import annotations

import pandas as pd

from .policy import CleaningPolicy


def build_duplicate_groups(samples: pd.DataFrame, policy: CleaningPolicy) -> pd.DataFrame:
    """为每个 sample 计算 dataset 内 duplicate_group_id。"""
    frame = samples.copy()
    if frame.empty:
        frame["duplicate_group_id"] = []
        frame["is_duplicate"] = []
        return frame

    frame["duplicate_group_id"] = [
        policy.group_id(str(dataset_name), str(md5_value))
        for dataset_name, md5_value in zip(frame["dataset"], frame["md5"])
    ]
    group_sizes = frame.groupby("duplicate_group_id")["sample_id"].transform("count")
    frame["is_duplicate"] = group_sizes > 1
    if not policy.global_cfg.get("singleton_groups", True):
        frame.loc[~frame["is_duplicate"], "duplicate_group_id"] = None
    return frame


def select_canonical_samples(grouped_samples: pd.DataFrame, policy: CleaningPolicy) -> pd.DataFrame:
    """每个 duplicate group 选唯一 canonical；规则稳定可重复。"""
    if grouped_samples.empty:
        return pd.DataFrame(
            columns=[
                "sample_id", "dataset", "md5", "duplicate_group_id", "is_duplicate",
                "canonical_sample_id", "keep", "decision", "reason", "policy_version",
            ]
        )

    tie_breakers = policy.tie_breakers()
    decisions = []
    # 按 group id 排序保证遍历顺序稳定
    for group_id, group in grouped_samples.groupby("duplicate_group_id", sort=True):
        ordered = group.sort_values(by=tie_breakers, kind="mergesort")
        canonical_id = str(ordered.iloc[0]["sample_id"])
        for _, row in ordered.iterrows():
            sample_id = str(row["sample_id"])
            is_canonical = sample_id == canonical_id
            decisions.append(
                {
                    "sample_id": sample_id,
                    "dataset": row["dataset"],
                    "md5": row["md5"],
                    "duplicate_group_id": group_id,
                    "is_duplicate": bool(row["is_duplicate"]),
                    "canonical_sample_id": canonical_id,
                    "keep": is_canonical,
                    "decision": "canonical" if is_canonical else "duplicate_alias",
                    "reason": (
                        "stable_canonical_selection"
                        if is_canonical
                        else "same_md5_as_canonical"
                    ),
                    "policy_version": policy.version,
                }
            )
    return pd.DataFrame(decisions)


def find_cross_dataset_duplicates(samples: pd.DataFrame) -> list[dict]:
    """同一 md5 出现在多个 dataset → 需人工复核 / strict fail。"""
    if samples.empty:
        return []
    collisions = []
    for md5_value, group in samples.groupby("md5", sort=True):
        datasets = sorted({str(x) for x in group["dataset"].tolist()})
        if len(datasets) > 1:
            collisions.append(
                {
                    "md5": str(md5_value),
                    "datasets": datasets,
                    "sample_ids": sorted(group["sample_id"].astype(str).tolist()),
                }
            )
    return collisions
