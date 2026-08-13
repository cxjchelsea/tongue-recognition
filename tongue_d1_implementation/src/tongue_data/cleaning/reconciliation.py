"""Duplicate 样本的 label / spatial 合并与冲突报告。"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .policy import CleaningPolicy

TIER_RANK = {
    "gold_candidate": 3,
    "silver": 2,
    "pseudo": 1,
    "weak": 0,
}


def _norm_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text
    return str(value)


def _norm_value(value: Any) -> int:
    return int(float(value))


def _label_sort_key(row: pd.Series) -> tuple:
    tier = TIER_RANK.get(str(row.get("supervision_tier")), -1)
    return (
        -tier,
        str(row.get("source_dataset", "")),
        str(row.get("source_field", "")),
        str(row.get("label_source", "")),
        str(row.get("source_label", "")),
        str(row.get("origin_sample_id", "")),
    )


def reconcile_labels(
    labels: pd.DataFrame,
    decisions: pd.DataFrame,
    policy: CleaningPolicy,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """将 alias 监督合并到 canonical；冲突不静默吞掉。

    conflict_policy=drop_conflicted_facts_from_clean：
    - 冲突 fact 写入 conflict report
    - 冲突 fact 不进入 labels_clean
    - 同 sample 的非冲突互补监督保留
    """
    conflict_policy = policy.conflict_policy()
    if conflict_policy != "drop_conflicted_facts_from_clean":
        raise ValueError(f"unsupported conflict_policy={conflict_policy!r}")

    if labels is None or labels.empty:
        empty = labels.copy() if labels is not None else pd.DataFrame()
        if "origin_sample_id" not in empty.columns:
            empty["origin_sample_id"] = []
        return empty, [], {"identical": 0, "complementary": 0, "conflicting": 0}

    sample_to_canonical = dict(
        zip(decisions["sample_id"].astype(str), decisions["canonical_sample_id"].astype(str))
    )
    sample_to_group = dict(
        zip(decisions["sample_id"].astype(str), decisions["duplicate_group_id"].astype(str))
    )

    work = labels.copy()
    work["origin_sample_id"] = work["sample_id"].astype(str)
    work["canonical_sample_id"] = work["origin_sample_id"].map(sample_to_canonical)
    work["duplicate_group_id"] = work["origin_sample_id"].map(sample_to_group)
    work = work[work["canonical_sample_id"].notna()].copy()
    work["label_key"] = work["canonical_label"].map(_norm_label)
    work["value_norm"] = work["value"].map(_norm_value)

    clean_rows = []
    conflicts = []
    stats = {"identical": 0, "complementary": 0, "conflicting": 0}

    for canonical_id, group in work.groupby("canonical_sample_id", sort=True):
        conflicted_task_labels = set()

        # 跨 origin 的 (task,label) 值冲突检测
        for (task, label_key), sub in group.groupby(
            ["canonical_task", "label_key"], sort=True
        ):
            values = sorted(set(int(v) for v in sub["value_norm"].tolist()))
            if len(values) > 1:
                conflicted_task_labels.add((str(task), str(label_key)))
                conflicts.append(
                    {
                        "duplicate_group_id": str(sub.iloc[0]["duplicate_group_id"]),
                        "canonical_sample_id": str(canonical_id),
                        "canonical_task": str(task),
                        "canonical_label": str(label_key),
                        "observed_values": values,
                        "source_sample_ids": sorted(set(sub["origin_sample_id"].astype(str))),
                        "source_datasets": sorted(set(sub["source_dataset"].astype(str))),
                        "conflict_type": "value_mismatch",
                    }
                )
                stats["conflicting"] += 1

        # binary true/false 互斥（同一 task）
        for task, sub in group.groupby("canonical_task", sort=True):
            label_set = set(sub["label_key"].tolist())
            if "true" in label_set and "false" in label_set:
                origins_true = set(
                    sub.loc[sub["label_key"] == "true", "origin_sample_id"].astype(str)
                )
                origins_false = set(
                    sub.loc[sub["label_key"] == "false", "origin_sample_id"].astype(str)
                )
                if origins_true != origins_false or len(origins_true | origins_false) > 1:
                    conflicted_task_labels.add((str(task), "true"))
                    conflicted_task_labels.add((str(task), "false"))
                    conflicts.append(
                        {
                            "duplicate_group_id": str(sub.iloc[0]["duplicate_group_id"]),
                            "canonical_sample_id": str(canonical_id),
                            "canonical_task": str(task),
                            "canonical_label": "true|false",
                            "observed_values": [1],
                            "source_sample_ids": sorted(origins_true | origins_false),
                            "source_datasets": sorted(set(sub["source_dataset"].astype(str))),
                            "conflict_type": "binary_true_false",
                        }
                    )
                    stats["conflicting"] += 1

        # 只丢弃冲突 fact；互补监督继续保留
        kept = group[
            ~group.apply(
                lambda row: (str(row["canonical_task"]), str(row["label_key"]))
                in conflicted_task_labels,
                axis=1,
            )
        ].copy()

        dedupe_keys = [
            "canonical_task",
            "label_key",
            "value_norm",
            "label_source",
            "source_field",
            "supervision_tier",
        ]
        if kept.empty:
            continue

        for _, sub in kept.groupby(dedupe_keys, sort=True, dropna=False):
            ordered = sorted((row for _, row in sub.iterrows()), key=_label_sort_key)
            representative = ordered[0].to_dict()
            origin_ids = sorted({str(r["origin_sample_id"]) for _, r in sub.iterrows()})
            if len(origin_ids) > 1:
                stats["identical"] += 1
            representative["sample_id"] = str(canonical_id)
            representative["origin_sample_id"] = (
                origin_ids[0] if len(origin_ids) == 1 else "|".join(origin_ids)
            )
            representative.pop("canonical_sample_id", None)
            representative.pop("duplicate_group_id", None)
            representative.pop("label_key", None)
            representative.pop("value_norm", None)
            clean_rows.append(representative)

        fact_count = kept[["canonical_task", "label_key", "value_norm"]].drop_duplicates().shape[0]
        if fact_count > 1:
            stats["complementary"] += fact_count - 1

    clean = pd.DataFrame(clean_rows)
    if clean.empty:
        cols = list(labels.columns) + ["origin_sample_id"]
        clean = pd.DataFrame(columns=list(dict.fromkeys(cols)))
    return clean, conflicts, stats


def reconcile_spatial(
    spatial: pd.DataFrame,
    decisions: pd.DataFrame,
    policy: CleaningPolicy | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """合并 spatial。

    - identical geometry → dedup，合并 provenance
    - 同 task/label 不同 geometry → 合法 multi-instance，全部保留，**不是 conflict**
    - review_groups：第一版无可靠自动判定，固定为 0
    """
    if policy is not None:
        geometry_policy = policy.spatial_different_geometry_policy()
        if geometry_policy != "multi_instance_keep":
            raise ValueError(
                f"unsupported spatial_different_geometry_policy={geometry_policy!r}"
            )

    empty_meta = {
        "multi_instance_groups": [],
        "review_groups": [],
    }
    empty_stats = {
        "identical_deduped": 0,
        "distinct_annotations_kept": 0,
        "multi_instance_groups": 0,
        "multi_instance_annotations": 0,
        "review_groups": 0,
    }

    if spatial is None or spatial.empty:
        empty = spatial.copy() if spatial is not None else pd.DataFrame()
        if "origin_sample_id" not in empty.columns:
            empty["origin_sample_id"] = []
        return empty, empty_meta, empty_stats

    sample_to_canonical = dict(
        zip(decisions["sample_id"].astype(str), decisions["canonical_sample_id"].astype(str))
    )
    work = spatial.copy()
    work["origin_sample_id"] = work["sample_id"].astype(str)
    work["sample_id"] = work["origin_sample_id"].map(sample_to_canonical)
    work = work[work["sample_id"].notna()].copy()

    def geom_key(row: pd.Series) -> tuple:
        return (
            str(row.get("annotation_task")),
            _norm_label(row.get("canonical_label")),
            str(row.get("annotation_type")),
            None if pd.isna(row.get("x_min")) else round(float(row["x_min"]), 6),
            None if pd.isna(row.get("y_min")) else round(float(row["y_min"]), 6),
            None if pd.isna(row.get("x_max")) else round(float(row["x_max"]), 6),
            None if pd.isna(row.get("y_max")) else round(float(row["y_max"]), 6),
            None if pd.isna(row.get("mask_path")) else str(row.get("mask_path")),
        )

    clean_rows = []
    multi_instance_groups = []
    stats = {
        "identical_deduped": 0,
        "distinct_annotations_kept": 0,
        "multi_instance_groups": 0,
        "multi_instance_annotations": 0,
        "review_groups": 0,
    }

    for canonical_id, group in work.groupby("sample_id", sort=True):
        seen_geom = {}
        by_task_label: dict[tuple[str, str], list[tuple]] = {}
        for _, row in group.iterrows():
            key = geom_key(row)
            task_label = (str(row.get("annotation_task")), _norm_label(row.get("canonical_label")))
            by_task_label.setdefault(task_label, []).append(key)
            if key in seen_geom:
                stats["identical_deduped"] += 1
                prev = seen_geom[key]
                origins = sorted(
                    {
                        *str(prev["origin_sample_id"]).split("|"),
                        str(row["origin_sample_id"]),
                    }
                )
                prev["origin_sample_id"] = "|".join([x for x in origins if x])
                continue
            out = row.to_dict()
            out["sample_id"] = str(canonical_id)
            out["annotation_id"] = f"{canonical_id}::{key[0]}::{key[1]}::{len(seen_geom)}"
            seen_geom[key] = out
            stats["distinct_annotations_kept"] += 1

        for task_label, keys in by_task_label.items():
            unique_keys = {k for k in keys}
            # 同 task/label 多个不同 geometry = 合法多实例检测，不是 conflict
            if len(unique_keys) > 1 and task_label[0]:
                annotation_count = len(unique_keys)
                multi_instance_groups.append(
                    {
                        "canonical_sample_id": str(canonical_id),
                        "annotation_task": task_label[0],
                        "canonical_label": task_label[1],
                        "distinct_geometries": annotation_count,
                        "semantics": "multi_instance",
                        "note": "different geometry under same task/label is legitimate multi-instance, not conflict",
                    }
                )
                stats["multi_instance_groups"] += 1
                stats["multi_instance_annotations"] += annotation_count

        clean_rows.extend(seen_geom.values())

    clean = pd.DataFrame(clean_rows)
    if clean.empty:
        cols = list(spatial.columns) + ["origin_sample_id"]
        clean = pd.DataFrame(columns=list(dict.fromkeys(cols)))

    meta = {
        "multi_instance_groups": multi_instance_groups,
        "review_groups": [],  # 第一版无可靠自动判定
    }
    return clean, meta, stats
