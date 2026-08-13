"""Group-level task vector 与分层特征。"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .policy import SplitPolicy


def _norm_label(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text
    return str(value)


def build_excluded_fact_keys(assignments: pd.DataFrame, exclude_pools: set[str]) -> set[tuple]:
    """从 supervision_assignments 提取不得主导分层的 fact 键。"""
    excluded: set[tuple] = set()
    if assignments is None or assignments.empty:
        return excluded
    pool_series = assignments["supervision_pool"].astype(str)
    bad = assignments[pool_series.isin(exclude_pools)]
    for sample_id, task, label in zip(
        bad["sample_id"].astype(str),
        bad["canonical_task"].astype(str),
        bad["canonical_label"].map(_norm_label),
    ):
        excluded.add((sample_id, task, label))
    return excluded


def build_group_task_vectors(
    sample_groups: pd.DataFrame,
    labels_clean: pd.DataFrame,
    assignments: pd.DataFrame,
    policy: SplitPolicy,
) -> dict[str, dict[str, int]]:
    """为每个 split_group 构建 supervised fact 计数（含 explicit negative）。"""
    exclude_pools = policy.exclude_pools()
    core_tasks = set(policy.core_tasks())
    excluded_keys = build_excluded_fact_keys(assignments, exclude_pools)

    sample_to_group = dict(
        zip(sample_groups["sample_id"].astype(str), sample_groups["split_group_id"].astype(str))
    )
    vectors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if labels_clean is None or labels_clean.empty:
        return {group_id: {} for group_id in sample_groups["split_group_id"].astype(str).unique()}

    work = labels_clean.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["canonical_task"] = work["canonical_task"].astype(str)
    work["canonical_label"] = work["canonical_label"].map(_norm_label)
    if core_tasks:
        work = work[work["canonical_task"].isin(core_tasks)]
    work["split_group_id"] = work["sample_id"].map(sample_to_group)
    work = work[work["split_group_id"].notna()]
    if excluded_keys:
        keep_mask = [
            (sample_id, task, label) not in excluded_keys
            for sample_id, task, label in zip(
                work["sample_id"], work["canonical_task"], work["canonical_label"]
            )
        ]
        work = work.loc[keep_mask]

    for sample_id, group_id, task, label, value in zip(
        work["sample_id"],
        work["split_group_id"].astype(str),
        work["canonical_task"],
        work["canonical_label"],
        work["value"].astype(float),
    ):
        if label in {"true", "false"}:
            fact_key = f"{task}:{label}"
        else:
            polarity = "positive" if int(value) == 1 else "negative"
            fact_key = f"{task}:{label}:{polarity}"
        vectors[group_id][fact_key] += 1

    for group_id in sample_groups["split_group_id"].astype(str).unique():
        vectors.setdefault(group_id, {})
    return {group_id: dict(vector) for group_id, vector in vectors.items()}
