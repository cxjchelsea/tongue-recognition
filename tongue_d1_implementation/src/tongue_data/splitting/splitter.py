"""Deterministic group-aware multilabel splitter。"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .policy import SplitPolicy


def _stable_hash(text: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}::{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_splits(
    split_groups: pd.DataFrame,
    sample_groups: pd.DataFrame,
    group_vectors: dict[str, dict[str, int]],
    policy: SplitPolicy,
) -> pd.DataFrame:
    """为每个 leakage group 分配 train/val/test/external_holdout。"""
    ratios = policy.target_ratios()
    seed = policy.seed
    split_names = ["train", "val", "test"]

    forced = split_groups[split_groups["forced_split"].notna()].copy()
    regular = split_groups[split_groups["forced_split"].isna()].copy()

    assignments = {}
    for _, row in forced.iterrows():
        assignments[str(row["split_group_id"])] = str(row["forced_split"])

    group_size = dict(
        zip(regular["split_group_id"].astype(str), regular["member_count"].astype(int))
    )
    group_datasets: dict[str, dict[str, int]] = {}
    for group_id, group in sample_groups.groupby("split_group_id"):
        if str(group_id) not in group_size:
            continue
        group_datasets[str(group_id)] = group["dataset"].astype(str).value_counts().to_dict()

    total_regular = int(sum(group_size.values())) or 1
    dataset_totals = defaultdict(float)
    for dataset_counts in group_datasets.values():
        for dataset_name, count in dataset_counts.items():
            dataset_totals[dataset_name] += count

    targets = {name: ratios.get(name, 0.0) * total_regular for name in split_names}
    current = {name: 0.0 for name in split_names}
    fact_totals = defaultdict(float)
    fact_split = {name: defaultdict(float) for name in split_names}
    dataset_split = {name: defaultdict(float) for name in split_names}

    for group_id, vector in group_vectors.items():
        if group_id not in group_size:
            continue
        for fact_key, count in vector.items():
            fact_totals[fact_key] += count

    group_fact_rarity = {}
    for group_id in regular["split_group_id"].astype(str):
        vector = group_vectors.get(group_id, {})
        if vector:
            rarity = min(fact_totals.get(fact_key, 0) for fact_key in vector)
        else:
            rarity = 10**9
        group_fact_rarity[group_id] = rarity

    ordered_groups = sorted(
        regular["split_group_id"].astype(str).tolist(),
        key=lambda group_id: (
            group_fact_rarity[group_id],
            -group_size.get(group_id, 0),
            _stable_hash(group_id, seed),
        ),
    )

    unstratifiable = []
    for fact_key, _total in sorted(fact_totals.items()):
        groups_with = [
            group_id
            for group_id in ordered_groups
            if group_vectors.get(group_id, {}).get(fact_key, 0) > 0
        ]
        if len(groups_with) < 3:
            unstratifiable.append({"fact": fact_key, "group_count": len(groups_with)})

    # 预计算稀有 fact 的 group 数
    fact_group_count = {
        fact_key: sum(
            1
            for group_id in ordered_groups
            if group_vectors.get(group_id, {}).get(fact_key, 0) > 0
        )
        for fact_key in fact_totals
    }

    for group_id in ordered_groups:
        size = group_size[group_id]
        vector = group_vectors.get(group_id, {})
        datasets = group_datasets.get(group_id, {})
        best_split = None
        best_score = None
        for split_name in split_names:
            hypothetical = dict(current)
            hypothetical[split_name] = hypothetical[split_name] + size
            ratio_penalty = sum(
                abs(hypothetical[name] / total_regular - ratios.get(name, 0.0))
                for name in split_names
            )

            task_penalty = 0.0
            for fact_key, count in vector.items():
                total = fact_totals.get(fact_key, 0) or 1
                for name in split_names:
                    predicted = fact_split[name][fact_key] + (count if name == split_name else 0)
                    desired = ratios.get(name, 0.0) * total
                    task_penalty += abs(predicted - desired) / total

            dataset_penalty = 0.0
            for dataset_name, count in datasets.items():
                total = dataset_totals.get(dataset_name, 0) or 1
                for name in split_names:
                    predicted = dataset_split[name][dataset_name] + (
                        count if name == split_name else 0
                    )
                    desired = ratios.get(name, 0.0) * total
                    dataset_penalty += abs(predicted - desired) / total

            rare_bonus = 0.0
            for fact_key in vector:
                groups_with = fact_group_count.get(fact_key, 0)
                if groups_with == 1 and split_name == "train":
                    rare_bonus -= 0.5
                if groups_with == 2 and split_name in {"train", "val"}:
                    rare_bonus -= 0.2

            score = (ratio_penalty * 2.0) + task_penalty + (dataset_penalty * 1.5) + rare_bonus
            tie = _stable_hash(f"{group_id}::{split_name}", seed)
            candidate = (score, tie, split_names.index(split_name))
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_split = split_name

        assignments[group_id] = best_split
        current[best_split] += size
        for fact_key, count in vector.items():
            fact_split[best_split][fact_key] += count
        for dataset_name, count in datasets.items():
            dataset_split[best_split][dataset_name] += count

    mapped = sample_groups.copy()
    mapped["split"] = mapped["split_group_id"].astype(str).map(assignments)
    mapped["split_reason"] = np.where(
        mapped["split"].astype(str) == "external_holdout",
        "forced_external_holdout",
        "deterministic_group_multilabel",
    )
    mapped["seed"] = seed
    mapped["policy_version"] = policy.version
    mapped.attrs["unstratifiable"] = unstratifiable
    mapped.attrs["targets"] = targets
    mapped.attrs["actual_regular_counts"] = current
    return mapped


def apply_effective_supervision(
    split_assignments: pd.DataFrame,
    supervision_assignments: pd.DataFrame,
    policy: SplitPolicy,
) -> pd.DataFrame:
    """基于 sample split 生成 effective eligibility（不覆盖 D2-A 原表）。"""
    sample_split = dict(
        zip(
            split_assignments["sample_id"].astype(str),
            split_assignments["split"].astype(str),
        )
    )
    result = supervision_assignments.copy()
    result["sample_split"] = result["sample_id"].astype(str).map(sample_split)
    pool = result["supervision_pool"].astype(str)
    split_name = result["sample_split"].astype(str)
    base_train = result["eligible_for_train"].astype(bool)
    base_val = result["eligible_for_val"].astype(bool)
    base_test = result["eligible_for_test"].astype(bool)

    effective_train = pd.Series(False, index=result.index)
    effective_val = pd.Series(False, index=result.index)
    effective_test = pd.Series(False, index=result.index)
    reason = pd.Series("split_aligned", index=result.index)

    holdout_mask = (split_name == "external_holdout") | (pool == "external_holdout")
    reason = reason.mask(holdout_mask, "external_holdout_only")

    pseudo_mask = (~holdout_mask) & (pool == "pseudo")
    effective_train = effective_train.mask(pseudo_mask & (split_name == "train") & base_train, True)
    reason = reason.mask(pseudo_mask & (split_name == "train") & base_train, "pseudo_train_only")
    reason = reason.mask(pseudo_mask & ~((split_name == "train") & base_train), "pseudo_blocked_by_split")

    train_mask = (~holdout_mask) & (~pseudo_mask) & (split_name == "train")
    effective_train = effective_train.mask(train_mask, base_train)

    val_mask = (~holdout_mask) & (~pseudo_mask) & (split_name == "val")
    effective_val = effective_val.mask(val_mask, base_val)
    reason = reason.mask(val_mask, "val_evaluation_supervision")

    test_mask = (~holdout_mask) & (~pseudo_mask) & (split_name == "test")
    effective_test = effective_test.mask(test_mask, base_test)
    reason = reason.mask(test_mask, "test_evaluation_supervision")

    result["effective_for_train"] = effective_train
    result["effective_for_val"] = effective_val
    result["effective_for_test"] = effective_test
    result["effective_reason"] = reason
    result["split_policy_version"] = policy.version
    return result


def attach_supervision_summary(
    split_assignments: pd.DataFrame,
    supervision_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """为每个 sample 汇总 supervision pools。"""
    result = split_assignments.copy()
    if supervision_assignments is None or supervision_assignments.empty:
        result["supervision_summary"] = "none"
        return result
    pooled = (
        supervision_assignments.assign(
            sample_id=supervision_assignments["sample_id"].astype(str),
            supervision_pool=supervision_assignments["supervision_pool"].astype(str),
        )
        .groupby("sample_id")["supervision_pool"]
        .agg(lambda pools: "|".join(sorted(set(pools))))
    )
    result["supervision_summary"] = result["sample_id"].astype(str).map(pooled).fillna("none")
    return result
