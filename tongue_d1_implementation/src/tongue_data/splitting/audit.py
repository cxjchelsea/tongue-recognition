"""Group / split / task / leakage 审计报告。"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .policy import SplitPolicy
from .stratification import _norm_label


def enrich_group_audit(
    base_audit: dict,
    sample_groups: pd.DataFrame,
    split_groups: pd.DataFrame,
    samples_clean: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict:
    """生成完整 group_audit.json 内容。"""
    audit = dict(base_audit)
    per_dataset = {}
    for dataset_name, group in sample_groups.groupby("dataset", sort=True):
        group_ids = group["split_group_id"].astype(str)
        sizes = group_ids.value_counts()
        per_dataset[str(dataset_name)] = {
            "sample_count": int(len(group)),
            "group_count": int(group_ids.nunique()),
            "singleton_groups": int((sizes == 1).sum()),
            "multi_sample_groups": int((sizes > 1).sum()),
            "max_group_size": int(sizes.max()) if len(sizes) else 0,
            "mean_group_size": float(sizes.mean()) if len(sizes) else 0.0,
        }
    audit["per_dataset"] = per_dataset

    # TongueDx 专项
    tonguedx = sample_groups[sample_groups["dataset"].astype(str) == "tonguedx"]
    if len(tonguedx):
        patients = tonguedx[tonguedx["patient_id"].notna()]["patient_id"].astype(str)
        patient_counts = patients.value_counts()
        audit["TongueDx"] = {
            "patient_groups": int(patients.nunique()),
            "patients_with_multiple_images": int((patient_counts > 1).sum()),
            "max_images_per_patient": int(patient_counts.max()) if len(patient_counts) else 0,
            "missing_patient_ids": int(audit.get("tonguedx_missing_patient_id_count", 0)),
            "canonical_with_multiple_origin_patient_ids": len(
                audit.get("canonical_with_multiple_origin_patient_ids", [])
            ),
        }
    else:
        audit["TongueDx"] = {}

    # TonguExpert L1/L2
    tonguexpert_ids = set(
        sample_groups.loc[sample_groups["dataset"].astype(str) == "tonguexpert", "sample_id"].astype(str)
    )
    if assignments is not None and len(assignments) and tonguexpert_ids:
        te = assignments[assignments["sample_id"].astype(str).isin(tonguexpert_ids)].copy()
        # L1/L2 以 label_source 为准
        # TonguExpert：L1=human，L2=model_prediction
        with_l1 = set(te.loc[te["label_source"].astype(str) == "human", "sample_id"].astype(str))
        with_l2 = set(
            te.loc[te["label_source"].astype(str) == "model_prediction", "sample_id"].astype(str)
        )
        audit["TonguExpert"] = {
            "samples_with_L1": int(len(with_l1)),
            "samples_with_L2": int(len(with_l2)),
            "samples_with_both_L1_and_L2": int(len(with_l1 & with_l2)),
        }
    else:
        audit["TonguExpert"] = {
            "samples_with_L1": 0,
            "samples_with_L2": 0,
            "samples_with_both_L1_and_L2": 0,
        }

    holdout = sample_groups[sample_groups["forced_split"].astype(str) == "external_holdout"]
    audit["external_holdout"] = {"sample_count": int(len(holdout))}
    audit["total_canonical_samples"] = int(len(samples_clean))
    audit["total_leakage_groups"] = int(len(split_groups))
    return audit


def enrich_split_groups_with_supervision(
    split_groups: pd.DataFrame,
    sample_groups: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """为 group 表补充 supervision 标志。"""
    sample_to_group = dict(
        zip(sample_groups["sample_id"].astype(str), sample_groups["split_group_id"].astype(str))
    )
    flags = defaultdict(
        lambda: {
            "gold_candidate": False,
            "silver": False,
            "pseudo": False,
            "auxiliary": False,
        }
    )
    if assignments is not None and len(assignments):
        for sample_id, pool in zip(
            assignments["sample_id"].astype(str),
            assignments["supervision_pool"].astype(str),
        ):
            group_id = sample_to_group.get(sample_id)
            if group_id is None:
                continue
            if pool in flags[group_id]:
                flags[group_id][pool] = True

    result = split_groups.copy()
    result["contains_gold_candidate"] = (
        result["split_group_id"].astype(str).map(lambda gid: flags[gid]["gold_candidate"])
    )
    result["contains_silver"] = (
        result["split_group_id"].astype(str).map(lambda gid: flags[gid]["silver"])
    )
    result["contains_pseudo"] = (
        result["split_group_id"].astype(str).map(lambda gid: flags[gid]["pseudo"])
    )
    result["contains_auxiliary"] = (
        result["split_group_id"].astype(str).map(lambda gid: flags[gid]["auxiliary"])
    )
    return result


def build_split_report(
    split_assignments: pd.DataFrame,
    policy: SplitPolicy,
    unstratifiable: list | None = None,
) -> dict:
    """生成 split_report.json。"""
    counts = split_assignments["split"].astype(str).value_counts().to_dict()
    regular = split_assignments[split_assignments["split"].astype(str).isin(["train", "val", "test"])]
    regular_total = max(int(len(regular)), 1)
    ratios_actual = {
        name: float(len(regular[regular["split"] == name]) / regular_total)
        for name in ["train", "val", "test"]
    }
    group_split = (
        split_assignments.groupby("split_group_id")["split"]
        .first()
        .astype(str)
        .value_counts()
        .to_dict()
    )
    per_dataset = {}
    for dataset_name, group in split_assignments.groupby("dataset", sort=True):
        per_dataset[str(dataset_name)] = {
            split_name: int((group["split"].astype(str) == split_name).sum())
            for split_name in ["train", "val", "test", "external_holdout"]
        }

    tonguedx = split_assignments[split_assignments["dataset"].astype(str) == "tonguedx"]
    patient_split = {}
    if len(tonguedx):
        for split_name in ["train", "val", "test"]:
            subset = tonguedx[tonguedx["split"].astype(str) == split_name]
            patient_split[f"patients_{split_name}"] = int(
                subset.loc[subset["patient_id"].notna(), "patient_id"].astype(str).nunique()
            )

    dsct = split_assignments[split_assignments["dataset"].astype(str) == "dsct"]
    return {
        "policy_version": policy.version,
        "seed": policy.seed,
        "samples_total": int(len(split_assignments)),
        "train": int(counts.get("train", 0)),
        "val": int(counts.get("val", 0)),
        "test": int(counts.get("test", 0)),
        "external_holdout": int(counts.get("external_holdout", 0)),
        "ratios_actual": ratios_actual,
        "ratios_target": policy.target_ratios(),
        "groups_total": int(split_assignments["split_group_id"].nunique()),
        "groups_train": int(group_split.get("train", 0)),
        "groups_val": int(group_split.get("val", 0)),
        "groups_test": int(group_split.get("test", 0)),
        "per_dataset": per_dataset,
        "TongueDx": patient_split,
        "DSCT": {"external_holdout_count": int(len(dsct))},
        "unstratifiable_labels": unstratifiable or [],
    }


def build_task_distribution(
    split_assignments: pd.DataFrame,
    labels_clean: pd.DataFrame,
    split_supervision: pd.DataFrame,
    policy: SplitPolicy,
) -> dict:
    """按 task/label 统计 train/val/test 的 positive / explicit_negative。"""
    sample_split = dict(
        zip(split_assignments["sample_id"].astype(str), split_assignments["split"].astype(str))
    )
    # 仅统计非 pseudo 的有效监督（val/test 评估与 train 可用）
    allowed = set()
    if split_supervision is not None and len(split_supervision):
        usable = split_supervision[
            (split_supervision["effective_for_train"].astype(bool))
            | (split_supervision["effective_for_val"].astype(bool))
            | (split_supervision["effective_for_test"].astype(bool))
        ]
        for _, row in usable.iterrows():
            allowed.add(
                (
                    str(row["sample_id"]),
                    str(row["canonical_task"]),
                    _norm_label(row["canonical_label"]),
                )
            )

    core_tasks = set(policy.core_tasks())
    buckets = defaultdict(lambda: defaultdict(lambda: {"positive": 0, "explicit_negative": 0}))

    for _, row in labels_clean.iterrows():
        sample_id = str(row["sample_id"])
        split_name = sample_split.get(sample_id)
        if split_name not in {"train", "val", "test"}:
            continue
        task = str(row["canonical_task"])
        if core_tasks and task not in core_tasks:
            continue
        label = _norm_label(row["canonical_label"])
        if allowed and (sample_id, task, label) not in allowed:
            continue
        value = int(float(row["value"]))
        key = f"{task}::{label}"
        if label in {"true", "false"}:
            # binary：true → positive 侧，false → explicit_negative 侧（按 present=false）
            if label == "true":
                buckets[key][split_name]["positive"] += 1
                buckets[key]["overall"]["positive"] += 1
            else:
                buckets[key][split_name]["explicit_negative"] += 1
                buckets[key]["overall"]["explicit_negative"] += 1
        else:
            if value == 1:
                buckets[key][split_name]["positive"] += 1
                buckets[key]["overall"]["positive"] += 1
            else:
                buckets[key][split_name]["explicit_negative"] += 1
                buckets[key]["overall"]["explicit_negative"] += 1

    report = {}
    distribution_warnings = []
    for fact_key, split_map in sorted(buckets.items()):
        entry = {}
        prevalences = {}
        for split_name in ["overall", "train", "val", "test"]:
            stats = split_map.get(split_name, {"positive": 0, "explicit_negative": 0})
            positive = int(stats["positive"])
            negative = int(stats["explicit_negative"])
            total = positive + negative
            prevalence = float(positive / total) if total else None
            entry[split_name] = {
                "positive": positive,
                "explicit_negative": negative,
                "total_supervised": total,
                "prevalence": prevalence,
            }
            if split_name in {"train", "val", "test"} and prevalence is not None:
                prevalences[split_name] = prevalence
        if len(prevalences) >= 2:
            values = list(prevalences.values())
            max_dev = max(values) - min(values)
            entry["max_absolute_prevalence_deviation"] = float(max_dev)
            if max_dev > 0.15:
                distribution_warnings.append(
                    {"fact": fact_key, "max_absolute_prevalence_deviation": float(max_dev)}
                )
        else:
            entry["max_absolute_prevalence_deviation"] = None
        report[fact_key] = entry

    return {"tasks": report, "distribution_warnings": distribution_warnings}
