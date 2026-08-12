"""D2-A clean 产物校验。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_clean(processed_dir: str | Path, policy_path: str | Path | None = None):
    root = Path(processed_dir)
    errors, warnings = [], []
    required = [
        "samples_clean.parquet",
        "labels_clean.parquet",
        "spatial_clean.parquet",
        "dedup_decisions.parquet",
        "supervision_assignments.parquet",
    ]
    for name in required:
        if not (root / name).exists():
            errors.append(f"missing: {name}")
    if errors:
        return errors, warnings

    samples = pd.read_parquet(root / "samples_clean.parquet")
    labels = pd.read_parquet(root / "labels_clean.parquet")
    spatial = pd.read_parquet(root / "spatial_clean.parquet")
    decisions = pd.read_parquet(root / "dedup_decisions.parquet")
    assignments = pd.read_parquet(root / "supervision_assignments.parquet")

    if samples["sample_id"].duplicated().any():
        errors.append("samples_clean: duplicate sample_id")

    # 每个 duplicate group 恰好一个 canonical keep
    for group_id, group in decisions.groupby("duplicate_group_id"):
        keep_n = int(group["keep"].sum())
        if keep_n != 1:
            errors.append(f"group {group_id} keep count={keep_n}")
        canonical_ids = set(group["canonical_sample_id"].astype(str))
        if len(canonical_ids) != 1:
            errors.append(f"group {group_id} has multiple canonical ids")

    # alias 指向存在的 canonical
    clean_ids = set(samples["sample_id"].astype(str))
    for _, row in decisions.iterrows():
        canon = str(row["canonical_sample_id"])
        if canon not in clean_ids:
            errors.append(f"canonical missing in samples_clean: {canon}")
            break
        if bool(row["keep"]) and str(row["sample_id"]) != canon:
            errors.append(f"keep=true but not canonical: {row['sample_id']}")
            break
        if (not bool(row["keep"])) and str(row["sample_id"]) == canon:
            errors.append(f"alias marked as canonical: {row['sample_id']}")
            break

    if len(labels) and (~labels["sample_id"].astype(str).isin(clean_ids)).any():
        errors.append("labels_clean reference missing sample")
    if len(spatial) and (~spatial["sample_id"].astype(str).isin(clean_ids)).any():
        errors.append("spatial_clean reference missing sample")

    # TonguExpert L2 仅 pseudo
    if len(assignments):
        l2 = assignments[
            (assignments["dataset"] == "tonguexpert")
            & (assignments["label_source"] == "model_prediction")
        ]
        if len(l2) and (l2["supervision_pool"] != "pseudo").any():
            errors.append("TonguExpert L2 not exclusively pseudo")
        if len(l2) and (l2["supervision_pool"] == "gold_candidate").any():
            errors.append("TonguExpert L2 marked gold_candidate")

        dsct = assignments[assignments["dataset"] == "dsct"]
        if len(dsct):
            if (dsct["supervision_pool"] != "external_holdout").any():
                errors.append("DSCT not exclusively external_holdout")
            if dsct["eligible_for_train"].astype(bool).any():
                errors.append("DSCT marked eligible_for_train")

        stained = assignments[assignments["dataset"] == "stained_coating"]
        if len(stained):
            bad = stained[
                stained["canonical_task"].astype(str).str.startswith("coating.color")
            ]
            if len(bad):
                errors.append("Stained assigned coating.color supervision")

    # stained labels 不得出现病理苔色
    if len(labels):
        stained_labels = labels[labels["source_dataset"] == "stained_coating"]
        if len(stained_labels):
            bad_tasks = stained_labels[
                stained_labels["canonical_task"].astype(str).str.startswith("coating.color")
            ]
            if len(bad_tasks):
                errors.append("Stained labels contain coating.color")

    # cross-dataset duplicate among clean samples
    if len(samples):
        for md5_value, group in samples.groupby("md5"):
            datasets = set(group["dataset"].astype(str))
            if len(datasets) > 1:
                errors.append(f"cross-dataset duplicate md5 in clean samples: {md5_value}")
                break

    # TMC 不得从缺失 bbox 推 negative
    if len(labels) and "annotation_type" in labels.columns:
        tmc_neg = labels[
            (labels["source_dataset"] == "tmc_tongue")
            & (labels["annotation_type"].astype(str) == "derived_image_level_from_bbox")
            & (pd.to_numeric(labels["value"], errors="coerce") == 0)
        ]
        if len(tmc_neg):
            errors.append("TMC contains negative labels derived from missing bbox")

    if policy_path is not None:
        # raw_mutation_allowed 必须为 false（配置层）
        from .policy import CleaningPolicy

        policy = CleaningPolicy(policy_path)
        if policy.global_cfg.get("raw_mutation_allowed", False):
            errors.append("policy allows raw mutation")

    return errors, warnings
