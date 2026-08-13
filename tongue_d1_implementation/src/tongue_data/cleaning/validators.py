"""D2-A / D2-A.1 clean 产物校验。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .policy import ALLOWED_CONFLICT_POLICIES, CleaningPolicy


def _norm_label(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text
    return str(value)


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

    for group_id, group in decisions.groupby("duplicate_group_id"):
        keep_n = int(group["keep"].sum())
        if keep_n != 1:
            errors.append(f"group {group_id} keep count={keep_n}")
        canonical_ids = set(group["canonical_sample_id"].astype(str))
        if len(canonical_ids) != 1:
            errors.append(f"group {group_id} has multiple canonical ids")

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

    # 冲突 fact 不得进入 clean / train-eligible
    conflict_path = root / "label_conflicts.json"
    if conflict_path.exists() and len(labels):
        label_conflicts = json.loads(conflict_path.read_text(encoding="utf-8"))
        for conflict in label_conflicts:
            sample_id = str(conflict.get("canonical_sample_id"))
            task = str(conflict.get("canonical_task"))
            label_key = str(conflict.get("canonical_label"))
            if label_key == "true|false":
                keys = {"true", "false"}
            else:
                keys = {label_key}
            subset = labels[
                (labels["sample_id"].astype(str) == sample_id)
                & (labels["canonical_task"].astype(str) == task)
                & (labels["canonical_label"].map(_norm_label).isin(keys))
            ]
            if len(subset):
                errors.append(
                    f"conflicted fact still in labels_clean: {sample_id} {task} {label_key}"
                )
                break
            if len(assignments):
                bad_assign = assignments[
                    (assignments["sample_id"].astype(str) == sample_id)
                    & (assignments["canonical_task"].astype(str) == task)
                    & (assignments["canonical_label"].map(_norm_label).isin(keys))
                    & (assignments["eligible_for_train"].astype(bool))
                ]
                if len(bad_assign):
                    errors.append(
                        f"conflicted fact still train-eligible: {sample_id} {task}"
                    )
                    break

    # identical bbox dedup：同 sample+task+label+geometry 最多一条
    if len(spatial):
        geom_cols = ["sample_id", "annotation_task", "canonical_label", "annotation_type",
                     "x_min", "y_min", "x_max", "y_max", "mask_path"]
        present = [c for c in geom_cols if c in spatial.columns]
        dup_geom = spatial.duplicated(subset=present, keep=False)
        if dup_geom.any():
            errors.append("identical spatial geometries not fully deduplicated")

    if len(assignments):
        l2 = assignments[
            (assignments["dataset"] == "tonguexpert")
            & (assignments["label_source"] == "model_prediction")
        ]
        if len(l2) and (l2["supervision_pool"] != "pseudo").any():
            errors.append("TonguExpert L2 not exclusively pseudo")

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

    if len(labels):
        stained_labels = labels[labels["source_dataset"] == "stained_coating"]
        if len(stained_labels):
            bad_tasks = stained_labels[
                stained_labels["canonical_task"].astype(str).str.startswith("coating.color")
            ]
            if len(bad_tasks):
                errors.append("Stained labels contain coating.color")

    if len(samples):
        for md5_value, group in samples.groupby("md5"):
            datasets = set(group["dataset"].astype(str))
            if len(datasets) > 1:
                errors.append(f"cross-dataset duplicate md5 in clean samples: {md5_value}")
                break

    if len(labels) and "annotation_type" in labels.columns:
        tmc_neg = labels[
            (labels["source_dataset"] == "tmc_tongue")
            & (labels["annotation_type"].astype(str) == "derived_image_level_from_bbox")
            & (pd.to_numeric(labels["value"], errors="coerce") == 0)
        ]
        if len(tmc_neg):
            errors.append("TMC contains negative labels derived from missing bbox")

    if policy_path is not None:
        try:
            policy = CleaningPolicy(policy_path)
        except ValueError as exc:
            errors.append(str(exc))
            return errors, warnings
        if policy.global_cfg.get("raw_mutation_allowed", False):
            errors.append("policy allows raw mutation")
        if policy.conflict_policy() not in ALLOWED_CONFLICT_POLICIES:
            errors.append(f"illegal conflict_policy={policy.conflict_policy()}")

    return errors, warnings
