"""按 label / annotation 粒度分配 supervision pool（避免 L1/L2 被 sample 级覆盖）。"""
from __future__ import annotations

import pandas as pd

from .policy import CleaningPolicy


def _override_for_label(dataset_cfg: dict, label_source: str) -> dict:
    overrides = dataset_cfg.get("label_source_overrides") or {}
    return dict(overrides.get(str(label_source), {}))


def build_supervision_assignments(
    labels_clean: pd.DataFrame,
    spatial_clean: pd.DataFrame,
    samples_clean: pd.DataFrame,
    decisions: pd.DataFrame,
    policy: CleaningPolicy,
) -> pd.DataFrame:
    """输出 label/annotation 级监督资格；不做最终 split。"""
    rows = []
    sample_meta = samples_clean.set_index("sample_id", drop=False)

    def base_from_dataset(dataset_name: str) -> dict:
        cfg = policy.dataset_cfg(dataset_name)
        return {
            "supervision_pool": cfg.get("supervision_pool", "silver"),
            "training_role": cfg.get("training_role", cfg.get("role", "unknown")),
            "eligible_for_train": bool(cfg.get("eligible_for_train", True)),
            "eligible_for_val": bool(cfg.get("eligible_for_val", True)),
            "eligible_for_test": bool(cfg.get("eligible_for_test", True)),
            "reason": f"dataset_policy:{dataset_name}",
        }

    if labels_clean is not None and not labels_clean.empty:
        for _, lab in labels_clean.iterrows():
            sample_id = str(lab["sample_id"])
            dataset_name = str(lab["source_dataset"])
            cfg = policy.dataset_cfg(dataset_name)
            assignment = base_from_dataset(dataset_name)
            override = _override_for_label(cfg, lab.get("label_source"))
            assignment.update({k: v for k, v in override.items() if k in assignment or k == "reason"})
            if override:
                assignment["reason"] = (
                    f"label_source_override:{lab.get('label_source')}"
                )
            # stained 任务白名单
            allowed = cfg.get("allowed_canonical_tasks")
            if allowed and str(lab["canonical_task"]) not in allowed:
                assignment.update(
                    {
                        "supervision_pool": "excluded",
                        "eligible_for_train": False,
                        "eligible_for_val": False,
                        "eligible_for_test": False,
                        "reason": "task_not_allowed_by_dataset_policy",
                    }
                )
            # DSCT 强制 external_holdout
            if assignment["supervision_pool"] == "external_holdout":
                assignment["eligible_for_train"] = False
                assignment["eligible_for_val"] = False
            # L2 pseudo 不得进 gold/silver/external
            if str(lab.get("label_source")) == "model_prediction":
                assignment["supervision_pool"] = "pseudo"
                if assignment["eligible_for_val"] or assignment["eligible_for_test"]:
                    # 伪标签默认可用于 train-only 实验
                    assignment["eligible_for_val"] = False
                    assignment["eligible_for_test"] = False
            rows.append(
                {
                    "sample_id": sample_id,
                    "canonical_sample_id": sample_id,
                    "origin_sample_id": lab.get("origin_sample_id", sample_id),
                    "dataset": dataset_name,
                    "unit_type": "label",
                    "canonical_task": lab.get("canonical_task"),
                    "canonical_label": lab.get("canonical_label"),
                    "label_source": lab.get("label_source"),
                    "supervision_tier": lab.get("supervision_tier"),
                    "supervision_pool": assignment["supervision_pool"],
                    "training_role": assignment["training_role"],
                    "eligible_for_train": assignment["eligible_for_train"],
                    "eligible_for_val": assignment["eligible_for_val"],
                    "eligible_for_test": assignment["eligible_for_test"],
                    "reason": assignment["reason"],
                    "policy_version": policy.version,
                }
            )

    if spatial_clean is not None and not spatial_clean.empty:
        for _, ann in spatial_clean.iterrows():
            sample_id = str(ann["sample_id"])
            dataset_name = str(ann["source_dataset"])
            assignment = base_from_dataset(dataset_name)
            # TMC bbox 衍生证据保持 silver；mask 用数据集默认
            if str(ann.get("label_source")) == "unknown_mask_origin":
                # 未确认人工 mask 来源：降为 auxiliary，避免进入 gold
                assignment["supervision_pool"] = "auxiliary"
                assignment["reason"] = "unverified_mask_origin"
            rows.append(
                {
                    "sample_id": sample_id,
                    "canonical_sample_id": sample_id,
                    "origin_sample_id": ann.get("origin_sample_id", sample_id),
                    "dataset": dataset_name,
                    "unit_type": "spatial",
                    "canonical_task": ann.get("annotation_task"),
                    "canonical_label": ann.get("canonical_label"),
                    "label_source": ann.get("label_source"),
                    "supervision_tier": ann.get("supervision_tier"),
                    "supervision_pool": assignment["supervision_pool"],
                    "training_role": assignment["training_role"],
                    "eligible_for_train": assignment["eligible_for_train"],
                    "eligible_for_val": assignment["eligible_for_val"],
                    "eligible_for_test": assignment["eligible_for_test"],
                    "reason": assignment["reason"],
                    "policy_version": policy.version,
                }
            )

    # 无 label/spatial 的 canonical sample 也保留 sample 级资格行（如纯分割已在 spatial）
    assigned_samples = {r["sample_id"] for r in rows}
    for sample_id in samples_clean["sample_id"].astype(str):
        if sample_id in assigned_samples:
            continue
        dataset_name = str(sample_meta.loc[sample_id]["dataset"])
        assignment = base_from_dataset(dataset_name)
        rows.append(
            {
                "sample_id": sample_id,
                "canonical_sample_id": sample_id,
                "origin_sample_id": sample_id,
                "dataset": dataset_name,
                "unit_type": "sample",
                "canonical_task": None,
                "canonical_label": None,
                "label_source": None,
                "supervision_tier": None,
                "supervision_pool": assignment["supervision_pool"],
                "training_role": assignment["training_role"],
                "eligible_for_train": assignment["eligible_for_train"],
                "eligible_for_val": assignment["eligible_for_val"],
                "eligible_for_test": assignment["eligible_for_test"],
                "reason": assignment["reason"],
                "policy_version": policy.version,
            }
        )

    return pd.DataFrame(rows)
