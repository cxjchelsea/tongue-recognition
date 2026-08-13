"""D2-A Cleaning Builder：从冻结 D1.1 manifest 生成 clean 产物与报告。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .dedup import (
    build_duplicate_groups,
    find_cross_dataset_duplicates,
    select_canonical_samples,
)
from .policy import CleaningPolicy
from .reconciliation import reconcile_labels, reconcile_spatial
from .supervision import build_supervision_assignments


class CleaningBuilder:
    def __init__(self, policy_path: str | Path):
        self.policy = CleaningPolicy(policy_path)

    def build(
        self,
        manifest_dir: str | Path,
        output_dir: str | Path,
        report_dir: str | Path,
    ) -> dict:
        if self.policy.global_cfg.get("raw_mutation_allowed", False):
            raise ValueError("raw_mutation_allowed must be false for D2-A")

        manifest_dir = Path(manifest_dir)
        output_dir = Path(output_dir)
        report_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        samples = pd.read_parquet(manifest_dir / "samples.parquet")
        labels = pd.read_parquet(manifest_dir / "labels.parquet")
        spatial = pd.read_parquet(manifest_dir / "spatial_annotations.parquet")

        samples_before = int(len(samples))
        labels_before = int(len(labels))
        spatial_before = int(len(spatial))

        grouped = build_duplicate_groups(samples, self.policy)
        decisions = select_canonical_samples(grouped, self.policy)
        cross_dups = find_cross_dataset_duplicates(samples)

        if cross_dups and self.policy.global_cfg.get("cross_dataset_duplicate_policy") == "fail":
            # 仍写出报告，再抛错，便于排障
            (report_dir / "cross_dataset_duplicates.json").write_text(
                json.dumps(cross_dups, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise ValueError(
                f"cross-dataset MD5 duplicates found: {len(cross_dups)} groups"
            )

        labels_clean, label_conflicts, label_stats = reconcile_labels(
            labels, decisions, self.policy
        )
        spatial_clean, spatial_meta, spatial_stats = reconcile_spatial(
            spatial, decisions, self.policy
        )

        # samples_clean：仅 keep=true 的 canonical
        keep_ids = set(decisions.loc[decisions["keep"], "sample_id"].astype(str))
        samples_clean = samples[samples["sample_id"].astype(str).isin(keep_ids)].copy()
        # 回填 duplicate_group_id
        group_map = dict(
            zip(decisions["sample_id"].astype(str), decisions["duplicate_group_id"].astype(str))
        )
        samples_clean["duplicate_group_id"] = samples_clean["sample_id"].astype(str).map(group_map)

        assignments = build_supervision_assignments(
            labels_clean, spatial_clean, samples_clean, decisions, self.policy
        )

        # 写出产物（不覆盖 D1 manifest）
        samples_clean.to_parquet(output_dir / "samples_clean.parquet", index=False)
        labels_clean.to_parquet(output_dir / "labels_clean.parquet", index=False)
        spatial_clean.to_parquet(output_dir / "spatial_clean.parquet", index=False)
        decisions.to_parquet(output_dir / "dedup_decisions.parquet", index=False)
        assignments.to_parquet(output_dir / "supervision_assignments.parquet", index=False)
        # 供 validate-clean 检查：冲突 fact 不得进入 clean labels
        (output_dir / "label_conflicts.json").write_text(
            json.dumps(label_conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        per_dataset = {}
        for dataset_name, before_group in samples.groupby("dataset", sort=True):
            after_n = int((samples_clean["dataset"] == dataset_name).sum())
            before_n = int(len(before_group))
            unique_md5 = int(before_group["md5"].nunique())
            per_dataset[str(dataset_name)] = {
                "samples_before": before_n,
                "unique_md5": unique_md5,
                "duplicate_files": before_n - unique_md5,
                "canonical_samples": after_n,
                "aliases": before_n - after_n,
                "labels_before": int((labels["source_dataset"] == dataset_name).sum())
                if "source_dataset" in labels.columns
                else int((labels["sample_id"].astype(str).str.startswith(f"{dataset_name}::")).sum()),
                "labels_after_merge": int((labels_clean["source_dataset"] == dataset_name).sum())
                if len(labels_clean) and "source_dataset" in labels_clean.columns
                else 0,
                "spatial_before": int((spatial["source_dataset"] == dataset_name).sum())
                if len(spatial) and "source_dataset" in spatial.columns
                else 0,
                "spatial_after_merge": int((spatial_clean["source_dataset"] == dataset_name).sum())
                if len(spatial_clean) and "source_dataset" in spatial_clean.columns
                else 0,
            }

        duplicate_groups = int(decisions["duplicate_group_id"].nunique())
        multi_groups = int(decisions.loc[decisions["is_duplicate"], "duplicate_group_id"].nunique())
        aliases = int((~decisions["keep"]).sum())

        conflict_report = {
            "note": "multi-instance geometry != conflict; only label value contradictions are conflicts",
            "conflict_policy": self.policy.conflict_policy(),
            "duplicate_groups_total": duplicate_groups,
            "duplicate_groups_with_multi_members": multi_groups,
            "label_stats": label_stats,
            "spatial_stats": spatial_stats,
            "label_conflicts": label_conflicts,
            "spatial_identical_duplicates": spatial_stats.get("identical_deduped", 0),
            "spatial_multi_instance_groups": spatial_meta.get("multi_instance_groups", []),
            "spatial_multi_instance_annotations": spatial_stats.get(
                "multi_instance_annotations", 0
            ),
            "spatial_review_groups": spatial_meta.get("review_groups", []),
            "cross_dataset_duplicate_groups": cross_dups,
            "groups_with_conflicting_labels": len(label_conflicts),
            "groups_with_identical_or_deduped_facts": label_stats.get("identical", 0),
            "complementary_facts_kept": label_stats.get("complementary", 0),
        }
        (report_dir / "conflict_report.json").write_text(
            json.dumps(conflict_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        pool_counts = {}
        if len(assignments):
            pool_counts = (
                assignments["supervision_pool"].astype(str).value_counts().to_dict()
            )

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            git_commit = None

        dedup_report = {
            "input_contract_version": "1.1",
            "cleaning_policy_version": self.policy.version,
            "git_commit": git_commit,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "samples_before": samples_before,
            "samples_after": int(len(samples_clean)),
            "labels_before": labels_before,
            "labels_after": int(len(labels_clean)),
            "spatial_before": spatial_before,
            "spatial_after": int(len(spatial_clean)),
            "duplicate_groups": duplicate_groups,
            "duplicate_multi_member_groups": multi_groups,
            "duplicate_aliases": aliases,
            "per_dataset": per_dataset,
            "label_conflicts": len(label_conflicts),
            "spatial_identical_deduped": spatial_stats.get("identical_deduped", 0),
            "spatial_multi_instance_groups": spatial_stats.get("multi_instance_groups", 0),
            "spatial_multi_instance_annotations": spatial_stats.get(
                "multi_instance_annotations", 0
            ),
            "spatial_review_groups": spatial_stats.get("review_groups", 0),
            "cross_dataset_duplicates": len(cross_dups),
            "supervision_pool_counts": {str(k): int(v) for k, v in pool_counts.items()},
        }
        (report_dir / "dedup_report.json").write_text(
            json.dumps(dedup_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        meta = {
            **dedup_report,
            "manifest_dir": str(manifest_dir),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "raw_mutation_allowed": False,
        }
        (output_dir / "cleaning_metadata.json").write_text(
            json.dumps(
                {
                    k: v
                    for k, v in meta.items()
                    if k not in {"manifest_dir", "output_dir", "report_dir"}
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return meta
