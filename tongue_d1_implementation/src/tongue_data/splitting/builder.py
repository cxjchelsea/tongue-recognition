"""D2-B/C Split Builder：group → audit → split → validate。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import (
    build_split_report,
    build_task_distribution,
    enrich_group_audit,
    enrich_split_groups_with_supervision,
)
from .grouping import build_leakage_components
from .policy import SplitPolicy
from .splitter import apply_effective_supervision, assign_splits, attach_supervision_summary
from .stratification import build_group_task_vectors
from .validators import compute_leakage_counts, validate_split


def _git_commit(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


class SplitBuilder:
    def __init__(self, policy_path: str | Path):
        self.policy = SplitPolicy(policy_path)
        self.policy_path = Path(policy_path)

    def build_groups(
        self,
        processed_dir: str | Path,
        manifest_dir: str | Path | None,
        output_dir: str | Path,
        report_dir: str | Path,
    ) -> dict:
        processed_dir = Path(processed_dir)
        output_dir = Path(output_dir)
        report_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        samples_clean = pd.read_parquet(processed_dir / "samples_clean.parquet")
        decisions = pd.read_parquet(processed_dir / "dedup_decisions.parquet")
        assignments = pd.read_parquet(processed_dir / "supervision_assignments.parquet")

        samples_raw = None
        if manifest_dir is not None:
            raw_path = Path(manifest_dir) / "samples.parquet"
            if raw_path.exists():
                samples_raw = pd.read_parquet(raw_path)

        sample_groups, split_groups, base_audit = build_leakage_components(
            samples_clean, decisions, samples_raw, self.policy
        )
        split_groups = enrich_split_groups_with_supervision(
            split_groups, sample_groups, assignments
        )
        group_audit = enrich_group_audit(
            base_audit, sample_groups, split_groups, samples_clean, assignments
        )

        sample_groups.to_parquet(output_dir / "sample_group_assignments.parquet", index=False)
        split_groups.to_parquet(output_dir / "split_groups.parquet", index=False)
        (report_dir / "group_audit.json").write_text(
            json.dumps(group_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return group_audit

    def build_split(
        self,
        processed_dir: str | Path,
        output_dir: str | Path,
        report_dir: str | Path,
        groups_path: str | Path | None = None,
    ) -> dict:
        processed_dir = Path(processed_dir)
        output_dir = Path(output_dir)
        report_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        samples_clean = pd.read_parquet(processed_dir / "samples_clean.parquet")
        labels_clean = pd.read_parquet(processed_dir / "labels_clean.parquet")
        assignments = pd.read_parquet(processed_dir / "supervision_assignments.parquet")
        decisions = pd.read_parquet(processed_dir / "dedup_decisions.parquet")

        if groups_path is None:
            sample_groups = pd.read_parquet(output_dir / "sample_group_assignments.parquet")
            split_groups = pd.read_parquet(output_dir / "split_groups.parquet")
        else:
            groups_root = Path(groups_path)
            if groups_root.is_file():
                split_groups = pd.read_parquet(groups_root)
                sample_groups = pd.read_parquet(
                    groups_root.parent / "sample_group_assignments.parquet"
                )
            else:
                sample_groups = pd.read_parquet(groups_root / "sample_group_assignments.parquet")
                split_groups = pd.read_parquet(groups_root / "split_groups.parquet")

        group_vectors = build_group_task_vectors(
            sample_groups, labels_clean, assignments, self.policy
        )
        split_assignments = assign_splits(
            split_groups, sample_groups, group_vectors, self.policy
        )
        unstratifiable = list(split_assignments.attrs.get("unstratifiable", []))
        split_assignments = attach_supervision_summary(split_assignments, assignments)

        split_supervision = apply_effective_supervision(
            split_assignments, assignments, self.policy
        )

        leakage = compute_leakage_counts(split_assignments, split_supervision, decisions)
        split_report = build_split_report(split_assignments, self.policy, unstratifiable)
        task_distribution = build_task_distribution(
            split_assignments, labels_clean, split_supervision, self.policy
        )

        package_root = Path(__file__).resolve().parents[3]
        meta = {
            "stage": "D2-B/C",
            "split_policy_version": self.policy.version,
            "seed": self.policy.seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(package_root),
            "samples_total": int(len(samples_clean)),
            "leakage_counts": leakage["counts"],
            "ratios_actual": split_report["ratios_actual"],
            "unstratifiable_label_count": len(unstratifiable),
        }

        split_assignments.to_parquet(output_dir / "split_assignments.parquet", index=False)
        split_supervision.to_parquet(
            output_dir / "split_supervision_assignments.parquet", index=False
        )
        (output_dir / "split_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "split_report.json").write_text(
            json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "leakage_report.json").write_text(
            json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "task_distribution.json").write_text(
            json.dumps(task_distribution, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        errors, warnings = validate_split(output_dir, processed_dir, self.policy_path)
        meta["validate_split_errors"] = errors
        meta["validate_split_warnings"] = warnings
        (output_dir / "split_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if errors:
            raise ValueError(f"validate-split failed: {errors}")
        return {
            "split_report": split_report,
            "leakage": leakage["counts"],
            "task_distribution_warnings": task_distribution.get("distribution_warnings", []),
        }

    def build_all(
        self,
        processed_dir: str | Path,
        manifest_dir: str | Path | None,
        output_dir: str | Path,
        report_dir: str | Path,
    ) -> dict:
        group_audit = self.build_groups(processed_dir, manifest_dir, output_dir, report_dir)
        split_result = self.build_split(processed_dir, output_dir, report_dir)
        return {"group_audit": group_audit, **split_result}
