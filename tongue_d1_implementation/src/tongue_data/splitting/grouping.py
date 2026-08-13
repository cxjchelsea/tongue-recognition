"""泄漏安全 split_group：union-find / connected components。"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .policy import SplitPolicy


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, node: str):
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, node: str) -> str:
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str):
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _patient_key(patient_id: str) -> str:
    return f"patient::tonguedx::{patient_id}"


def _sample_key(sample_id: str) -> str:
    return f"sample::{sample_id}"


def _md5_key(md5_value: str) -> str:
    return f"md5::{md5_value}"


def build_leakage_components(
    samples_clean: pd.DataFrame,
    decisions: pd.DataFrame,
    samples_raw: pd.DataFrame | None,
    policy: SplitPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构建 sample → split_group_id，并返回 group 表与 audit。"""
    uf = UnionFind()
    warnings = []
    identity_collisions = []

    # 原始 sample → patient（含 alias）
    raw_patient = {}
    if samples_raw is not None and len(samples_raw):
        for _, row in samples_raw.iterrows():
            raw_patient[str(row["sample_id"])] = (
                None if not bool(row.get("patient_id_available", False)) else str(row.get("patient_id"))
            )

    # canonical → origin sample ids
    origins_by_canonical = defaultdict(list)
    for _, row in decisions.iterrows():
        origins_by_canonical[str(row["canonical_sample_id"])].append(str(row["sample_id"]))

    missing_patient = 0
    for _, row in samples_clean.iterrows():
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        md5_value = str(row["md5"])
        sample_node = _sample_key(sample_id)
        uf.add(sample_node)
        uf.union(sample_node, _md5_key(md5_value))

        dataset_cfg = policy.dataset_cfg(dataset)
        grouping = dataset_cfg.get("grouping", "canonical_sample")

        if grouping == "external_holdout":
            # DSCT：每个 sample 独立组件，后续 forced_split
            continue

        if grouping == "patient_id" and dataset == "tonguedx":
            origin_patients = set()
            for origin_id in origins_by_canonical.get(sample_id, [sample_id]):
                patient = raw_patient.get(origin_id)
                if patient is None and origin_id == sample_id:
                    # clean 行自身
                    if bool(row.get("patient_id_available", False)) and pd.notna(row.get("patient_id")):
                        patient = str(row["patient_id"])
                if patient is not None and str(patient).strip() not in {"", "None", "nan"}:
                    origin_patients.add(str(patient))
            if len(origin_patients) > 1:
                identity_collisions.append(
                    {
                        "canonical_sample_id": sample_id,
                        "patient_ids": sorted(origin_patients),
                    }
                )
                # 全部 union
                for patient in origin_patients:
                    uf.union(sample_node, _patient_key(patient))
            elif len(origin_patients) == 1:
                uf.union(sample_node, _patient_key(next(iter(origin_patients))))
            else:
                missing_patient += 1
                warnings.append(f"tonguedx missing patient_id: {sample_id}")
                # fallback：不与 unknown 合并
        # canonical_sample：仅 sample+md5 关系（已建立）

    # 为每个 sample 分配稳定 split_group_id
    component_members = defaultdict(list)
    for _, row in samples_clean.iterrows():
        sample_id = str(row["sample_id"])
        root = uf.find(_sample_key(sample_id))
        component_members[root].append(sample_id)

    # 稳定命名：按成员字典序最小 sample_id
    root_to_group = {}
    for root, members in component_members.items():
        members_sorted = sorted(members)
        root_to_group[root] = f"grp::{members_sorted[0]}"

    assignment_rows = []
    for _, row in samples_clean.iterrows():
        sample_id = str(row["sample_id"])
        dataset = str(row["dataset"])
        dataset_cfg = policy.dataset_cfg(dataset)
        root = uf.find(_sample_key(sample_id))
        group_id = root_to_group[root]
        forced = dataset_cfg.get("forced_split")
        grouping = dataset_cfg.get("grouping", "canonical_sample")
        if grouping == "patient_id":
            group_type = "patient_component"
        elif grouping == "external_holdout":
            group_type = "external_holdout"
        else:
            group_type = "canonical_sample"
        patient_id = None
        if bool(row.get("patient_id_available", False)) and pd.notna(row.get("patient_id")):
            patient_id = str(row["patient_id"])
        assignment_rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "split_group_id": group_id,
                "group_type": group_type,
                "patient_id": patient_id,
                "md5": str(row["md5"]),
                "duplicate_group_id": row.get("duplicate_group_id"),
                "forced_split": forced,
                "group_reason": grouping,
                "split_policy_version": policy.version,
            }
        )
    sample_groups = pd.DataFrame(assignment_rows)

    # group 表
    group_rows = []
    for group_id, group in sample_groups.groupby("split_group_id", sort=True):
        datasets = sorted(set(group["dataset"].astype(str)))
        patients = sorted({p for p in group["patient_id"].dropna().astype(str).tolist() if p})
        forced_values = {f for f in group["forced_split"].dropna().astype(str).tolist() if f}
        forced_split = next(iter(forced_values)) if len(forced_values) == 1 else None
        if "external_holdout" in set(group["group_type"]):
            forced_split = "external_holdout"
        group_rows.append(
            {
                "split_group_id": group_id,
                "group_type": sorted(set(group["group_type"].astype(str)))[0],
                "member_count": int(len(group)),
                "datasets": "|".join(datasets),
                "patient_ids": "|".join(patients),
                "md5_count": int(group["md5"].nunique()),
                "eligible_for_regular_split": forced_split is None,
                "forced_split": forced_split,
                "reason": sorted(set(group["group_reason"].astype(str)))[0],
                "policy_version": policy.version,
            }
        )
    split_groups = pd.DataFrame(group_rows)

    audit = {
        "total_canonical_samples": int(len(samples_clean)),
        "total_leakage_groups": int(len(split_groups)),
        "tonguedx_missing_patient_id_count": int(missing_patient),
        "canonical_with_multiple_origin_patient_ids": identity_collisions,
        "warnings_count": len(warnings),
        "warnings": warnings[:200],
    }
    return sample_groups, split_groups, audit
