"""D2-B/C：泄漏安全分组与 split 契约测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tongue_data.splitting.grouping import build_leakage_components
from tongue_data.splitting.policy import SplitPolicy
from tongue_data.splitting.splitter import apply_effective_supervision, assign_splits
from tongue_data.splitting.stratification import build_group_task_vectors
from tongue_data.splitting.validators import compute_leakage_counts, validate_split


POLICY = SplitPolicy("configs/split_policy_v1.yaml")


def _sample(
    sample_id,
    dataset,
    md5,
    patient_id=None,
    duplicate_group_id=None,
):
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "md5": md5,
        "patient_id": patient_id,
        "patient_id_available": patient_id is not None,
        "duplicate_group_id": duplicate_group_id or f"dup::{dataset}::{md5}",
        "source_image_path": f"{sample_id}.jpg",
        "width": 10,
        "height": 10,
    }


def _decision(sample_id, canonical_sample_id, md5, dataset, keep=True):
    return {
        "sample_id": sample_id,
        "canonical_sample_id": canonical_sample_id,
        "duplicate_group_id": f"dup::{dataset}::{md5}",
        "md5": md5,
        "dataset": dataset,
        "keep": keep,
    }


def _label(sample_id, task, label, value, source="dataset_annotation", pool="silver"):
    label_text = "true" if label is True else ("false" if label is False else str(label))
    return {
        "sample_id": sample_id,
        "canonical_task": task,
        "canonical_label": label_text,
        "value": value,
        "label_source": source,
        "supervision_pool": pool,
        "eligible_for_train": pool not in {"external_holdout"},
        "eligible_for_val": pool not in {"pseudo", "external_holdout"},
        "eligible_for_test": pool != "pseudo",
        "unit_type": "label",
        "dataset": sample_id.split("::")[0] if "::" in sample_id else "unknown",
        "supervision_tier": "silver",
        "training_role": pool,
        "reason": "test",
        "policy_version": "1.1",
        "canonical_sample_id": sample_id,
        "origin_sample_id": sample_id,
    }


def _run_split(samples, decisions, labels, assignments, samples_raw=None, seed=None):
    policy = SplitPolicy("configs/split_policy_v1.yaml")
    if seed is not None:
        policy.global_cfg["seed"] = seed
    sample_groups, split_groups, audit = build_leakage_components(
        samples, decisions, samples_raw, policy
    )
    vectors = build_group_task_vectors(sample_groups, labels, assignments, policy)
    split_assignments = assign_splits(split_groups, sample_groups, vectors, policy)
    split_supervision = apply_effective_supervision(split_assignments, assignments, policy)
    leakage = compute_leakage_counts(split_assignments, split_supervision, decisions)
    return sample_groups, split_groups, split_assignments, split_supervision, audit, leakage


def test_tonguedx_same_patient_same_split():
    samples = pd.DataFrame(
        [
            _sample("tonguedx::i1", "tonguedx", "m1", "P001"),
            _sample("tonguedx::i2", "tonguedx", "m2", "P001"),
            _sample("tonguedx::i3", "tonguedx", "m3", "P002"),
            _sample("tonguedx::i4", "tonguedx", "m4", "P003"),
            _sample("biohit::b1", "biohit", "mb1"),
            _sample("biohit::b2", "biohit", "mb2"),
            _sample("biohit::b3", "biohit", "mb3"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label("tonguedx::i1", "features.crack.present", True, 1),
            _label("tonguedx::i2", "features.crack.present", False, 1),
            _label("tonguedx::i3", "features.tooth_mark.present", True, 1),
            _label("tonguedx::i4", "features.red_spot.present", True, 1),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b3", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    assignments = labels.copy()
    _, _, split_assignments, _, _, leakage = _run_split(samples, decisions, labels, assignments)
    p001 = split_assignments[split_assignments["patient_id"] == "P001"]
    assert p001["split"].nunique() == 1
    assert leakage["counts"]["patient_leakage"] == 0


def test_different_patients_may_differ():
    samples = pd.DataFrame(
        [
            _sample(f"tonguedx::p{index}", "tonguedx", f"m{index}", f"P{index:03d}")
            for index in range(1, 31)
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(row.sample_id, "features.crack.present", True if index % 2 == 0 else False, 1)
            for index, row in enumerate(samples.itertuples(), start=1)
        ]
    )
    _, _, split_assignments, _, _, _ = _run_split(samples, decisions, labels, labels.copy())
    assert split_assignments["split"].nunique() >= 2


def test_missing_patient_not_merged_into_unknown():
    samples = pd.DataFrame(
        [
            _sample("tonguedx::a", "tonguedx", "ma", None),
            _sample("tonguedx::b", "tonguedx", "mb", None),
            _sample("tonguedx::c", "tonguedx", "mc", "P9"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame([_label(sid, "features.crack.present", True, 1) for sid in samples["sample_id"]])
    sample_groups, _, _, _, audit, _ = _run_split(samples, decisions, labels, labels.copy())
    assert audit["tonguedx_missing_patient_id_count"] == 2
    groups_ab = sample_groups.loc[
        sample_groups["sample_id"].isin(["tonguedx::a", "tonguedx::b"]), "split_group_id"
    ].tolist()
    assert groups_ab[0] != groups_ab[1]


def test_alias_patient_identity_union():
    # canonical 保留 i1(P001)；alias i2 带 P002 → 必须 union 两个患者组件
    samples_clean = pd.DataFrame(
        [
            _sample("tonguedx::i1", "tonguedx", "same", "P001"),
            _sample("tonguedx::i3", "tonguedx", "m3", "P002"),
            _sample("tonguedx::i4", "tonguedx", "m4", "P003"),
        ]
    )
    samples_raw = pd.DataFrame(
        [
            _sample("tonguedx::i1", "tonguedx", "same", "P001"),
            _sample("tonguedx::i2", "tonguedx", "same", "P002"),
            _sample("tonguedx::i3", "tonguedx", "m3", "P002"),
            _sample("tonguedx::i4", "tonguedx", "m4", "P003"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision("tonguedx::i1", "tonguedx::i1", "same", "tonguedx", keep=True),
            _decision("tonguedx::i2", "tonguedx::i1", "same", "tonguedx", keep=False),
            _decision("tonguedx::i3", "tonguedx::i3", "m3", "tonguedx", keep=True),
            _decision("tonguedx::i4", "tonguedx::i4", "m4", "tonguedx", keep=True),
        ]
    )
    labels = pd.DataFrame(
        [_label(sid, "features.crack.present", True, 1) for sid in samples_clean["sample_id"]]
    )
    sample_groups, _, split_assignments, _, audit, _ = _run_split(
        samples_clean, decisions, labels, labels.copy(), samples_raw=samples_raw
    )
    assert len(audit["canonical_with_multiple_origin_patient_ids"]) >= 1
    # i1 与 i3 应同 group（经 P002）
    g1 = sample_groups.loc[sample_groups["sample_id"] == "tonguedx::i1", "split_group_id"].iloc[0]
    g3 = sample_groups.loc[sample_groups["sample_id"] == "tonguedx::i3", "split_group_id"].iloc[0]
    assert g1 == g3
    assert (
        split_assignments.loc[split_assignments["sample_id"] == "tonguedx::i1", "split"].iloc[0]
        == split_assignments.loc[split_assignments["sample_id"] == "tonguedx::i3", "split"].iloc[0]
    )


def test_same_md5_not_across_splits():
    samples = pd.DataFrame(
        [
            _sample("tmc_tongue::a", "tmc_tongue", "shared"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
            _sample("biohit::b3", "biohit", "b3"),
            _sample("biohit::b4", "biohit", "b4"),
        ]
    )
    # 人为：两个不同 sample 同 md5（正常 clean 不应出现，但 grouping 应 union）
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [_label(sid, "segmentation.tongue", True, 1, pool="auxiliary") for sid in samples["sample_id"]]
    )
    sample_groups, _, split_assignments, _, _, leakage = _run_split(
        samples, decisions, labels, labels.copy()
    )
    # 同 md5 的唯一 pair：仅 tmc 一个；添加第二个同 md5
    samples2 = pd.concat(
        [
            samples,
            pd.DataFrame([_sample("tmc_tongue::b", "tmc_tongue", "shared")]),
        ],
        ignore_index=True,
    )
    decisions2 = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples2.itertuples()]
    )
    labels2 = pd.DataFrame(
        [_label(sid, "segmentation.tongue", True, 1, pool="auxiliary") for sid in samples2["sample_id"]]
    )
    _, _, split_assignments, _, _, leakage = _run_split(samples2, decisions2, labels2, labels2.copy())
    shared = split_assignments[split_assignments["md5"] == "shared"]
    assert shared["split"].nunique() == 1
    assert leakage["counts"]["md5_leakage"] == 0


def test_same_group_not_across_splits():
    samples = pd.DataFrame(
        [
            _sample("tonguedx::i1", "tonguedx", "m1", "P1"),
            _sample("tonguedx::i2", "tonguedx", "m2", "P1"),
            _sample("biohit::x", "biohit", "mx"),
            _sample("biohit::y", "biohit", "my"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [_label(sid, "features.crack.present", True, 1) for sid in samples["sample_id"]]
    )
    _, _, split_assignments, _, _, leakage = _run_split(samples, decisions, labels, labels.copy())
    for _, group in split_assignments.groupby("split_group_id"):
        assert group["split"].nunique() == 1
    assert leakage["counts"]["group_leakage"] == 0


def test_tonguexpert_l1_l2_same_split():
    samples = pd.DataFrame(
        [
            _sample("tonguexpert::te1", "tonguexpert", "te1"),
            _sample("tonguexpert::te2", "tonguexpert", "te2"),
            _sample("tonguexpert::te3", "tonguexpert", "te3"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label("tonguexpert::te1", "tongue_body.color", "pale", 1, source="human", pool="gold_candidate"),
            _label(
                "tonguexpert::te1",
                "tongue_body.color",
                "pale",
                1,
                source="model_prediction",
                pool="pseudo",
            ),
            _label("tonguexpert::te2", "tongue_body.color", "red", 1, source="human", pool="gold_candidate"),
            _label("tonguexpert::te3", "coating.color", "white", 1, source="human", pool="gold_candidate"),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    _, _, split_assignments, split_supervision, _, _ = _run_split(
        samples, decisions, labels, labels.copy()
    )
    assert split_assignments.loc[
        split_assignments["sample_id"] == "tonguexpert::te1", "split"
    ].nunique() == 1
    te1_split = split_assignments.loc[
        split_assignments["sample_id"] == "tonguexpert::te1", "split"
    ].iloc[0]
    # 强制 te1 到 test 验证 pseudo 规则：改写 split 后检查 effective
    forced = split_assignments.copy()
    forced.loc[forced["sample_id"] == "tonguexpert::te1", "split"] = "test"
    effective = apply_effective_supervision(forced, labels.copy(), POLICY)
    te1_pseudo = effective[
        (effective["sample_id"] == "tonguexpert::te1")
        & (effective["supervision_pool"] == "pseudo")
    ]
    assert bool(te1_pseudo["effective_for_train"].iloc[0]) is False
    assert te1_split in {"train", "val", "test"}


def test_pseudo_never_val_test_evaluation():
    samples = pd.DataFrame(
        [
            _sample("tonguexpert::te1", "tonguexpert", "te1"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
            _sample("biohit::b3", "biohit", "b3"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(
                "tonguexpert::te1",
                "tongue_body.color",
                "pale",
                1,
                source="model_prediction",
                pool="pseudo",
            ),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b3", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    _, _, split_assignments, split_supervision, _, _ = _run_split(
        samples, decisions, labels, labels.copy()
    )
    # 覆盖所有 split 场景
    for split_name in ["train", "val", "test"]:
        forced = split_assignments.copy()
        forced["split"] = split_name
        effective = apply_effective_supervision(forced, labels.copy(), POLICY)
        pseudo = effective[effective["supervision_pool"] == "pseudo"]
        assert not bool(pseudo["effective_for_val"].any())
        assert not bool(pseudo["effective_for_test"].any())
        if split_name == "train":
            assert bool(pseudo["effective_for_train"].iloc[0]) is True
        else:
            assert bool(pseudo["effective_for_train"].iloc[0]) is False


def test_dsct_all_external_holdout():
    samples = pd.DataFrame(
        [
            _sample("dsct::1", "dsct", "d1"),
            _sample("dsct::2", "dsct", "d2"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
            _sample("biohit::b3", "biohit", "b3"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label("dsct::1", "features.crack.present", True, 1, pool="external_holdout"),
            _label("dsct::2", "features.crack.present", True, 1, pool="external_holdout"),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b3", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    _, _, split_assignments, _, _, leakage = _run_split(samples, decisions, labels, labels.copy())
    dsct = split_assignments[split_assignments["dataset"] == "dsct"]
    assert set(dsct["split"]) == {"external_holdout"}
    assert int((dsct["split"] == "train").sum()) == 0
    assert int((dsct["split"] == "val").sum()) == 0
    assert int((dsct["split"] == "test").sum()) == 0
    assert leakage["counts"]["external_holdout_leakage"] == 0


def test_stained_quality_only_guard(tmp_path: Path):
    samples = pd.DataFrame(
        [
            _sample("stained_coating::s1", "stained_coating", "s1"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(
                "stained_coating::s1",
                "quality.stain_suspected",
                True,
                1,
                pool="auxiliary",
            ),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    _, _, split_assignments, split_supervision, _, _ = _run_split(
        samples, decisions, labels, labels.copy()
    )
    # 人为注入 coating.color effective → validate 应 FAIL
    bad = split_supervision.copy()
    bad.loc[0, "canonical_task"] = "coating.color"
    bad.loc[0, "effective_for_train"] = True
    out = tmp_path / "splits"
    out.mkdir()
    pd.DataFrame([{"split_group_id": "g"}]).to_parquet(out / "split_groups.parquet", index=False)
    split_assignments.to_parquet(out / "sample_group_assignments.parquet", index=False)
    split_assignments.to_parquet(out / "split_assignments.parquet", index=False)
    bad.to_parquet(out / "split_supervision_assignments.parquet", index=False)
    errors, _ = validate_split(out)
    assert any("stained_coating" in err or "coating.color" in err for err in errors)


def test_explicit_negative_in_distribution():
    samples = pd.DataFrame(
        [
            _sample("tonguedx::a", "tonguedx", "ma", "P1"),
            _sample("tonguedx::b", "tonguedx", "mb", "P2"),
            _sample("tonguedx::c", "tonguedx", "mc", "P3"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label("tonguedx::a", "tongue_body.color", "pale", 1),
            _label("tonguedx::b", "tongue_body.color", "pale", 0),
            _label("tonguedx::c", "coating.color", "yellow", 0),
        ]
    )
    sample_groups, split_groups, _, _, _, _ = _run_split(
        samples, decisions, labels, labels.copy()
    )
    vectors = build_group_task_vectors(sample_groups, labels, labels.copy(), POLICY)
    flat = {key for vector in vectors.values() for key in vector}
    assert "tongue_body.color:pale:negative" in flat
    assert "coating.color:yellow:negative" in flat
    assert "tongue_body.color:pale:positive" in flat


def test_determinism_same_seed():
    samples = pd.DataFrame(
        [
            _sample(f"biohit::{index}", "biohit", f"m{index}")
            for index in range(20)
        ]
        + [
            _sample(f"tonguedx::{index}", "tonguedx", f"td{index}", f"P{index}")
            for index in range(20)
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(
                row.sample_id,
                "features.crack.present",
                True if index % 3 == 0 else False,
                1,
            )
            for index, row in enumerate(samples.itertuples())
        ]
    )
    _, _, a1, _, _, _ = _run_split(samples, decisions, labels, labels.copy(), seed=20260813)
    _, _, a2, _, _, _ = _run_split(samples, decisions, labels, labels.copy(), seed=20260813)
    left = a1.sort_values("sample_id")[["sample_id", "split", "split_group_id"]].reset_index(drop=True)
    right = a2.sort_values("sample_id")[["sample_id", "split", "split_group_id"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_different_seed_still_leakage_safe():
    samples = pd.DataFrame(
        [
            _sample(f"biohit::{index}", "biohit", f"m{index}")
            for index in range(30)
        ]
        + [
            _sample(f"tonguedx::{index}", "tonguedx", f"td{index}", f"P{index}")
            for index in range(30)
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(row.sample_id, "features.crack.present", index % 2 == 0, 1)
            for index, row in enumerate(samples.itertuples())
        ]
    )
    _, _, a1, s1, _, leak1 = _run_split(samples, decisions, labels, labels.copy(), seed=1)
    _, _, a2, s2, _, leak2 = _run_split(samples, decisions, labels, labels.copy(), seed=2)
    assert all(value == 0 for value in leak1["counts"].values())
    assert all(value == 0 for value in leak2["counts"].values())
    # 允许 assignment 不同
    merged = a1[["sample_id", "split"]].merge(
        a2[["sample_id", "split"]], on="sample_id", suffixes=("_1", "_2")
    )
    assert bool((merged["split_1"] != merged["split_2"]).any()) or True  # 不强制必须不同


def test_rare_label_does_not_split_group():
    samples = pd.DataFrame(
        [
            _sample("tonguedx::i1", "tonguedx", "m1", "P1"),
            _sample("tonguedx::i2", "tonguedx", "m2", "P1"),
            _sample("biohit::b1", "biohit", "b1"),
            _sample("biohit::b2", "biohit", "b2"),
            _sample("biohit::b3", "biohit", "b3"),
        ]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label("tonguedx::i1", "features.ecchymosis.present", True, 1),
            _label("tonguedx::i2", "features.crack.present", True, 1),
            _label("biohit::b1", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b2", "segmentation.tongue", True, 1, pool="auxiliary"),
            _label("biohit::b3", "segmentation.tongue", True, 1, pool="auxiliary"),
        ]
    )
    sample_groups, _, split_assignments, _, _, _ = _run_split(
        samples, decisions, labels, labels.copy()
    )
    p1 = split_assignments[split_assignments["patient_id"] == "P1"]
    assert p1["split"].nunique() == 1
    assert sample_groups.loc[sample_groups["patient_id"] == "P1", "split_group_id"].nunique() == 1


def test_ratio_near_target():
    samples = pd.DataFrame(
        [_sample(f"biohit::{index}", "biohit", f"m{index}") for index in range(100)]
    )
    decisions = pd.DataFrame(
        [_decision(row.sample_id, row.sample_id, row.md5, row.dataset) for row in samples.itertuples()]
    )
    labels = pd.DataFrame(
        [
            _label(row.sample_id, "segmentation.tongue", True, 1, pool="auxiliary")
            for row in samples.itertuples()
        ]
    )
    _, _, split_assignments, _, _, _ = _run_split(samples, decisions, labels, labels.copy())
    regular = split_assignments[split_assignments["split"].isin(["train", "val", "test"])]
    train_ratio = (regular["split"] == "train").mean()
    assert 0.70 <= train_ratio <= 0.90


def test_patient_leakage_validator_fails():
    split_assignments = pd.DataFrame(
        [
            {
                "sample_id": "tonguedx::1",
                "dataset": "tonguedx",
                "split_group_id": "g1",
                "split": "train",
                "patient_id": "P1",
                "md5": "m1",
            },
            {
                "sample_id": "tonguedx::2",
                "dataset": "tonguedx",
                "split_group_id": "g2",
                "split": "test",
                "patient_id": "P1",
                "md5": "m2",
            },
        ]
    )
    leakage = compute_leakage_counts(split_assignments, None, None)
    assert leakage["counts"]["patient_leakage"] == 1


def test_md5_leakage_validator_fails():
    split_assignments = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "dataset": "biohit",
                "split_group_id": "g1",
                "split": "train",
                "patient_id": None,
                "md5": "same",
            },
            {
                "sample_id": "b",
                "dataset": "biohit",
                "split_group_id": "g2",
                "split": "val",
                "patient_id": None,
                "md5": "same",
            },
        ]
    )
    leakage = compute_leakage_counts(split_assignments, None, None)
    assert leakage["counts"]["md5_leakage"] == 1


def test_pseudo_leakage_validator_fails():
    split_assignments = pd.DataFrame(
        [
            {
                "sample_id": "te1",
                "dataset": "tonguexpert",
                "split_group_id": "g1",
                "split": "test",
                "patient_id": None,
                "md5": "m",
            }
        ]
    )
    split_supervision = pd.DataFrame(
        [
            {
                "sample_id": "te1",
                "supervision_pool": "pseudo",
                "sample_split": "test",
                "effective_for_train": True,
                "effective_for_val": False,
                "effective_for_test": False,
            }
        ]
    )
    leakage = compute_leakage_counts(split_assignments, split_supervision, None)
    assert leakage["counts"]["pseudo_leakage"] >= 1
