"""D2-A：去重、合并、监督池与跨集冲突测试。"""
from pathlib import Path
import pandas as pd
import pytest

from tongue_data.cleaning.policy import CleaningPolicy
from tongue_data.cleaning.dedup import (
    build_duplicate_groups,
    find_cross_dataset_duplicates,
    select_canonical_samples,
)
from tongue_data.cleaning.reconciliation import reconcile_labels, reconcile_spatial
from tongue_data.cleaning.supervision import build_supervision_assignments
from tongue_data.cleaning.builder import CleaningBuilder
from tongue_data.cleaning.validators import validate_clean
from tongue_data.utils import normalize_na


POLICY = CleaningPolicy("configs/cleaning_policy_v1.yaml")


def _sample(sample_id, dataset, md5, path, source_sample_id=None):
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "source_sample_id": source_sample_id or Path(path).name,
        "source_image_path": path,
        "md5": md5,
        "width": 10,
        "height": 10,
        "patient_id": None,
        "patient_id_available": False,
        "source_split": None,
        "duplicate_group_id": None,
        "dataset_version": None,
        "ingest_version": "1.1",
    }


def _as_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _label(sample_id, dataset, task, label, value, field=None, source="dataset_annotation", tier="silver"):
    label_text = _as_text(label)
    return {
        "sample_id": sample_id,
        "canonical_task": task,
        "canonical_label": label_text,
        "value": value,
        "label_available": True,
        "source_dataset": dataset,
        "source_field": field or task,
        "source_label": label_text,
        "annotation_type": "image_level",
        "label_source": source,
        "supervision_tier": tier,
        "mapping_status": "exact",
        "mapping_version": "1.0",
        "confidence": None,
        "note": None,
    }


def test_one_canonical_per_md5_group():
    samples = pd.DataFrame(
        [
            _sample("tmc::b.jpg", "tmc_tongue", "abc", "b.jpg"),
            _sample("tmc::a.jpg", "tmc_tongue", "abc", "a.jpg"),
        ]
    )
    grouped = build_duplicate_groups(samples, POLICY)
    decisions = select_canonical_samples(grouped, POLICY)
    assert int(decisions["keep"].sum()) == 1
    assert decisions.loc[decisions["keep"], "sample_id"].iloc[0] == "tmc::a.jpg"


def test_canonical_selection_deterministic():
    samples = pd.DataFrame(
        [
            _sample("ds::2", "dsct", "m1", "z.jpg", "z"),
            _sample("ds::1", "dsct", "m1", "a.jpg", "a"),
        ]
    )
    d1 = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    d2 = select_canonical_samples(build_duplicate_groups(samples.sample(frac=1, random_state=0), POLICY), POLICY)
    assert (
        d1.loc[d1["keep"], "sample_id"].iloc[0]
        == d2.loc[d2["keep"], "sample_id"].iloc[0]
        == "ds::1"
    )


def test_complementary_labels_merge():
    samples = pd.DataFrame(
        [
            _sample("dx::a", "tonguedx", "m", "a.jpg"),
            _sample("dx::b", "tonguedx", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label("dx::a", "tonguedx", "features.crack.present", True, 1, "Crack"),
            _label("dx::b", "tonguedx", "features.tooth_mark.present", True, 1, "Toothmark"),
        ]
    )
    clean, conflicts, _ = reconcile_labels(labels, decisions, POLICY)
    assert conflicts == []
    tasks = set(clean["canonical_task"])
    assert tasks == {"features.crack.present", "features.tooth_mark.present"}
    assert clean["sample_id"].nunique() == 1


def test_identical_labels_dedup():
    samples = pd.DataFrame(
        [
            _sample("dx::a", "tonguedx", "m", "a.jpg"),
            _sample("dx::b", "tonguedx", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label("dx::a", "tonguedx", "features.crack.present", True, 1),
            _label("dx::b", "tonguedx", "features.crack.present", True, 1),
        ]
    )
    clean, conflicts, stats = reconcile_labels(labels, decisions, POLICY)
    assert conflicts == []
    assert len(clean) == 1
    assert stats["identical"] >= 1


def test_conflicting_pale_not_silent_merge():
    samples = pd.DataFrame(
        [
            _sample("dx::a", "tonguedx", "m", "a.jpg"),
            _sample("dx::b", "tonguedx", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label("dx::a", "tonguedx", "tongue_body.color", "pale", 1, "TonguePale"),
            _label("dx::b", "tonguedx", "tongue_body.color", "pale", 0, "TonguePale"),
        ]
    )
    clean, conflicts, _ = reconcile_labels(labels, decisions, POLICY)
    assert len(conflicts) == 1
    assert not ((clean["canonical_label"].astype(str) == "pale").any())


def test_na_not_in_conflict():
    assert normalize_na("NA") is None
    # NA 根本不会进入 labels 表，故无 conflict 行
    samples = pd.DataFrame([_sample("dx::a", "tonguedx", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [_label("dx::a", "tonguedx", "tongue_body.color", "pale", 1, "TonguePale")]
    )
    clean, conflicts, _ = reconcile_labels(labels, decisions, POLICY)
    assert conflicts == []
    assert len(clean) == 1


def test_tonguexpert_l2_always_pseudo():
    samples = pd.DataFrame([_sample("tx::a", "tonguexpert", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label(
                "tx::a",
                "tonguexpert",
                "tongue_body.color",
                "dark",
                1,
                "zhi_label",
                source="model_prediction",
                tier="pseudo",
            )
        ]
    )
    labels_clean, _, _ = reconcile_labels(labels, decisions, POLICY)
    samples_clean = samples.copy()
    assign = build_supervision_assignments(
        labels_clean, pd.DataFrame(), samples_clean, decisions, POLICY
    )
    assert (assign["supervision_pool"] == "pseudo").all()


def test_dsct_external_holdout_not_train():
    samples = pd.DataFrame([_sample("ds::a", "dsct", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [_label("ds::a", "dsct", "features.crack.severity", "mild", 1)]
    )
    labels_clean, _, _ = reconcile_labels(labels, decisions, POLICY)
    assign = build_supervision_assignments(
        labels_clean, pd.DataFrame(), samples, decisions, POLICY
    )
    assert (assign["supervision_pool"] == "external_holdout").all()
    assert not assign["eligible_for_train"].astype(bool).any()


def test_stained_cannot_be_coating_yellow():
    samples = pd.DataFrame([_sample("st::a", "stained_coating", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    # 即便错误出现 coating.color，也应被排除
    labels = pd.DataFrame(
        [_label("st::a", "stained_coating", "coating.color", "yellow", 1)]
    )
    labels_clean, _, _ = reconcile_labels(labels, decisions, POLICY)
    assign = build_supervision_assignments(
        labels_clean, pd.DataFrame(), samples, decisions, POLICY
    )
    bad = assign[assign["canonical_task"].astype(str) == "coating.color"]
    assert (bad["supervision_pool"] == "excluded").all()
    assert not bad["eligible_for_train"].astype(bool).any()


def test_tmc_missing_bbox_does_not_create_negative():
    # D2 不新增 negative；空 spatial + 无 label 时 clean labels 为空
    samples = pd.DataFrame([_sample("tmc::a", "tmc_tongue", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(columns=[
        "sample_id","canonical_task","canonical_label","value","label_available",
        "source_dataset","source_field","source_label","annotation_type","label_source",
        "supervision_tier","mapping_status","mapping_version","confidence","note"
    ])
    clean, conflicts, _ = reconcile_labels(labels, decisions, POLICY)
    assert len(clean) == 0
    assert conflicts == []


def test_cross_dataset_duplicate_detected():
    samples = pd.DataFrame(
        [
            _sample("tmc::a", "tmc_tongue", "same", "a.jpg"),
            _sample("dx::a", "tonguedx", "same", "a.jpg"),
        ]
    )
    collisions = find_cross_dataset_duplicates(samples)
    assert len(collisions) == 1


def test_builder_end_to_end_and_raw_untouched(tmp_path):
    manifest = tmp_path / "manifest"
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    manifest.mkdir()
    samples = pd.DataFrame(
        [
            _sample("dx::b", "tonguedx", "m1", "b.jpg"),
            _sample("dx::a", "tonguedx", "m1", "a.jpg"),
            _sample("ds::1", "dsct", "m2", "1.jpg"),
        ]
    )
    labels = pd.DataFrame(
        [
            _label("dx::a", "tonguedx", "features.crack.present", True, 1, "Crack"),
            _label("dx::b", "tonguedx", "features.tooth_mark.present", True, 1, "Toothmark"),
            _label("ds::1", "dsct", "features.crack.severity", "mild", 1),
        ]
    )
    spatial = pd.DataFrame(columns=[
        "sample_id","annotation_id","annotation_task","canonical_label","annotation_type",
        "x_min","y_min","x_max","y_max","mask_path","source_dataset","source_label",
        "label_source","supervision_tier","mapping_status","mapping_version","note"
    ])
    samples.to_parquet(manifest / "samples.parquet", index=False)
    labels.to_parquet(manifest / "labels.parquet", index=False)
    spatial.to_parquet(manifest / "spatial_annotations.parquet", index=False)

    # 模拟 raw 文件 md5 校验：记录源表 md5 在 clean 后不变
    before_md5 = set(samples["md5"])
    meta = CleaningBuilder("configs/cleaning_policy_v1.yaml").build(
        manifest, processed, reports
    )
    after_samples = pd.read_parquet(manifest / "samples.parquet")
    assert set(after_samples["md5"]) == before_md5
    assert meta["samples_before"] == 3
    assert meta["samples_after"] == 2
    errors, warnings = validate_clean(processed, "configs/cleaning_policy_v1.yaml")
    assert errors == []


def _bbox(sample_id, dataset, task, label, x_min, y_min, x_max, y_max, annotation_id=None):
    return {
        "sample_id": sample_id,
        "annotation_id": annotation_id or f"{sample_id}::{x_min}",
        "annotation_task": task,
        "canonical_label": _as_text(label),
        "annotation_type": "bbox",
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "mask_path": None,
        "source_dataset": dataset,
        "source_label": _as_text(label),
        "label_source": "dataset_annotation",
        "supervision_tier": "silver",
        "mapping_status": "exact",
        "mapping_version": "1.0",
        "note": None,
    }


def test_legitimate_multi_instance_bboxes_not_conflict():
    samples = pd.DataFrame([_sample("tmc::a", "tmc_tongue", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    spatial = pd.DataFrame(
        [
            _bbox("tmc::a", "tmc_tongue", "features.red_spot.present", True, 0, 0, 10, 10),
            _bbox("tmc::a", "tmc_tongue", "features.red_spot.present", True, 20, 20, 30, 30),
        ]
    )
    clean, meta, stats = reconcile_spatial(spatial, decisions, POLICY)
    assert len(clean) == 2
    assert stats["multi_instance_groups"] == 1
    assert stats["multi_instance_annotations"] == 2
    assert stats["review_groups"] == 0
    assert meta["review_groups"] == []
    assert len(meta["multi_instance_groups"]) == 1


def test_duplicate_origins_keep_multi_instance_bboxes():
    samples = pd.DataFrame(
        [
            _sample("tmc::a", "tmc_tongue", "m", "a.jpg"),
            _sample("tmc::b", "tmc_tongue", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    spatial = pd.DataFrame(
        [
            _bbox("tmc::a", "tmc_tongue", "features.red_spot.present", True, 0, 0, 10, 10),
            _bbox("tmc::b", "tmc_tongue", "features.red_spot.present", True, 20, 20, 30, 30),
        ]
    )
    clean, meta, stats = reconcile_spatial(spatial, decisions, POLICY)
    assert len(clean) == 2
    assert clean["sample_id"].nunique() == 1
    assert stats["multi_instance_groups"] == 1
    origins = set()
    for value in clean["origin_sample_id"].astype(str):
        origins.update(value.split("|"))
    assert origins == {"tmc::a", "tmc::b"}


def test_identical_bbox_dedup_with_provenance():
    samples = pd.DataFrame(
        [
            _sample("tmc::a", "tmc_tongue", "m", "a.jpg"),
            _sample("tmc::b", "tmc_tongue", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    spatial = pd.DataFrame(
        [
            _bbox("tmc::a", "tmc_tongue", "features.red_spot.present", True, 1, 2, 3, 4),
            _bbox("tmc::b", "tmc_tongue", "features.red_spot.present", True, 1, 2, 3, 4),
        ]
    )
    clean, meta, stats = reconcile_spatial(spatial, decisions, POLICY)
    assert len(clean) == 1
    assert stats["identical_deduped"] == 1
    assert stats["multi_instance_groups"] == 0
    assert set(str(clean.iloc[0]["origin_sample_id"]).split("|")) == {"tmc::a", "tmc::b"}


def test_label_conflict_drops_only_conflicted_fact():
    samples = pd.DataFrame(
        [
            _sample("dx::a", "tonguedx", "m", "a.jpg"),
            _sample("dx::b", "tonguedx", "m", "b.jpg"),
        ]
    )
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label("dx::a", "tonguedx", "tongue_body.color", "pale", 1, "TonguePale"),
            _label("dx::a", "tonguedx", "features.crack.present", True, 1, "Crack"),
            _label("dx::b", "tonguedx", "tongue_body.color", "pale", 0, "TonguePale"),
            _label("dx::b", "tonguedx", "features.tooth_mark.present", True, 1, "Toothmark"),
        ]
    )
    clean, conflicts, _ = reconcile_labels(labels, decisions, POLICY)
    assert len(conflicts) == 1
    assert not (clean["canonical_label"].astype(str) == "pale").any()
    tasks = set(clean["canonical_task"].astype(str))
    assert "features.crack.present" in tasks
    assert "features.tooth_mark.present" in tasks


def test_unknown_conflict_policy_fail_fast(tmp_path):
    bad = Path("configs/cleaning_policy_v1.yaml").read_text(encoding="utf-8")
    bad = bad.replace(
        "conflict_policy: drop_conflicted_facts_from_clean",
        "conflict_policy: some_unknown_policy",
    )
    path = tmp_path / "bad_policy.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported conflict_policy"):
        CleaningPolicy(path)


def test_tonguexpert_l1_l2_pool_regression():
    samples = pd.DataFrame([_sample("tx::a", "tonguexpert", "m", "a.jpg")])
    decisions = select_canonical_samples(build_duplicate_groups(samples, POLICY), POLICY)
    labels = pd.DataFrame(
        [
            _label(
                "tx::a", "tonguexpert", "tongue_body.color", "dark", 1, "labels_zhi",
                source="human", tier="gold_candidate",
            ),
            _label(
                "tx::a", "tonguexpert", "tongue_body.color", "dark", 1, "zhi_label",
                source="model_prediction", tier="pseudo",
            ),
        ]
    )
    labels_clean, _, _ = reconcile_labels(labels, decisions, POLICY)
    assign = build_supervision_assignments(
        labels_clean, pd.DataFrame(), samples, decisions, POLICY
    )
    human = assign[assign["label_source"] == "human"]
    model = assign[assign["label_source"] == "model_prediction"]
    assert (human["supervision_pool"] == "gold_candidate").all()
    assert (model["supervision_pool"] == "pseudo").all()
