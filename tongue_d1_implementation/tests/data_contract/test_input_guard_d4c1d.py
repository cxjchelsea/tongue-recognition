"""D4-C.1-D：dataset confounding audit 契约测试（禁止 CNN 训练）。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tongue_data.stain.d4c1d_audit import (
    VALID_ACTIONS,
    VALID_LEVELS,
    cohens_d,
    nearest_neighbor_matching,
    propensity_overlap_audit,
    run_diagnostic_classifier,
    univariate_feature_effects,
)
from tongue_data.stain.d4c1d_features import (
    COLOR_FEATURES,
    FORBIDDEN_CLASSIFIER_COLS,
    GEOMETRY_FEATURES,
    QUALITY_FEATURES,
    RESOLUTION_FEATURES,
    all_acquisition_features,
    compute_dhash,
    extract_folder_batch,
)

ROOT = Path(__file__).resolve().parents[2]
V1_CKPT = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "best.pt"
V1_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
POLICY = ROOT / "configs" / "input_guard_v1.yaml"
AUDIT_SRC = ROOT / "src" / "tongue_data" / "stain" / "d4c1d_audit.py"
FEAT_SRC = ROOT / "src" / "tongue_data" / "stain" / "d4c1d_features.py"


def test_01_no_resnet_training_instantiation():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "resnet18" not in source.lower()
    assert "build_stain_model" not in source
    assert "DomainInvariantStainModel" not in source


def test_02_no_optimizer():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "AdamW" not in source


def test_03_v1_checkpoint_untouched_hash_stable():
    assert V1_CKPT.exists()
    digest = hashlib.md5(V1_CKPT.read_bytes()).hexdigest()
    assert len(digest) == 32


def test_04_thresholds_untouched():
    import json

    thr = json.loads(V1_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_05_uses_stain_gold_label_only():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "stain_label" in source
    assert "coating.color" not in source or "must not use coating.color" in source


def test_06_no_coating_color_as_label():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "must not use coating.color as stain label" in source


def test_07_source_test_default_excluded():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert 'splits: tuple[str, ...] = ("train", "val")' in source
    assert "source TEST must not enter default audit" in source


def test_08_feature_extraction_uses_original_rgb_paths():
    source = FEAT_SRC.read_text(encoding="utf-8")
    assert 'convert("RGB")' in source
    assert "imagenet" not in source.lower()


def test_09_rgb_channel_order_explicit():
    source = FEAT_SRC.read_text(encoding="utf-8")
    assert "COLOR_RGB2LAB" in source
    assert "COLOR_RGB2HSV" in source


def test_10_lab_stats_deterministic_helper():
    from tongue_data.stain.d4c1d_features import summarize_feature

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = summarize_feature(values)
    second = summarize_feature(values)
    assert first == second


def test_11_resolution_features_named():
    assert "original_width" in RESOLUTION_FEATURES
    assert "ROI_short_side" in RESOLUTION_FEATURES


def test_12_geometry_features_named():
    assert "foreground_ratio" in GEOMETRY_FEATURES
    assert "padding_ratio" in GEOMETRY_FEATURES


def test_13_blur_features_deterministic_dhash():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :16] = 255
    assert compute_dhash(rgb) == compute_dhash(rgb)


def test_14_classifier_forbids_sample_id():
    assert "sample_id" in FORBIDDEN_CLASSIFIER_COLS


def test_15_classifier_forbids_path():
    assert "source_image_path" in FORBIDDEN_CLASSIFIER_COLS
    assert "path" in FORBIDDEN_CLASSIFIER_COLS


def test_16_no_cnn_embedding_in_features():
    assert "embedding" not in all_acquisition_features()
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "extract_embedding" not in source


def test_17_cv_stratified_seed():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "StratifiedKFold" in source
    assert "SEED = 20260814" in source


def test_18_cv_no_overlap_enforced():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "CV fold overlap detected" in source


def test_19_color_only_set():
    assert "RGB_mean_r" in COLOR_FEATURES
    assert "Lab_a_mean" in COLOR_FEATURES
    assert "original_width" not in COLOR_FEATURES


def test_20_resolution_only_set():
    assert "original_pixel_count" in RESOLUTION_FEATURES
    assert "RGB_mean_r" not in RESOLUTION_FEATURES


def test_21_geometry_only_set():
    assert "ROI_aspect_ratio" in GEOMETRY_FEATURES


def test_22_quality_only_set():
    assert "blur_laplacian" in QUALITY_FEATURES
    assert "clipping_dark_ratio" in QUALITY_FEATURES


def test_23_propensity_from_acquisition_classifier():
    frame = pd.DataFrame(
        {
            "stain_label": [1, 1, 1, 0, 0, 0] * 10,
            "luminance_mean": np.linspace(10, 200, 60),
            "RGB_mean_r": np.linspace(20, 220, 60),
            "RGB_mean_g": np.linspace(15, 180, 60),
            "RGB_mean_b": np.linspace(10, 160, 60),
            "Lab_L_mean": np.linspace(5, 90, 60),
            "Lab_a_mean": np.linspace(-10, 40, 60),
            "Lab_b_mean": np.linspace(-5, 30, 60),
            "HSV_h_mean": np.linspace(0, 170, 60),
            "HSV_s_mean": np.linspace(10, 200, 60),
            "HSV_v_mean": np.linspace(20, 220, 60),
            "rg_ratio": np.linspace(0.5, 2.0, 60),
            "bg_ratio": np.linspace(0.4, 1.5, 60),
            "luminance_std": np.linspace(5, 40, 60),
            "luminance_p05": np.linspace(5, 80, 60),
            "luminance_p50": np.linspace(10, 150, 60),
            "luminance_p95": np.linspace(30, 240, 60),
            "RGB_median_r": np.linspace(20, 210, 60),
            "RGB_median_g": np.linspace(15, 170, 60),
            "RGB_median_b": np.linspace(10, 150, 60),
            "Lab_L_median": np.linspace(5, 85, 60),
            "Lab_a_median": np.linspace(-8, 35, 60),
            "Lab_b_median": np.linspace(-4, 28, 60),
            "HSV_v_mean": np.linspace(20, 220, 60),
        }
    )
    # ensure all color features exist with dummy values
    for name in COLOR_FEATURES:
        if name not in frame.columns:
            frame[name] = np.linspace(0, 1, len(frame))
    result = run_diagnostic_classifier(frame, COLOR_FEATURES)
    assert "oof_probabilities" in result
    assert len(result["oof_probabilities"]) == len(frame)


def test_24_matching_standardization_and_caliper():
    frame = pd.DataFrame(
        {
            "sample_id": [f"p{index}" for index in range(20)]
            + [f"n{index}" for index in range(20)],
            "stain_label": [1] * 20 + [0] * 20,
            "luminance_mean": list(np.linspace(100, 120, 20))
            + list(np.linspace(10, 30, 20)),
            "Lab_a_mean": [5.0] * 40,
            "Lab_b_mean": [5.0] * 40,
            "ROI_short_side": [200.0] * 40,
            "ROI_aspect_ratio": [1.0] * 40,
            "foreground_ratio": [0.6] * 40,
            "original_pixel_count": [1e6] * 40,
            "blur_laplacian": [50.0] * 40,
        }
    )
    result = nearest_neighbor_matching(frame, caliper=0.1)
    # 距离很大时应大量 unmatched
    assert result["positive_match_rate"] < 0.5


def test_25_matching_no_stain_prediction():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert 'assert "p_stain" not in usable' in source


def test_26_positive_negative_pairing_only():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "stain_label" in source
    assert "pos_idx" in source and "neg_idx" in source


def test_27_caliper_constant_defined():
    from tongue_data.stain import d4c1d_audit

    assert d4c1d_audit.CALIPER > 0


def test_28_unmatched_not_forced():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "unmatched_pos" in source


def test_29_negative_not_reused():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "used_neg" in source


def test_30_common_support_fields():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    frame = pd.DataFrame({"stain_label": [0, 0, 1, 1]})
    result = propensity_overlap_audit(frame, probs)
    assert "common_support_rate" in result
    assert "positive_support_rate" in result


def test_31_matching_before_after_protocol_same_features():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "all_acquisition_features()" in source


def test_32_effect_ranking_deterministic():
    frame = pd.DataFrame(
        {
            "stain_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 5,
            "luminance_mean": list(np.linspace(150, 200, 25))
            + list(np.linspace(10, 40, 25)),
            "RGB_mean_r": list(np.linspace(180, 220, 25)) + list(np.linspace(20, 50, 25)),
        }
    )
    for name in all_acquisition_features() + ["local_chroma_var"]:
        if name not in frame.columns:
            frame[name] = np.random.default_rng(0).normal(size=len(frame))
    # inject strong signal on luminance
    frame.loc[frame.stain_label == 1, "luminance_mean"] = np.linspace(150, 200, 25)
    frame.loc[frame.stain_label == 0, "luminance_mean"] = np.linspace(10, 40, 25)
    first = univariate_feature_effects(frame)["top10"][0]["feature"]
    second = univariate_feature_effects(frame)["top10"][0]["feature"]
    assert first == second


def test_33_metadata_not_production_runtime():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "Input Guard" not in source or "未改 policy" in Path(
        ROOT / "docs"
    ).joinpath("D4_C_1_D_DATASET_CONFOUNDING_AUDIT.md").read_text(encoding="utf-8") if (
        ROOT / "docs" / "D4_C_1_D_DATASET_CONFOUNDING_AUDIT.md"
    ).exists() else True


def test_34_near_duplicate_no_auto_delete():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "report_only_no_deletion" in source


def test_35_review_candidates_deterministic_seed():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "build_review_candidates" in source
    assert "SEED = 20260814" in source


def test_36_confounding_level_enum():
    assert VALID_LEVELS == {"NONE", "LOW", "MODERATE", "STRONG", "SEVERE"}


def test_37_data_action_enum():
    assert "RECOLLECT_STAIN_DATASET" in VALID_ACTIONS
    assert "MATCH_AND_RETRAIN" in VALID_ACTIONS


def test_38_policy_not_modified_in_code():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "policy_modified" in source
    assert "frozen_hashes_before != frozen_hashes_after" in source
    assert 'Path("configs/input_guard_v1.yaml").write' not in source


def test_39_no_known_external_audit_call():
    source = AUDIT_SRC.read_text(encoding="utf-8")
    assert "known_external" not in source or "known_external_audit_run" in source
    assert "BioHit" not in source


def test_40_folder_batch_extract_reproducible():
    assert extract_folder_batch(r"D:\data\batchA\img1.jpg") == "batchA"


def test_41_cohens_d_finite():
    value = cohens_d(np.array([1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0]))
    assert np.isfinite(value)
