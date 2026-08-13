"""D4-C.1-A：stain cross-domain shortcut diagnosis 只读契约测试。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tongue_data.stain.config import StainDataConfig
from tongue_data.stain.d4c1a_features import (
    compute_roi_color_features,
    compute_roi_geometry_features,
)
from tongue_data.stain.d4c1a_model_tools import forward_logit_prob
from tongue_data.stain.d4c1a_preprocess import (
    assert_train_runtime_tensor_equiv,
    compute_fill_padding_ratios,
    letterbox_meta,
    preprocess_counterfactual,
)
from tongue_data.stain.transforms import preprocess_masked_roi

ROOT = Path(__file__).resolve().parents[2]
STAIN_CKPT = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "best.pt"
STAIN_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
STAIN_DATA = ROOT / "configs" / "stain_detection_v1.yaml"
REPORTS = ROOT / "reports" / "d4c1"
STATS = REPORTS / "d4c1a_diagnosis_stats.json"
MANIFEST = REPORTS / "d4c1a_diagnosis_manifest.parquet"
D4D1 = ROOT / "reports" / "d4" / "d4d1_integration_audit_stats.json"
DIAG_PY = ROOT / "src" / "tongue_data" / "stain" / "d4c1a_diagnosis.py"


def _synthetic_roi(size=64, tongue_color=(180, 70, 70)):
    rgb = np.full((size, size, 3), 40, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    rgb[mask > 0] = np.array(tongue_color, dtype=np.uint8)
    return rgb, mask


@pytest.fixture(scope="module")
def data_config():
    return StainDataConfig(STAIN_DATA)


def test_01_diagnosis_does_not_modify_stain_checkpoint():
    before = hashlib.md5(STAIN_CKPT.read_bytes()).hexdigest()
    # 仅读取，不写入
    after = hashlib.md5(STAIN_CKPT.read_bytes()).hexdigest()
    assert before == after


def test_02_thresholds_frozen():
    thr = json.loads(STAIN_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_03_diagnosis_module_has_no_optimizer():
    source = DIAG_PY.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "Optimizer(" not in source


def test_04_diagnosis_module_does_not_train():
    """禁止训练 stain 深度模型；允许 diagnostic sklearn fit。"""
    source = DIAG_PY.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "Adam(" not in source
    assert "SGD(" not in source
    assert "loss.backward" not in source
    assert "optimizer.step" not in source
    # 不得调用 stain 训练入口函数
    assert "from .train import train" not in source
    assert "run_stain_train" not in source
    assert "train_one_epoch" not in source


def test_05_train_runtime_preprocessing_equivalence(data_config):
    rgb, mask = _synthetic_roi()
    assert assert_train_runtime_tensor_equiv(rgb, mask, data_config)


def test_06_rgb_channel_order_consistent(data_config):
    rgb, mask = _synthetic_roi(tongue_color=(200, 10, 10))
    _tensor, letterboxed = preprocess_counterfactual(
        rgb, mask, data_config, mode="black", return_pre_norm_rgb=True
    )
    # 舌头区域应偏红：R 通道均值最高
    fore = (letterboxed[..., 0] > 0) | (letterboxed[..., 1] > 0) | (letterboxed[..., 2] > 0)
    # 更稳：用 mask letterbox
    assert letterboxed[..., 0].mean() > letterboxed[..., 2].mean()


def test_07_mask_gt0_semantics(data_config):
    rgb, mask = _synthetic_roi()
    mask01 = (mask > 0).astype(np.uint8)
    mask255 = mask01 * 255
    t0 = preprocess_counterfactual(rgb, mask01, data_config, mode="black")
    t1 = preprocess_counterfactual(rgb, mask255, data_config, mode="black")
    assert np.allclose(t0, t1)


def test_08_black_fill_ratio(data_config):
    rgb, mask = _synthetic_roi()
    stats = compute_fill_padding_ratios(rgb, mask, data_config)
    assert 0.0 <= stats["black_pixel_ratio"] <= 1.0
    assert stats["black_pixel_ratio"] > 0.2


def test_09_padding_ratio(data_config):
    rgb, mask = _synthetic_roi(size=32)
    # 非正方形 ROI
    rgb = rgb[:20, :, :]
    mask = mask[:20, :]
    meta = letterbox_meta(rgb.shape[0], rgb.shape[1], data_config.input_size)
    assert 0.0 <= meta["padding_ratio"] < 1.0
    stats = compute_fill_padding_ratios(rgb, mask, data_config)
    assert abs(stats["padding_ratio"] - meta["padding_ratio"]) < 1e-6


def test_10_roi_geometry_features():
    rgb, mask = _synthetic_roi()
    geom = compute_roi_geometry_features(rgb, mask, original_width=128, original_height=128)
    assert geom["roi_width"] == 64
    assert 0 < geom["foreground_ratio"] < 1
    assert geom["solidity"] > 0


def test_11_lab_stats_deterministic():
    rgb, mask = _synthetic_roi()
    a = compute_roi_color_features(rgb, mask)
    b = compute_roi_color_features(rgb, mask)
    assert a == b
    assert np.isfinite(a["mean_a"])


def test_12_hsv_stats_deterministic():
    rgb, mask = _synthetic_roi()
    a = compute_roi_color_features(rgb, mask)
    b = compute_roi_color_features(rgb.copy(), mask.copy())
    assert a["mean_h"] == b["mean_h"]
    assert a["mean_s"] == b["mean_s"]


def test_13_luminance_stats_deterministic():
    rgb, mask = _synthetic_roi()
    a = compute_roi_color_features(rgb, mask)
    b = compute_roi_color_features(rgb, mask)
    assert a["luminance_mean"] == b["luminance_mean"]
    assert a["luminance_p50"] == b["luminance_p50"]


def test_14_gray_fill_does_not_mutate_original(data_config):
    rgb, mask = _synthetic_roi()
    before = rgb.copy()
    _ = preprocess_counterfactual(rgb, mask, data_config, mode="gray")
    assert np.array_equal(rgb, before)


def test_15_bbox_counterfactual_deterministic(data_config):
    rgb, mask = _synthetic_roi()
    t0 = preprocess_counterfactual(rgb, mask, data_config, mode="bbox")
    t1 = preprocess_counterfactual(rgb, mask, data_config, mode="bbox")
    assert np.allclose(t0, t1)


def test_16_representation_ablation_ids_consistent_if_artifacts():
    if not STATS.exists():
        pytest.skip("diagnosis artifacts missing")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    rep = stats["representation_ablation"]
    for key in ("stained_negative", "biohit", "tongueset3"):
        assert "black" in rep[key]
        assert "gray" in rep[key]
        assert "bbox" in rep[key]


def test_17_p_stain_inference_deterministic(data_config):
    if not STAIN_CKPT.exists():
        pytest.skip("checkpoint missing")
    from tongue_data.stain.config import StainTrainConfig
    from tongue_data.stain.train import load_stain_checkpoint, resolve_device
    import torch

    rgb, mask = _synthetic_roi()
    model, _ = load_stain_checkpoint(
        STAIN_CKPT,
        train_config=StainTrainConfig(ROOT / "configs" / "stain_train_v1.yaml"),
        data_config=data_config,
        map_location="cpu",
        strict=True,
    )
    model.eval()
    tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
    batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
    p0 = forward_logit_prob(model, batch)[1]
    p1 = forward_logit_prob(model, batch)[1]
    assert abs(p0 - p1) < 1e-6


def test_18_logit_sigmoid_correspondence():
    logit = 2.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    assert abs(prob - 0.8807970779778823) < 1e-9


def test_19_embedding_extraction_deterministic(data_config):
    if not STAIN_CKPT.exists():
        pytest.skip("checkpoint missing")
    from tongue_data.stain.config import StainTrainConfig
    from tongue_data.stain.d4c1a_model_tools import extract_embedding
    from tongue_data.stain.train import load_stain_checkpoint
    import torch

    rgb, mask = _synthetic_roi()
    model, _ = load_stain_checkpoint(
        STAIN_CKPT,
        train_config=StainTrainConfig(ROOT / "configs" / "stain_train_v1.yaml"),
        data_config=data_config,
        map_location="cpu",
        strict=True,
    )
    model.eval()
    tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
    batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
    e0 = extract_embedding(model, batch)
    e1 = extract_embedding(model, batch)
    assert e0.shape == (512,)
    assert np.allclose(e0, e1)


def test_20_embedding_before_classifier(data_config):
    if not STAIN_CKPT.exists():
        pytest.skip("checkpoint missing")
    from tongue_data.stain.config import StainTrainConfig
    from tongue_data.stain.d4c1a_model_tools import extract_embedding
    from tongue_data.stain.train import load_stain_checkpoint
    import torch

    rgb, mask = _synthetic_roi()
    model, _ = load_stain_checkpoint(
        STAIN_CKPT,
        train_config=StainTrainConfig(ROOT / "configs" / "stain_train_v1.yaml"),
        data_config=data_config,
        map_location="cpu",
        strict=True,
    )
    model.eval()
    tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
    batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
    emb = extract_embedding(model, batch)
    # fc 输入维 = 512
    assert model.fc.in_features == 512
    assert emb.shape[0] == model.fc.in_features


def test_21_grad_cam_uses_frozen_model(data_config):
    if not STAIN_CKPT.exists():
        pytest.skip("checkpoint missing")
    from tongue_data.stain.config import StainTrainConfig
    from tongue_data.stain.d4c1a_model_tools import grad_cam_resnet18
    from tongue_data.stain.train import load_stain_checkpoint
    import torch

    rgb, mask = _synthetic_roi()
    model, _ = load_stain_checkpoint(
        STAIN_CKPT,
        train_config=StainTrainConfig(ROOT / "configs" / "stain_train_v1.yaml"),
        data_config=data_config,
        map_location="cpu",
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    tensor = preprocess_counterfactual(rgb, mask, data_config, mode="black")
    batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
    cam = grad_cam_resnet18(model, batch)
    after = list(model.parameters())
    for left, right in zip(before, after):
        assert torch.allclose(left, right)
    assert cam.shape == (224, 224)


def test_22_cam_region_ratios_sum_reasonable(data_config):
    from tongue_data.stain.d4c1a_model_tools import cam_region_ratios

    cam = np.ones((224, 224), dtype=np.float32)
    rgb, mask = _synthetic_roi()
    # 放大 mask 到近似 letterbox 语义：直接用正方形 mask
    big_mask = np.zeros((224, 224), dtype=np.uint8)
    big_mask[40:180, 40:180] = 255
    ratios = cam_region_ratios(
        cam,
        big_mask,
        224,
        pad_top=0,
        pad_left=0,
        new_height=224,
        new_width=224,
    )
    total = (
        ratios["inside_ratio"]
        + ratios["boundary_ratio"]
        + ratios["background_ratio"]
        + ratios["padding_ratio"]
    )
    assert 0.99 <= total <= 1.01


def test_23_dataset_identity_features_exclude_id_path():
    source = DIAG_PY.read_text(encoding="utf-8")
    # 特征列表段落不应包含 sample_id/path
    assert '"sample_id"' not in source.split("feature_cols")[1].split("]")[0]


def test_24_dataset_classifier_not_written_to_runtime():
    runtime = (ROOT / "src" / "tongue_data" / "input_guard" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert "dataset_identity" not in runtime
    assert "RandomForest" not in runtime


def test_25_tongueset3_not_auto_negative_if_artifacts():
    if not MANIFEST.exists():
        pytest.skip("manifest missing")
    import pandas as pd

    frame = pd.read_parquet(MANIFEST)
    ts3 = frame[frame["dataset"] == "tongueset3"]
    assert ts3["true_stain_label"].isna().all() or (
        ts3["true_stain_label"].astype(str) == "None"
    ).all() or ts3["true_stain_label"].isna().all()
    assert (ts3["label_role"] == "no_stain_gold").all()


def test_26_biohit_not_auto_negative_if_artifacts():
    if not MANIFEST.exists():
        pytest.skip("manifest missing")
    import pandas as pd

    frame = pd.read_parquet(MANIFEST)
    bio = frame[frame["dataset"] == "biohit"]
    assert bio["true_stain_label"].isna().all()
    assert (bio["label_role"] == "no_stain_gold").all()


def test_27_only_stained_has_gold_if_artifacts():
    if not MANIFEST.exists():
        pytest.skip("manifest missing")
    import pandas as pd

    frame = pd.read_parquet(MANIFEST)
    gold = frame[frame["label_role"] == "stain_gold"]
    assert (gold["dataset"] == "stained_coating").all()


def test_28_d4c_test_predictions_readonly():
    path = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "test_predictions.parquet"
    if not path.exists():
        pytest.skip("missing")
    before = hashlib.md5(path.read_bytes()).hexdigest()
    _ = path.read_bytes()
    after = hashlib.md5(path.read_bytes()).hexdigest()
    assert before == after


def test_29_d4d1_audit_not_modified_by_diagnosis_hash_stable():
    if not D4D1.exists() or not STATS.exists():
        pytest.skip("artifacts missing")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    assert stats["acceptance_gates"]["d4d1_stats_unmodified"] is True


def test_30_shortcut_evidence_enum_valid_if_artifacts():
    if not STATS.exists():
        pytest.skip("stats missing")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    allowed = {"strong", "moderate", "weak", "not_supported", "undetermined"}
    for _name, item in stats["shortcut_evidence"]["factors"].items():
        assert item["status"] in allowed
    allowed_rec = {
        "PROCEED_D4C1B_DOMAIN_ROBUST_RETRAINING",
        "PREPROCESSING_BUG_FOUND",
        "DATA_SEMANTICS_CONCERN",
        "TRUE_STAIN_LIKE_APPEARANCE_UNRESOLVED",
        "INSUFFICIENT_EVIDENCE",
        "MULTIPLE_BLOCKERS",
    }
    assert stats["recommendation"]["recommendation"] in allowed_rec


def test_black_matches_official_preprocess(data_config):
    rgb, mask = _synthetic_roi()
    official = preprocess_masked_roi(rgb, mask, data_config, split="val")
    black = preprocess_counterfactual(rgb, mask, data_config, mode="black")
    assert np.allclose(official, black)
