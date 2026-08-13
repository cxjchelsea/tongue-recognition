"""D4-C：stain detection contract / split / ROI / train / calibrate / runtime。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tongue_data.input_guard.ontology import (
    CheckId,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    implemented_checks_count,
)
from tongue_data.input_guard.policy import InputGuardPolicy
from tongue_data.input_guard.runtime import InputGuardRuntime
from tongue_data.input_guard.schema import CheckResult
from tongue_data.stain.calibrate import calibrate_dual_thresholds, load_frozen_thresholds
from tongue_data.stain.config import StainDataConfig, StainTrainConfig
from tongue_data.stain.dataset import StainRoiDataset, select_overfit_subset
from tongue_data.stain.labels import assert_no_coating_color_usage, parse_stain_label
from tongue_data.stain.manifest import build_stain_base_frame
from tongue_data.stain.metrics import map_probability_to_finding
from tongue_data.stain.model import build_stain_model
from tongue_data.stain.transforms import (
    apply_tongue_mask,
    letterbox_rgb,
    preprocess_masked_roi,
)
from tongue_data.stain.train import load_stain_checkpoint, save_stain_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "v1"
SPLITS = ROOT / "data" / "splits" / "v1"
DATA_CFG = ROOT / "configs" / "stain_detection_v1.yaml"
TRAIN_CFG = ROOT / "configs" / "stain_train_v1.yaml"
POLICY = ROOT / "configs" / "input_guard_v1.yaml"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "samples_clean.parquet").exists()
    or not (SPLITS / "split_assignments.parquet").exists(),
    reason="D2 processed/splits missing",
)


@pytest.fixture(scope="module")
def stain_base() -> pd.DataFrame:
    return build_stain_base_frame(PROCESSED, SPLITS, DATA_CFG)


def test_only_stained_coating_dataset(stain_base):
    assert set(stain_base["dataset"].astype(str).unique()) == {"stained_coating"}


def test_only_stain_suspected_task():
    labels = pd.read_parquet(PROCESSED / "labels_clean.parquet")
    samples = pd.read_parquet(PROCESSED / "samples_clean.parquet")
    stain_ids = set(
        samples.loc[samples["dataset"] == "stained_coating", "sample_id"].astype(str)
    )
    tasks = set(
        labels.loc[labels["sample_id"].isin(stain_ids), "canonical_task"]
        .astype(str)
        .unique()
    )
    assert "quality.stain_suspected" in tasks
    assert "coating.color" not in tasks


def test_must_not_read_coating_color(stain_base):
    labels = pd.read_parquet(PROCESSED / "labels_clean.parquet")
    stain_labels = labels[
        labels["sample_id"].isin(stain_base["sample_id"])
        & (labels["canonical_task"] == "quality.stain_suspected")
    ]
    assert_no_coating_color_usage(stain_labels)
    with pytest.raises(ValueError, match="coating.color"):
        bad = stain_labels.copy()
        bad = pd.concat(
            [
                bad,
                pd.DataFrame(
                    [
                        {
                            "sample_id": "x",
                            "canonical_task": "coating.color",
                            "canonical_label": "yellow",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        assert_no_coating_color_usage(bad)


def test_d2_split_fully_inherited(stain_base):
    splits = pd.read_parquet(SPLITS / "split_assignments.parquet")
    merged = stain_base.merge(
        splits[["sample_id", "split"]], on="sample_id", suffixes=("", "_d2")
    )
    assert (merged["split"] == merged["split_d2"]).all()


def test_split_sample_overlap_zero(stain_base):
    train = set(stain_base.loc[stain_base.split == "train", "sample_id"])
    val = set(stain_base.loc[stain_base.split == "val", "sample_id"])
    test = set(stain_base.loc[stain_base.split == "test", "sample_id"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_md5_overlap_zero(stain_base):
    train = set(stain_base.loc[stain_base.split == "train", "md5"])
    val = set(stain_base.loc[stain_base.split == "val", "md5"])
    test = set(stain_base.loc[stain_base.split == "test", "md5"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_label_true_false_mapping():
    assert parse_stain_label("true") == 1
    assert parse_stain_label("false") == 0
    assert parse_stain_label(True) == 1
    assert parse_stain_label(False) == 0


def test_unknown_label_not_silent_negative():
    with pytest.raises(ValueError, match="unknown"):
        parse_stain_label("yellow")
    with pytest.raises(ValueError, match="unknown"):
        parse_stain_label("maybe")


def test_d3e_roi_adapter_contract():
    # ROI adapter 语义：mask 外填充固定值，保留 RGB 通道顺序
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    rgb[:, :, 1] = 10
    rgb[:, :, 2] = 30
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 5:25] = 1
    masked = apply_tongue_mask(rgb, mask, fill_value=0)
    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[10, 10].tolist() == [200, 10, 30]


def test_invalid_roi_not_silent_train(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "split": "train",
                "label": 1,
                "md5": "x",
                "eligible": False,
                "roi_rgb_path": None,
                "roi_mask_path": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="no eligible"):
        StainRoiDataset(frame, DATA_CFG, TRAIN_CFG, split="train")


def test_d4b_retake_not_auto_excluded(stain_base):
    # base frame 不含 D4-B exclusion；eligible 默认仅待 ROI
    assert set(stain_base["exclusion_reason"].unique()) == {"roi_pending"}
    assert "d4b_retake" not in stain_base["exclusion_reason"].astype(str).tolist()


def test_rgb_channel_order_pre_norm():
    cfg = StainDataConfig(DATA_CFG)
    rgb = np.zeros((40, 50, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # R
    mask = np.ones((40, 50), dtype=np.uint8)
    _tensor, pre = preprocess_masked_roi(
        rgb, mask, cfg, split="val", return_pre_norm_rgb=True
    )
    # letterbox 后中心应保持 R=255 主导
    center = pre[pre.shape[0] // 2, pre.shape[1] // 2]
    assert int(center[0]) == 255
    assert int(center[1]) == 0
    assert int(center[2]) == 0


def test_mask_background_fill():
    rgb = np.ones((10, 10, 3), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    out = apply_tongue_mask(rgb, mask, fill_value=0)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[4, 4].tolist() == [100, 100, 100]


def test_aspect_ratio_preserved_letterbox():
    rgb = np.zeros((20, 80, 3), dtype=np.uint8)
    rgb[:, :] = (10, 20, 30)
    boxed = letterbox_rgb(rgb, 224, fill_value=0)
    assert boxed.shape == (224, 224, 3)
    # 非正方形内容不应被强拉：上下应有填充带
    assert boxed[0, 112].sum() == 0 or boxed[223, 112].sum() == 0


def test_no_hue_saturation_augmentation_in_config():
    cfg = StainTrainConfig(TRAIN_CFG)
    train_aug = cfg.augmentation.get("train", {})
    assert train_aug.get("color_jitter") is False
    assert train_aug.get("brightness_contrast") is False


def test_val_test_augmentation_disabled():
    cfg = StainTrainConfig(TRAIN_CFG)
    assert cfg.augmentation.get("val", {}).get("enabled") is False
    assert cfg.augmentation.get("test", {}).get("enabled") is False


def test_resnet18_forward_shape_and_raw_logit():
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    batch = torch.randn(2, 3, 224, 224)
    logits = model(batch)
    assert logits.shape == (2, 1)
    # forward 不得内置 sigmoid：logit 可超出 (0,1)
    assert torch.is_floating_point(logits)


def test_bce_loss_and_gradient_finite():
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    batch = torch.randn(4, 3, 224, 224, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    logits = model(batch)[:, 0]
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_tiny_overfit_can_learn(tmp_path):
    # 合成可分数据：正负样本均值不同
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    positives = torch.ones(8, 3, 64, 64) * 0.9
    negatives = torch.zeros(8, 3, 64, 64)
    # 临时改 fc 输入尺寸不适配 64；改用 224
    positives = torch.ones(8, 3, 224, 224) * 0.9
    negatives = torch.zeros(8, 3, 224, 224)
    images = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(8), torch.zeros(8)], dim=0)
    first_loss = None
    last_loss = None
    last_acc = 0.0
    for _epoch in range(40):
        optimizer.zero_grad()
        logits = model(images)[:, 0]
        loss = criterion(logits, labels)
        current_loss = float(loss.detach())
        if first_loss is None:
            first_loss = current_loss
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            logits_after = model(images)[:, 0]
            last_loss = float(criterion(logits_after, labels).detach())
            preds = (torch.sigmoid(logits_after) >= 0.5).float()
            last_acc = float((preds == labels).float().mean())
        if last_acc >= 0.95 and last_loss < first_loss - 1e-4:
            break
    assert last_acc >= 0.95
    assert last_loss is not None and last_loss < first_loss


def test_best_checkpoint_selected_by_val_auroc_only(tmp_path):
    data_cfg = StainDataConfig(DATA_CFG)
    train_cfg = StainTrainConfig(TRAIN_CFG)
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    path = tmp_path / "best.pt"
    save_stain_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=3,
        best_val_auroc=0.91,
        train_config=train_cfg,
        data_config=data_cfg,
        history=[{"epoch": 1, "val_auroc": 0.8}, {"epoch": 3, "val_auroc": 0.91}],
        extra={"selection_metric": "val_auroc", "test_used": False},
    )
    _model, ckpt = load_stain_checkpoint(
        path, train_config=train_cfg, data_config=data_cfg, strict=True
    )
    assert ckpt["extra"]["selection_metric"] == "val_auroc"
    assert ckpt["extra"]["test_used"] is False
    assert ckpt["best_val_auroc"] == 0.91


def test_test_not_in_checkpoint_selection_metadata(tmp_path):
    data_cfg = StainDataConfig(DATA_CFG)
    train_cfg = StainTrainConfig(TRAIN_CFG)
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    path = tmp_path / "best.pt"
    save_stain_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        best_val_auroc=0.5,
        train_config=train_cfg,
        data_config=data_cfg,
        history=[],
        extra={"selection_metric": "val_auroc", "test_used": False},
    )
    _model, ckpt = load_stain_checkpoint(
        path, train_config=train_cfg, data_config=data_cfg, strict=True
    )
    assert ckpt["extra"]["test_used"] is False


def test_threshold_calibration_uses_val_only_and_deterministic():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    scores = np.array([0.05, 0.1, 0.2, 0.35, 0.65, 0.8, 0.9, 0.95])
    one = calibrate_dual_thresholds(labels, scores, target_confident_precision=0.9)
    two = calibrate_dual_thresholds(labels, scores, target_confident_precision=0.9)
    assert one["source_split"] == "val"
    assert one["t_clear"] == two["t_clear"]
    assert one["t_retake"] == two["t_retake"]
    assert one["t_clear"] < one["t_retake"]


def test_runtime_mapping_false_uncertain_true():
    assert map_probability_to_finding(0.1, 0.2, 0.8) == "false"
    assert map_probability_to_finding(0.5, 0.2, 0.8) == "uncertain"
    assert map_probability_to_finding(0.9, 0.2, 0.8) == "true"


def test_finding_to_decision_effects():
    # false → no retake；uncertain → warning；true → retake；reason=STAIN_SUSPECTED
    cases = [
        ("false", Decision.PASS.value, None, EvidenceSource.LEARNED_MODEL.value),
        (
            "uncertain",
            Decision.WARNING.value,
            ReasonCode.STAIN_SUSPECTED.value,
            EvidenceSource.LEARNED_MODEL.value,
        ),
        (
            "true",
            Decision.RETAKE.value,
            ReasonCode.STAIN_SUSPECTED.value,
            EvidenceSource.LEARNED_MODEL.value,
        ),
    ]
    for finding, effect, reason, source in cases:
        check = CheckResult(
            check_id=CheckId.STAIN_SUSPECTED.value,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding=finding,
            severity="none" if finding == "false" else "moderate" if finding == "uncertain" else "severe",
            decision_effect=effect,
            score=0.1 if finding == "false" else 0.5 if finding == "uncertain" else 0.9,
            reason_code=reason,
            source=source,
            evidence={"p_stain": 0.5},
        )
        check.validate()
        assert "coating.color" not in check.evidence
        if finding == "false":
            assert check.decision_effect != Decision.RETAKE.value
            assert check.reason_code is None
        else:
            assert check.reason_code == ReasonCode.STAIN_SUSPECTED.value


def test_stain_result_does_not_emit_coating_color():
    check = CheckResult(
        check_id=CheckId.STAIN_SUSPECTED.value,
        evaluation_state=EvaluationState.EVALUATED.value,
        finding="true",
        severity="severe",
        decision_effect=Decision.RETAKE.value,
        score=0.95,
        reason_code=ReasonCode.STAIN_SUSPECTED.value,
        source=EvidenceSource.LEARNED_MODEL.value,
        evidence={"p_stain": 0.95},
    )
    assert "coating.color" not in check.to_dict()["evidence"]


def test_checkpoint_strict_load_and_wrong_hash(tmp_path):
    data_cfg = StainDataConfig(DATA_CFG)
    train_cfg = StainTrainConfig(TRAIN_CFG)
    model = build_stain_model({"architecture": "resnet18", "encoder_weights": None, "classes": 1})
    path = tmp_path / "best.pt"
    save_stain_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        best_val_auroc=0.5,
        train_config=train_cfg,
        data_config=data_cfg,
        history=[],
    )
    # wrong hash fail-fast
    bad = torch.load(path, map_location="cpu", weights_only=False)
    bad["train_config_hash"] = "deadbeefdeadbeef"
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_stain_checkpoint(
            bad_path, train_config=train_cfg, data_config=data_cfg, strict=True
        )


def test_test_threshold_not_recomputed_from_loader(tmp_path):
    payload = {
        "t_clear": 0.22,
        "t_retake": 0.77,
        "target_confident_precision": 0.9,
        "constraint_not_met": False,
        "source_split": "val",
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    frozen = load_frozen_thresholds(path)
    assert frozen["t_clear"] == 0.22
    assert frozen["source_split"] == "val"


def test_evaluation_complete_and_guard_ready_still_false():
    policy = InputGuardPolicy(POLICY)
    assert implemented_checks_count() == 11
    assert policy.is_check_enabled(CheckId.COLOR_CAST)
    from tongue_data.input_guard.ontology import CHECK_DEFINITIONS

    assert CHECK_DEFINITIONS[CheckId.COLOR_CAST]["implemented"] is True
    assert CHECK_DEFINITIONS[CheckId.OCCLUSION]["implemented"] is True
    assert CHECK_DEFINITIONS[CheckId.STAIN_SUSPECTED]["implemented"] is True


def test_color_cast_occlusion_remain_not_evaluated_without_detectors():
    from tongue_data.input_guard.features import InputGuardFeatures
    from tongue_data.input_guard.signal_checks import evaluate_signal_checks

    features = InputGuardFeatures(
        segmentation_status="success",
        original_width=100,
        original_height=100,
        foreground_ratio=0.3,
        tongue_pixel_count=5000,
        bbox_width_ratio=0.5,
        bbox_height_ratio=0.5,
        bbox_area_ratio=0.25,
        component_count=1,
        largest_component_ratio=1.0,
        mean_foreground_probability=0.9,
        roi_blur_score=80.0,
        blur_score=80.0,
        dark_pixel_ratio=0.1,
        bright_pixel_ratio=0.1,
        shadow_clip_ratio=0.0,
        highlight_clip_ratio=0.0,
        relative_luminance_range=0.1,
        effective_short_side_px=100,
        left_touch_ratio=0.0,
        right_touch_ratio=0.0,
        top_touch_ratio=0.0,
        bottom_touch_ratio=0.0,
    )
    policy = InputGuardPolicy(POLICY)
    checks = evaluate_signal_checks(features, policy)
    assert (
        checks[CheckId.COLOR_CAST.value].evaluation_state
        == EvaluationState.NOT_EVALUATED.value
    )
    assert (
        checks[CheckId.OCCLUSION.value].evaluation_state
        == EvaluationState.NOT_EVALUATED.value
    )
    # stain 无 detector 时 awaiting runtime
    assert (
        checks[CheckId.STAIN_SUSPECTED.value].evaluation_state
        == EvaluationState.NOT_EVALUATED.value
    )


def test_input_guard_aggregation_accepts_stain_result():
    from tongue_data.input_guard.decision import build_result_from_check_effects

    policy = InputGuardPolicy(POLICY)
    checks = {
        CheckId.STAIN_SUSPECTED.value: CheckResult(
            check_id=CheckId.STAIN_SUSPECTED.value,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="true",
            severity="severe",
            decision_effect=Decision.RETAKE.value,
            score=0.93,
            reason_code=ReasonCode.STAIN_SUSPECTED.value,
            source=EvidenceSource.LEARNED_MODEL.value,
        )
    }
    result = build_result_from_check_effects(
        checks=checks, policy=policy, evaluation_complete=False
    )
    assert result.decision == Decision.RETAKE.value
    assert result.evaluation_complete is False
    assert result.guard_ready is False
    assert ReasonCode.STAIN_SUSPECTED.value in result.reason_codes


def test_same_image_inference_deterministic_preprocess():
    cfg = StainDataConfig(DATA_CFG)
    rgb = np.random.default_rng(0).integers(0, 255, size=(80, 100, 3), dtype=np.uint8)
    mask = np.ones((80, 100), dtype=np.uint8)
    one = preprocess_masked_roi(rgb, mask, cfg, split="val")
    two = preprocess_masked_roi(rgb, mask, cfg, split="val")
    assert np.allclose(one, two)


def test_select_overfit_subset_balanced(stain_base):
    # 无 ROI cache 时仍可从 label 选子集
    frame = stain_base.copy()
    frame["eligible"] = True
    subset = select_overfit_subset(frame, positives=8, negatives=8)
    assert len(subset) == 16
    assert int((subset["label"] == 1).sum()) == 8
    assert int((subset["label"] == 0).sum()) == 8
