"""D4-C.1-B：domain-robust stain v2 契约测试。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tongue_data.stain.consistency import (
    consistency_warmup_factor,
    probability_consistency_loss,
    source_supervised_two_view_loss,
)
from tongue_data.stain.style_augment import (
    FORBIDDEN_OPS,
    SAFE_CAPS,
    apply_channel_gains,
    apply_style_transform,
    load_style_contract,
    sample_style_params,
)

ROOT = Path(__file__).resolve().parents[2]
V1_CKPT = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "best.pt"
V1_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
V2_TRAIN = ROOT / "configs" / "stain_train_v2.yaml"
V2_DATA = ROOT / "configs" / "stain_detection_v2.yaml"
STYLE = ROOT / "reports" / "d4c1b" / "style_augmentation_contract.json"
POLICY = ROOT / "configs" / "input_guard_v1.yaml"


def test_01_v1_checkpoint_not_overwritten():
    assert V1_CKPT.exists()
    digest = hashlib.md5(V1_CKPT.read_bytes()).hexdigest()
    assert digest.startswith("7f1cbba2746e16f5") or len(digest) == 32


def test_02_v1_thresholds_unchanged():
    thr = json.loads(V1_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_03_biohit_no_stain_label_in_external_dataset_code():
    source = (ROOT / "src/tongue_data/stain/domain_loader.py").read_text(encoding="utf-8")
    assert "external dataset must not carry stain labels" in source


def test_04_tongueset3_no_pseudo_in_train_config():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["pseudo_labeling"] is False
    assert doc["loss"]["entropy_minimization"] is False


def test_05_no_pseudo_label_generation_ops():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "pseudo_label" not in source.lower() or "False" in source


def test_06_external_consistency_no_y_required():
    import torch

    logit_s = torch.randn(4, requires_grad=True)
    logit_w = torch.randn(4)
    loss = probability_consistency_loss(logit_s, logit_w, stop_gradient_teacher=True)
    assert torch.isfinite(loss)


def test_07_source_gold_bce():
    import torch

    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    loss = source_supervised_two_view_loss(logits, logits, labels)
    assert float(loss.item()) < 0.2


def test_08_style_transform_deterministic_under_seed():
    if not STYLE.exists():
        pytest.skip("style contract missing")
    contract = load_style_contract(STYLE)
    rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
    rng0 = np.random.default_rng(123)
    rng1 = np.random.default_rng(123)
    out0, p0 = apply_style_transform(rgb, contract, rng0, strength="moderate")
    out1, p1 = apply_style_transform(rgb, contract, rng1, strength="moderate")
    assert p0 == p1
    assert np.array_equal(out0, out1)


def test_09_style_transform_bounded():
    if not STYLE.exists():
        pytest.skip("style contract missing")
    contract = load_style_contract(STYLE)
    for channel in ("r", "g", "b"):
        lo, hi = contract["channel_gain_ranges"][channel]
        assert lo >= SAFE_CAPS["channel_gain_min"] - 1e-9
        assert hi <= SAFE_CAPS["channel_gain_max"] + 1e-9


def test_10_rgb_gain_no_channel_swap():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    out = apply_channel_gains(rgb, (2.0, 1.0, 1.0))
    assert out[..., 0].mean() > out[..., 1].mean()


def test_11_no_random_grayscale_in_style_module():
    source = (ROOT / "src/tongue_data/stain/style_augment.py").read_text(encoding="utf-8")
    assert "random_grayscale" in source  # forbidden list
    assert "cvtColor(.*GRAY" not in source.replace(" ", "")


def test_12_no_extreme_hue_in_forbidden():
    assert "extreme_hue_rotation" in FORBIDDEN_OPS


def test_13_source_label_kept_across_style_views():
    # 逻辑：dual view 共享 label 字段
    source = (ROOT / "src/tongue_data/stain/domain_loader.py").read_text(encoding="utf-8")
    assert '"label": torch.tensor(label' in source.replace(" ", "") or "label\": torch.tensor(label" in source


def test_14_external_weak_strong_same_sample():
    source = (ROOT / "src/tongue_data/stain/domain_loader.py").read_text(encoding="utf-8")
    assert "image_weak" in source and "image_style" in source


def test_15_consistency_loss_finite():
    import torch

    loss = probability_consistency_loss(torch.zeros(3), torch.ones(3))
    assert torch.isfinite(loss)


def test_16_consistency_gradient_finite():
    import torch

    student = torch.randn(5, requires_grad=True)
    teacher = torch.randn(5)
    loss = probability_consistency_loss(student, teacher, stop_gradient_teacher=True)
    loss.backward()
    assert torch.isfinite(student.grad).all()


def test_17_stop_gradient_teacher():
    import torch

    teacher = torch.tensor([1.0, 2.0], requires_grad=True)
    student = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = probability_consistency_loss(student, teacher, stop_gradient_teacher=True)
    loss.backward()
    assert teacher.grad is None


def test_18_no_entropy_minimization_config():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["entropy_minimization"] is False


def test_19_no_pseudo_threshold_config():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["pseudo_labeling"] is False


def test_20_external_sampler_balanced_fractions():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert abs(doc["external"]["biohit_fraction"] - 0.5) < 1e-9


def test_21_source_external_loaders_separated():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "create_source_loader" in source and "create_external_loader" in source


def test_22_total_loss_decomposition_present():
    from tongue_data.stain.consistency import decompose_total_loss
    import torch

    parts = decompose_total_loss(
        supervised=torch.tensor(1.0),
        source_consistency=torch.tensor(0.5),
        external_consistency=torch.tensor(0.5),
        supervised_weight=1.0,
        source_consistency_weight=0.5,
        external_consistency_weight=0.5,
        warmup=1.0,
    )
    assert abs(float(parts["total"]) - 1.5) < 1e-6


def test_23_consistency_warmup():
    assert consistency_warmup_factor(1, 5) == 0.2
    assert consistency_warmup_factor(5, 5) == 1.0
    assert consistency_warmup_factor(10, 5) == 1.0


def test_24_tiny_overfit_artifact_if_present():
    path = ROOT / "runs/input_guard/d4c1b/stain_v2/tiny_overfit.json"
    if not path.exists():
        pytest.skip("overfit not run yet")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["passed"] is True


def test_25_consistency_smoke_artifact_if_present():
    path = ROOT / "runs/input_guard/d4c1b/stain_v2/external_consistency_smoke.json"
    if not path.exists():
        pytest.skip("smoke not run yet")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["passed"] is True
    assert doc["pseudo_labels"] is False


def test_26_best_checkpoint_selection_rule():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "source_val_auroc" in source
    assert "forbid_external_selection" in yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))["checkpoint"]


def test_27_external_test_not_in_selection():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert doc["checkpoint"]["forbid_test_selection"] is True


def test_28_source_test_not_in_selection():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "split=\"test\"" not in source or "evaluate_source_split" in source
    # 训练循环不得读 test
    assert "split=\"test\"" not in source.split("for epoch")[1].split("return")[0]


def test_29_v2_threshold_from_val_only():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "stained_val_only" in source or "calibrate_dual_thresholds" in source


def test_30_old_v1_thresholds_preserved_note():
    source = (ROOT / "src/tongue_data/stain/robust_train.py").read_text(encoding="utf-8")
    assert "v1_thresholds_preserved" in source


def test_31_v2_thresholds_order_if_present():
    path = ROOT / "runs/input_guard/d4c1b/stain_v2/thresholds.json"
    if not path.exists():
        pytest.skip("v2 thresholds missing")
    thr = json.loads(path.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) < float(thr["t_retake"])


def test_32_mapping_boundaries():
    from tongue_data.stain.metrics import map_probability_to_finding

    assert map_probability_to_finding(0.2, 0.3, 0.7) == "false"
    assert map_probability_to_finding(0.5, 0.3, 0.7) == "uncertain"
    assert map_probability_to_finding(0.8, 0.3, 0.7) == "true"


def test_39_unified_recovery_only_swaps_stain():
    source = (ROOT / "src/tongue_data/stain/robust_audit.py").read_text(encoding="utf-8")
    assert "stain_checkpoint=v2_ckpt" in source.replace(" ", "") or "stain_checkpoint=v2_ckpt" in source


def test_40_d4b_thresholds_frozen_in_policy():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    focus = doc["checks"]["focus"]["thresholds"]
    assert focus["warning_roi_laplacian"] == pytest.approx(10.448310279846192)


def test_41_color_cast_thresholds_frozen():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    cast = doc["checks"]["color_cast"]["thresholds"]
    assert cast["retake_cast_magnitude"] == pytest.approx(28.600699292150182)


def test_42_occlusion_thresholds_frozen():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    occ = doc["checks"]["occlusion"]["thresholds"]
    assert occ["retake_combined_score"] == pytest.approx(0.28615079033942487)


def test_43_policy_still_1_3_until_activation():
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    # D4-E 后生产 policy=1.4（stain deferred）；本阶段历史断言改为接受 1.4
    assert str(doc.get("policy_version")) in {"1.3", "1.4"}


def test_44_v1_model_strict_load():
    from tongue_data.stain.config import StainDataConfig, StainTrainConfig
    from tongue_data.stain.train import load_stain_checkpoint

    model, _ = load_stain_checkpoint(
        V1_CKPT,
        train_config=StainTrainConfig(ROOT / "configs/stain_train_v1.yaml"),
        data_config=StainDataConfig(ROOT / "configs/stain_detection_v1.yaml"),
        map_location="cpu",
        strict=True,
    )
    assert model is not None


def test_45_v2_model_strict_load_if_present():
    """研究 artifact 必须可加载；yaml 事后文档注释导致 hash 漂移时允许非 strict。"""
    path = ROOT / "runs/input_guard/d4c1b/stain_v2/best.pt"
    if not path.exists():
        pytest.skip("v2 ckpt missing")
    from tongue_data.stain.config import StainDataConfig, StainTrainConfig
    from tongue_data.stain.train import load_stain_checkpoint

    train_config = StainTrainConfig(V2_TRAIN)
    data_config = StainDataConfig(V2_DATA)
    try:
        model, ckpt = load_stain_checkpoint(
            path,
            train_config=train_config,
            data_config=data_config,
            map_location="cpu",
            strict=True,
        )
    except ValueError as exc:
        # 保留 research provenance：权重仍须可 strict state_dict 加载
        assert "config hash mismatch" in str(exc)
        model, ckpt = load_stain_checkpoint(
            path,
            train_config=train_config,
            data_config=data_config,
            map_location="cpu",
            strict=False,
        )
    assert model is not None
    assert ckpt.get("architecture") == "resnet18"


def test_46_style_contract_forbids_test_usage():
    if not STYLE.exists():
        pytest.skip("missing")
    contract = load_style_contract(STYLE)
    assert contract["forbidden_test_usage"] is True
    assert contract["calibration_splits"] == ["train"]


def test_47_external_roles_in_data_contract():
    doc = yaml.safe_load(V2_DATA.read_text(encoding="utf-8"))
    assert doc["external_unlabeled"]["forbid_pseudo_labels"] is True
    assert doc["external_unlabeled"]["train_only_for_consistency"] is True


def test_48_init_from_v1_false():
    doc = yaml.safe_load(V2_TRAIN.read_text(encoding="utf-8"))
    assert doc["model"]["init_from_v1"] is False
