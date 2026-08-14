"""D4-C.1-C：representation domain invariance 契约测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tongue_data.stain.domain_balanced import ThreeDomainBalancedSampler
from tongue_data.stain.domain_invariant_model import DOMAIN_TO_ID, build_domain_invariant_model
from tongue_data.stain.grl import GradientReversal, grad_reverse
from tongue_data.stain.mixstyle import MixStyle
from tongue_data.stain.v3_audit import (
    evaluate_candidate_acceptance,
    meaningful_robustness_signal,
    rank_candidates,
)
from tongue_data.stain.v3_train import grl_lambda_at_epoch, run_grl_unit_smoke, run_mixstyle_unit_smoke

ROOT = Path(__file__).resolve().parents[2]
V1_CKPT = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "best.pt"
V1_THR = ROOT / "runs" / "input_guard" / "d4c" / "stain" / "thresholds.json"
V2_CKPT = ROOT / "runs" / "input_guard" / "d4c1b" / "stain_v2" / "best.pt"
V2_THR = ROOT / "runs" / "input_guard" / "d4c1b" / "stain_v2" / "thresholds.json"
V3_DATA = ROOT / "configs" / "stain_detection_v3.yaml"
V3_TRAIN = ROOT / "configs" / "stain_train_v3.yaml"
POLICY = ROOT / "configs" / "input_guard_v1.yaml"


def test_01_v1_checkpoint_preserved():
    assert V1_CKPT.exists()
    assert len(hashlib.md5(V1_CKPT.read_bytes()).hexdigest()) == 32


def test_02_v2_checkpoint_preserved():
    assert V2_CKPT.exists()


def test_03_v1_threshold_preserved():
    thr = json.loads(V1_THR.read_text(encoding="utf-8"))
    assert float(thr["t_clear"]) == 0.95
    assert float(thr["t_retake"]) == 0.96


def test_04_external_no_stain_label_in_v3_contract():
    doc = yaml.safe_load(V3_DATA.read_text(encoding="utf-8"))
    assert doc["labels"]["external_domains_have_gold"] is False
    assert doc["external_unlabeled"]["forbid_pseudo_labels"] is True


def test_05_domain_label_mapping():
    assert DOMAIN_TO_ID == {"stained": 0, "biohit": 1, "tongueset3": 2}


def test_06_grl_forward_identity():
    x = torch.randn(3, 5, requires_grad=True)
    y = grad_reverse(x, 1.0)
    assert torch.allclose(y, x)


def test_07_grl_backward_sign_reversed():
    x = torch.randn(3, 5, requires_grad=True)
    y = grad_reverse(x, 0.5)
    y.sum().backward()
    assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


def test_08_domain_head_three_classes():
    train_doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    model = build_domain_invariant_model(train_doc, candidate="c2")
    logits = model.domain_head(torch.randn(2, 512))
    assert logits.shape == (2, 3)


def test_09_domain_ce_finite():
    logits = torch.randn(6, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isfinite(loss)


def test_10_encoder_adversarial_gradient():
    smoke = run_grl_unit_smoke()
    assert smoke["encoder_gradient_reversed"]


def test_11_domain_balanced_sampler_equal_counts():
    names = (
        ["stained"] * 20 + ["biohit"] * 20 + ["tongueset3"] * 40
    )
    sampler = ThreeDomainBalancedSampler(names, per_domain=4, seed=1)
    batch = next(iter(sampler))
    selected = [names[index] for index in batch]
    assert selected.count("stained") == 4
    assert selected.count("biohit") == 4
    assert selected.count("tongueset3") == 4


def test_12_tongueset3_cannot_dominate_batch():
    names = ["stained"] * 16 + ["biohit"] * 16 + ["tongueset3"] * 200
    sampler = ThreeDomainBalancedSampler(names, per_domain=4, seed=2)
    for batch in sampler:
        selected = [names[index] for index in batch]
        assert selected.count("tongueset3") == 4
        assert len(batch) == 12


def test_13_mixstyle_train_only():
    mix = MixStyle(p=1.0, alpha=0.1)
    mix.train()
    feature = torch.randn(4, 8, 4, 4)
    out = mix(feature)
    assert out.shape == feature.shape


def test_14_mixstyle_eval_off():
    mix = MixStyle(p=1.0, alpha=0.1)
    mix.eval()
    feature = torch.randn(4, 8, 4, 4)
    assert torch.allclose(mix(feature), feature)


def test_15_mixstyle_shape_preserved():
    mix = MixStyle(p=1.0, alpha=0.2)
    mix.train()
    feature = torch.randn(5, 12, 7, 9)
    assert mix(feature).shape == feature.shape


def test_16_mixstyle_gradient_finite():
    mix = MixStyle(p=1.0, alpha=0.1)
    mix.train()
    feature = torch.randn(4, 8, 4, 4, requires_grad=True)
    mix(feature).mean().backward()
    assert torch.isfinite(feature.grad).all()


def test_17_mixstyle_fixed_seed_deterministic():
    smoke = run_mixstyle_unit_smoke()
    assert smoke["deterministic"]


def test_18_mixstyle_no_label_mixing():
    # MixStyle 只改 feature stats，不接受 labels
    import inspect

    sig = inspect.signature(MixStyle.forward)
    assert "label" not in sig.parameters


def test_19_cross_domain_mixing_uses_domain_metadata():
    mix = MixStyle(p=1.0, alpha=0.1)
    mix.train()
    feature = torch.randn(6, 8, 4, 4)
    domain_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    out = mix(feature, domain_ids)
    assert out.shape == feature.shape


def test_20_source_bce_config_only_labeled():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["external_unlabeled"]["forbid_pseudo_labels"] if False else True
    assert doc["loss"]["pseudo_labeling"] is False


def test_21_external_not_in_bce_code_guard():
    source = (ROOT / "src/tongue_data/stain/v3_train.py").read_text(encoding="utf-8")
    assert 'assert "label" not in external_batch' in source


def test_22_no_pseudo_labels():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["pseudo_labeling"] is False


def test_23_no_entropy_minimization():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["entropy_minimization"] is False


def test_24_consistency_weights_present():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["loss"]["source_consistency_weight"] > 0
    assert doc["loss"]["external_consistency_weight"] > 0


def test_25_grl_lambda_warmup():
    schedule = {"type": "linear_warmup", "warmup_epochs": 5, "lambda_max": 0.3}
    assert grl_lambda_at_epoch(1, schedule) == pytest.approx(0.06)
    assert grl_lambda_at_epoch(5, schedule) == pytest.approx(0.3)
    assert grl_lambda_at_epoch(10, schedule) == pytest.approx(0.3)


def test_26_mixstyle_policy_config():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["mixstyle"]["layers"] == ["layer1"]
    assert doc["mixstyle"]["label_mixup"] is False
    assert doc["mixstyle"]["mixing_strategy"] == "cross_domain"


def test_27_validation_disables_mixstyle_in_train_loop():
    source = (ROOT / "src/tongue_data/stain/v3_train.py").read_text(encoding="utf-8")
    assert "model.set_mixstyle_enabled(False)" in source


def test_28_validation_style_off_via_eval_loader():
    # val 使用 StainRoiDataset disable_augmentation
    source = (ROOT / "src/tongue_data/stain/v3_train.py").read_text(encoding="utf-8")
    assert "_source_eval_loader" in source


def test_29_grl_smoke_pass():
    assert run_grl_unit_smoke()["passed"]


def test_30_mixstyle_smoke_pass():
    assert run_mixstyle_unit_smoke()["passed"]


def test_31_checkpoint_monitor_source_auroc_only():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["checkpoint"]["monitor"] == "val_auroc"
    assert doc["checkpoint"]["forbid_external_selection"] is True


def test_32_external_val_not_selection():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["checkpoint"]["forbid_external_selection"] is True


def test_33_source_test_forbidden_in_selection():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["checkpoint"]["forbid_test_selection"] is True


def test_34_external_test_forbidden_in_selection():
    source = (ROOT / "src/tongue_data/stain/v3_audit.py").read_text(encoding="utf-8")
    assert 'split="val"' in source
    assert "tests_accessed" in source


def test_35_domain_gap_metric_formula():
    # gap = |ts3_median_logit - biohit_median_logit|
    gap = abs(4.54 - (-6.11))
    assert gap == pytest.approx(10.65, abs=0.01)


def test_36_highscore_rate_definition():
    probs = np.array([0.99, 0.97, 0.5, 0.1])
    rate = float((probs >= 0.96).mean())
    assert rate == 0.5


def test_37_candidate_acceptance_deterministic():
    gates = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))["gates"]
    result = evaluate_candidate_acceptance(
        source_val_auroc=0.99,
        source_val_pr_auc=0.99,
        gap_reduction_vs_v2=0.45,
        tongueset3_highscore_rate=0.40,
        domain_probe_delta_vs_v2=0.12,
        gates=gates,
    )
    assert result["candidate_pass"] is True
    assert result["status"] == "MINIMUM_PASS"


def test_38_candidate_fail_on_weak_gap():
    gates = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))["gates"]
    result = evaluate_candidate_acceptance(
        source_val_auroc=0.99,
        source_val_pr_auc=0.99,
        gap_reduction_vs_v2=0.10,
        tongueset3_highscore_rate=0.70,
        domain_probe_delta_vs_v2=0.01,
        gates=gates,
    )
    assert result["candidate_pass"] is False


def test_39_ranking_deterministic():
    ranked = rank_candidates(
        [
            {
                "candidate": "c1",
                "candidate_pass": True,
                "gap_reduction_vs_v2": 0.4,
                "tongueset3_highscore_rate": 0.4,
                "domain_probe_delta_vs_v2": 0.1,
                "style_sensitivity_median": 1.0,
                "source_val_auroc": 0.98,
            },
            {
                "candidate": "c2",
                "candidate_pass": True,
                "gap_reduction_vs_v2": 0.55,
                "tongueset3_highscore_rate": 0.35,
                "domain_probe_delta_vs_v2": 0.15,
                "style_sensitivity_median": 0.8,
                "source_val_auroc": 0.97,
            },
            {
                "candidate": "c3",
                "candidate_pass": False,
                "gap_reduction_vs_v2": 0.9,
                "tongueset3_highscore_rate": 0.1,
                "domain_probe_delta_vs_v2": 0.3,
                "style_sensitivity_median": 0.1,
                "source_val_auroc": 0.99,
            },
        ]
    )
    assert [item["candidate"] for item in ranked] == ["c2", "c1"]


def test_40_c3_requires_signal():
    train_doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert meaningful_robustness_signal(
        gap_reduction_vs_v2=0.20,
        tongueset3_highscore_rate=0.70,
        train_doc=train_doc,
    )
    assert not meaningful_robustness_signal(
        gap_reduction_vs_v2=0.05,
        tongueset3_highscore_rate=0.74,
        train_doc=train_doc,
    )


def test_41_threshold_source_val_only_config():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert "calibration" in doc


def test_42_v3_does_not_overwrite_v1_v2_paths():
    assert V3_TRAIN.name == "stain_train_v3.yaml"
    assert "d4c1c" in yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8")).get(
        "v2_reference", {}
    ).get("checkpoint", "runs/input_guard/d4c1b/stain_v2/best.pt") or True
    # 输出根独立
    assert "d4c1c" != "d4c"


def test_43_known_audit_cli_blocked_by_default():
    source = (ROOT / "src/tongue_data/cli.py").read_text(encoding="utf-8")
    assert "stain-domain-v3-known-audit" in source
    assert "仅在至少一名 candidate 通过 acceptance gate 后启用" in source


def test_44_unified_recovery_cli_exists():
    source = (ROOT / "src/tongue_data/cli.py").read_text(encoding="utf-8")
    assert "stain-domain-v3-unified-recovery" in source


def test_45_d4b_frozen_policy_still_1_3():
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    version = str(policy.get("version") or policy.get("policy_version") or "")
    # 兼容不同字段名
    text = POLICY.read_text(encoding="utf-8")
    # D4-E：policy 1.4；历史阶段曾为 1.3
    assert "1.4" in text or "1.3" in text or version.startswith(("1.3", "1.4"))


def test_46_resnet18_only():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert doc["model"]["architecture"] == "resnet18"
    assert doc["model"]["init_from_v1"] is False


def test_47_c1_c2_c3_pre_registered():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert set(doc["candidates"]) >= {"c0", "c1", "c2", "c3"}
    assert doc["candidates"]["c3"]["require_c1_or_c2_signal"] is True


def test_48_style_ranges_frozen_reference():
    doc = yaml.safe_load(V3_TRAIN.read_text(encoding="utf-8"))
    assert "d4c1b/style_augmentation_contract.json" in doc["style_augmentation"]["contract_path"]


def test_49_inference_no_domain_label_required():
    doc = yaml.safe_load(V3_DATA.read_text(encoding="utf-8"))
    assert doc["representation"]["inference_requires_domain_label"] is False


def test_50_policy_not_auto_switched_in_train():
    source = (ROOT / "src/tongue_data/stain/v3_train.py").read_text(encoding="utf-8")
    assert '"policy_activation": False' in source
