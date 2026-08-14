import argparse
import json
from pathlib import Path

from .manifest import ManifestBuilder
from .validators import validate_contract, validate_manifest
from .cleaning import CleaningBuilder, validate_clean
from .splitting import SplitBuilder, validate_split
from .segmentation import SegmentationBuilder, validate_segmentation
from .segmentation.dataset import smoke_test_dataset
from .segmentation.training import (
    evaluate_checkpoint_on_split,
    preflight_full_training,
    run_full_training,
    run_smoke_training,
    run_tiny_overfit,
)


def emit(errors, warnings):
    for item in warnings:
        print(f"[WARN] {item}")
    for item in errors:
        print(f"[ERROR] {item}")
    if not errors:
        print("OK")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(prog="tongue-data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate_contract_parser = sub.add_parser("validate-contract")
    validate_contract_parser.add_argument(
        "--ontology", default="ontology/tongue_phenotype_v1.yaml"
    )
    validate_contract_parser.add_argument(
        "--mappings-dir", default="ontology/mappings"
    )
    validate_contract_parser.add_argument("--strict", action="store_true")

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument("--output", required=True)

    validate_manifest_parser = sub.add_parser("validate-manifest")
    validate_manifest_parser.add_argument("--manifest-dir", required=True)

    # D2-A
    clean_parser = sub.add_parser("clean")
    clean_parser.add_argument("--manifest-dir", required=True)
    clean_parser.add_argument("--policy", required=True)
    clean_parser.add_argument("--output", required=True)
    clean_parser.add_argument("--report-dir", required=True)

    validate_clean_parser = sub.add_parser("validate-clean")
    validate_clean_parser.add_argument("--processed-dir", required=True)
    validate_clean_parser.add_argument("--policy", default=None)

    # D2-B/C
    build_groups_parser = sub.add_parser("build-groups")
    build_groups_parser.add_argument("--processed-dir", required=True)
    build_groups_parser.add_argument("--manifest-dir", default=None)
    build_groups_parser.add_argument("--policy", required=True)
    build_groups_parser.add_argument("--output", required=True)
    build_groups_parser.add_argument("--report-dir", required=True)

    split_parser = sub.add_parser("split")
    split_parser.add_argument("--processed-dir", required=True)
    split_parser.add_argument("--groups", default=None)
    split_parser.add_argument("--policy", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--report-dir", required=True)

    build_splits_parser = sub.add_parser("build-splits")
    build_splits_parser.add_argument("--processed-dir", required=True)
    build_splits_parser.add_argument("--manifest-dir", default=None)
    build_splits_parser.add_argument("--policy", required=True)
    build_splits_parser.add_argument("--output", required=True)
    build_splits_parser.add_argument("--report-dir", required=True)

    validate_split_parser = sub.add_parser("validate-split")
    validate_split_parser.add_argument("--split-dir", required=True)
    validate_split_parser.add_argument("--processed-dir", default=None)
    validate_split_parser.add_argument("--policy", default=None)

    # D3-A
    build_seg_parser = sub.add_parser("build-segmentation-manifest")
    build_seg_parser.add_argument("--processed-dir", required=True)
    build_seg_parser.add_argument("--split-dir", required=True)
    build_seg_parser.add_argument("--config", required=True)
    build_seg_parser.add_argument("--output", required=True)
    build_seg_parser.add_argument("--report-dir", required=True)
    build_seg_parser.add_argument("--no-smoke", action="store_true")

    validate_seg_parser = sub.add_parser("validate-segmentation")
    validate_seg_parser.add_argument("--segmentation-dir", required=True)
    validate_seg_parser.add_argument("--config", default=None)
    validate_seg_parser.add_argument("--split-dir", default=None)

    smoke_parser = sub.add_parser("segmentation-smoke-test")
    smoke_parser.add_argument("--segmentation-dir", required=True)
    smoke_parser.add_argument("--config", required=True)

    # D3-B
    train_smoke_parser = sub.add_parser("segmentation-train-smoke")
    train_smoke_parser.add_argument("--segmentation-dir", required=True)
    train_smoke_parser.add_argument("--data-config", required=True)
    train_smoke_parser.add_argument("--train-config", required=True)
    train_smoke_parser.add_argument("--output", required=True)

    overfit_parser = sub.add_parser("segmentation-overfit")
    overfit_parser.add_argument("--segmentation-dir", required=True)
    overfit_parser.add_argument("--data-config", required=True)
    overfit_parser.add_argument("--train-config", required=True)
    overfit_parser.add_argument("--output", required=True)

    # D3-C：训练与 test 评估物理分离
    full_train_parser = sub.add_parser("segmentation-train")
    full_train_parser.add_argument("--segmentation-dir", required=True)
    full_train_parser.add_argument("--data-config", required=True)
    full_train_parser.add_argument("--train-config", required=True)
    full_train_parser.add_argument("--output", required=True)
    full_train_parser.add_argument("--resume", default=None)

    preflight_parser = sub.add_parser("segmentation-preflight")
    preflight_parser.add_argument("--segmentation-dir", required=True)
    preflight_parser.add_argument("--data-config", required=True)
    preflight_parser.add_argument("--train-config", required=True)

    eval_parser = sub.add_parser("segmentation-evaluate")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--segmentation-dir", required=True)
    eval_parser.add_argument("--data-config", required=True)
    eval_parser.add_argument("--train-config", required=True)
    eval_parser.add_argument("--split", required=True, choices=["val", "test"])
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required when --split test; only after best checkpoint freeze",
    )

    # D3-E：原图推理 / ROI（不训练）
    infer_parser = sub.add_parser("segmentation-infer")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--data-config", required=True)
    infer_parser.add_argument("--train-config", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--device", default="auto")
    infer_parser.add_argument("--sample-id", default=None)
    infer_parser.add_argument("--roi-margin-ratio", type=float, default=0.05)

    infer_batch_parser = sub.add_parser("segmentation-infer-regression")
    infer_batch_parser.add_argument("--checkpoint", required=True)
    infer_batch_parser.add_argument("--segmentation-dir", required=True)
    infer_batch_parser.add_argument("--data-config", required=True)
    infer_batch_parser.add_argument("--train-config", required=True)
    infer_batch_parser.add_argument("--output", required=True)
    infer_batch_parser.add_argument("--device", default="auto")
    infer_batch_parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required; D3-E test access is engineering regression only",
    )

    # D4-A：Input Guard 契约校验 / smoke（不训练 QC）
    validate_guard_parser = sub.add_parser("validate-input-guard")
    validate_guard_parser.add_argument(
        "--policy", default="configs/input_guard_v1.yaml"
    )

    guard_smoke_parser = sub.add_parser("input-guard-smoke")
    guard_smoke_parser.add_argument("--checkpoint", required=True)
    guard_smoke_parser.add_argument("--segmentation-dir", required=True)
    guard_smoke_parser.add_argument("--data-config", required=True)
    guard_smoke_parser.add_argument("--train-config", required=True)
    guard_smoke_parser.add_argument(
        "--policy", default="configs/input_guard_v1.yaml"
    )
    guard_smoke_parser.add_argument("--output", required=True)
    guard_smoke_parser.add_argument("--device", default="auto")

    # D4-B：feature audit / calibrate / runtime / test audit
    feature_audit_parser = sub.add_parser("input-guard-feature-audit")
    feature_audit_parser.add_argument("--checkpoint", required=True)
    feature_audit_parser.add_argument("--segmentation-dir", required=True)
    feature_audit_parser.add_argument("--data-config", required=True)
    feature_audit_parser.add_argument("--train-config", required=True)
    feature_audit_parser.add_argument("--output", required=True)
    feature_audit_parser.add_argument("--device", default="auto")

    calibrate_parser = sub.add_parser("input-guard-calibrate")
    calibrate_parser.add_argument("--checkpoint", required=True)
    calibrate_parser.add_argument("--segmentation-dir", required=True)
    calibrate_parser.add_argument("--data-config", required=True)
    calibrate_parser.add_argument("--train-config", required=True)
    calibrate_parser.add_argument(
        "--policy", default="configs/input_guard_v1.yaml"
    )
    calibrate_parser.add_argument("--output", required=True)
    calibrate_parser.add_argument("--device", default="auto")

    guard_run_parser = sub.add_parser("input-guard-run")
    guard_run_parser.add_argument("--image", required=True)
    guard_run_parser.add_argument("--checkpoint", required=True)
    guard_run_parser.add_argument("--data-config", required=True)
    guard_run_parser.add_argument("--train-config", required=True)
    guard_run_parser.add_argument(
        "--policy", default="configs/input_guard_v1.yaml"
    )
    guard_run_parser.add_argument("--device", default="auto")
    guard_run_parser.add_argument("--sample-id", default=None)
    guard_run_parser.add_argument("--stain-checkpoint", default=None)
    guard_run_parser.add_argument(
        "--stain-data-config", default="configs/stain_detection_v1.yaml"
    )
    guard_run_parser.add_argument(
        "--stain-train-config", default="configs/stain_train_v1.yaml"
    )
    guard_run_parser.add_argument("--stain-thresholds", default=None)

    test_audit_parser = sub.add_parser("input-guard-test-audit")
    test_audit_parser.add_argument("--checkpoint", required=True)
    test_audit_parser.add_argument("--segmentation-dir", required=True)
    test_audit_parser.add_argument("--data-config", required=True)
    test_audit_parser.add_argument("--train-config", required=True)
    test_audit_parser.add_argument(
        "--policy", default="configs/input_guard_v1.yaml"
    )
    test_audit_parser.add_argument("--output", required=True)
    test_audit_parser.add_argument("--device", default="auto")
    test_audit_parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required; engineering audit only after threshold freeze",
    )

    # D4-C：stain detection baseline（分离 command，禁止 train+calibrate+test 混跑）
    stain_build = sub.add_parser("stain-build-manifest")
    stain_build.add_argument("--processed-dir", default="data/processed/v1")
    stain_build.add_argument("--split-dir", default="data/splits/v1")
    stain_build.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_build.add_argument("--output", default="data/stain/v1")

    stain_preflight = sub.add_parser("stain-preflight")
    stain_preflight.add_argument("--processed-dir", default="data/processed/v1")
    stain_preflight.add_argument("--split-dir", default="data/splits/v1")
    stain_preflight.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_preflight.add_argument("--checkpoint", required=True)
    stain_preflight.add_argument(
        "--seg-data-config", default="configs/segmentation_v1.yaml"
    )
    stain_preflight.add_argument(
        "--seg-train-config", default="configs/segmentation_train_v1.yaml"
    )
    stain_preflight.add_argument("--output", default="data/stain/v1")
    stain_preflight.add_argument("--device", default="auto")

    stain_smoke = sub.add_parser("stain-train-smoke")
    stain_smoke.add_argument("--manifest", default="data/stain/v1/stain_manifest.parquet")
    stain_smoke.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_smoke.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_smoke.add_argument("--output", default="runs/input_guard/d4c/stain/smoke")
    stain_smoke.add_argument("--device", default="auto")

    stain_overfit = sub.add_parser("stain-overfit")
    stain_overfit.add_argument("--manifest", default="data/stain/v1/stain_manifest.parquet")
    stain_overfit.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_overfit.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_overfit.add_argument("--output", default="runs/input_guard/d4c/stain/overfit")
    stain_overfit.add_argument("--device", default="auto")

    stain_train = sub.add_parser("stain-train")
    stain_train.add_argument("--manifest", default="data/stain/v1/stain_manifest.parquet")
    stain_train.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_train.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_train.add_argument("--output", default="runs/input_guard/d4c/stain")
    stain_train.add_argument("--device", default="auto")

    stain_calibrate = sub.add_parser("stain-calibrate")
    stain_calibrate.add_argument("--run-dir", default="runs/input_guard/d4c/stain")
    stain_calibrate.add_argument(
        "--manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    stain_calibrate.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_calibrate.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_calibrate.add_argument("--device", default="auto")
    stain_calibrate.add_argument(
        "--update-policy",
        default="configs/input_guard_v1.yaml",
        help="write val-frozen thresholds into Input Guard policy 1.2",
    )

    stain_evaluate = sub.add_parser("stain-evaluate")
    stain_evaluate.add_argument("--run-dir", default="runs/input_guard/d4c/stain")
    stain_evaluate.add_argument(
        "--manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    stain_evaluate.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_evaluate.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_evaluate.add_argument("--device", default="auto")
    stain_evaluate.add_argument(
        "--allow-test",
        action="store_true",
        help="required; test once after model+thresholds freeze",
    )
    stain_evaluate.add_argument("--d4b-audit", default=None)

    stain_infer = sub.add_parser("stain-infer")
    stain_infer.add_argument("--image", required=True)
    stain_infer.add_argument("--seg-checkpoint", required=True)
    stain_infer.add_argument(
        "--seg-data-config", default="configs/segmentation_v1.yaml"
    )
    stain_infer.add_argument(
        "--seg-train-config", default="configs/segmentation_train_v1.yaml"
    )
    stain_infer.add_argument("--stain-checkpoint", required=True)
    stain_infer.add_argument("--data-config", default="configs/stain_detection_v1.yaml")
    stain_infer.add_argument("--train-config", default="configs/stain_train_v1.yaml")
    stain_infer.add_argument("--thresholds", default=None)
    stain_infer.add_argument("--device", default="auto")
    stain_infer.add_argument("--sample-id", default=None)

    # D4-D：color_cast / occlusion / unified guard
    d4d_cal = sub.add_parser("input-guard-d4d-calibrate")
    d4d_cal.add_argument("--checkpoint", required=True)
    d4d_cal.add_argument("--segmentation-dir", required=True)
    d4d_cal.add_argument("--data-config", required=True)
    d4d_cal.add_argument("--train-config", required=True)
    d4d_cal.add_argument("--policy", default="configs/input_guard_v1.yaml")
    d4d_cal.add_argument("--d4d-config", default="configs/input_guard_d4d_v1.yaml")
    d4d_cal.add_argument("--output", default="runs/input_guard/d4d")
    d4d_cal.add_argument("--device", default="auto")
    d4d_cal.add_argument("--max-samples", type=int, default=None)

    d4d_synth = sub.add_parser("input-guard-d4d-synthetic-audit")
    d4d_synth.add_argument("--checkpoint", required=True)
    d4d_synth.add_argument("--segmentation-dir", required=True)
    d4d_synth.add_argument("--data-config", required=True)
    d4d_synth.add_argument("--train-config", required=True)
    d4d_synth.add_argument("--policy", default="configs/input_guard_v1.yaml")
    d4d_synth.add_argument("--d4d-config", default="configs/input_guard_d4d_v1.yaml")
    d4d_synth.add_argument("--output", default="runs/input_guard/d4d")
    d4d_synth.add_argument("--device", default="auto")
    d4d_synth.add_argument("--max-samples", type=int, default=120)

    unified_run = sub.add_parser("input-guard-unified-run")
    unified_run.add_argument("--image", required=True)
    unified_run.add_argument("--seg-checkpoint", required=True)
    unified_run.add_argument("--data-config", required=True)
    unified_run.add_argument("--train-config", required=True)
    unified_run.add_argument("--policy", default="configs/input_guard_v1.yaml")
    unified_run.add_argument("--stain-checkpoint", default=None)
    unified_run.add_argument("--stain-thresholds", default=None)
    unified_run.add_argument("--device", default="auto")
    unified_run.add_argument("--sample-id", default=None)

    unified_audit = sub.add_parser("input-guard-unified-audit")
    unified_audit.add_argument("--checkpoint", required=True)
    unified_audit.add_argument("--segmentation-dir", required=True)
    unified_audit.add_argument("--data-config", required=True)
    unified_audit.add_argument("--train-config", required=True)
    unified_audit.add_argument("--policy", default="configs/input_guard_v1.yaml")
    unified_audit.add_argument("--output", default="runs/input_guard/d4d")
    unified_audit.add_argument("--stain-checkpoint", default=None)
    unified_audit.add_argument("--stain-thresholds", default=None)
    unified_audit.add_argument("--device", default="auto")
    unified_audit.add_argument("--allow-test", action="store_true")

    # D4-D.1：只读 integration audit（不改 threshold / runtime）
    integration_audit = sub.add_parser("input-guard-integration-audit")
    integration_audit.add_argument(
        "--seg-checkpoint",
        default="runs/segmentation/d3c/baseline/best.pt",
    )
    integration_audit.add_argument(
        "--segmentation-dir", default="data/segmentation/v1"
    )
    integration_audit.add_argument(
        "--data-config", default="configs/segmentation_v1.yaml"
    )
    integration_audit.add_argument(
        "--train-config", default="configs/segmentation_train_v1.yaml"
    )
    integration_audit.add_argument(
        "--stain-checkpoint", default="runs/input_guard/d4c/stain/best.pt"
    )
    integration_audit.add_argument(
        "--stain-thresholds",
        default="runs/input_guard/d4c/stain/thresholds.json",
    )
    integration_audit.add_argument("--policy", default="configs/input_guard_v1.yaml")
    integration_audit.add_argument("--output", default="reports/d4")
    integration_audit.add_argument("--device", default="auto")
    integration_audit.add_argument(
        "--d4c-test-predictions",
        default="runs/input_guard/d4c/stain/test_predictions.parquet",
    )

    # D4-E：production stain-deferred unified audit + final freeze docs
    d4e_audit = sub.add_parser("input-guard-d4e-production-audit")
    d4e_audit.add_argument(
        "--seg-checkpoint", default="runs/segmentation/d3c/baseline/best.pt"
    )
    d4e_audit.add_argument("--segmentation-dir", default="data/segmentation/v1")
    d4e_audit.add_argument("--data-config", default="configs/segmentation_v1.yaml")
    d4e_audit.add_argument(
        "--train-config", default="configs/segmentation_train_v1.yaml"
    )
    d4e_audit.add_argument("--policy", default="configs/input_guard_v1.yaml")
    d4e_audit.add_argument(
        "--output", default="reports/d4/d4e_production_unified_audit.json"
    )
    d4e_audit.add_argument(
        "--stain-checkpoint", default="runs/input_guard/d4c/stain/best.pt"
    )
    d4e_audit.add_argument(
        "--stain-thresholds",
        default="runs/input_guard/d4c/stain/thresholds.json",
    )
    d4e_audit.add_argument("--device", default="auto")

    # D4-C.1-A：stain cross-domain shortcut diagnosis（只读）
    stain_diagnose = sub.add_parser("stain-domain-diagnose")
    stain_diagnose.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    stain_diagnose.add_argument(
        "--segmentation-dir", default="data/segmentation/v1"
    )
    stain_diagnose.add_argument(
        "--seg-checkpoint", default="runs/segmentation/d3c/baseline/best.pt"
    )
    stain_diagnose.add_argument(
        "--seg-data-config", default="configs/segmentation_v1.yaml"
    )
    stain_diagnose.add_argument(
        "--seg-train-config", default="configs/segmentation_train_v1.yaml"
    )
    stain_diagnose.add_argument(
        "--stain-checkpoint", default="runs/input_guard/d4c/stain/best.pt"
    )
    stain_diagnose.add_argument(
        "--stain-data-config", default="configs/stain_detection_v1.yaml"
    )
    stain_diagnose.add_argument(
        "--stain-train-config", default="configs/stain_train_v1.yaml"
    )
    stain_diagnose.add_argument(
        "--stain-thresholds",
        default="runs/input_guard/d4c/stain/thresholds.json",
    )
    stain_diagnose.add_argument("--output", default="reports/d4c1")
    stain_diagnose.add_argument("--device", default="auto")

    # D4-C.1-B：domain-robust stain v2
    style_contract = sub.add_parser("stain-style-contract")
    style_contract.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    style_contract.add_argument(
        "--roi-index", default="reports/d4c1/roi_cache/index.parquet"
    )
    style_contract.add_argument(
        "--output", default="reports/d4c1b/style_augmentation_contract.json"
    )

    v2_overfit = sub.add_parser("stain-v2-overfit")
    v2_overfit.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v2_overfit.add_argument("--data-config", default="configs/stain_detection_v2.yaml")
    v2_overfit.add_argument("--train-config", default="configs/stain_train_v2.yaml")
    v2_overfit.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v2_overfit.add_argument("--output", default="runs/input_guard/d4c1b/stain_v2")
    v2_overfit.add_argument("--device", default="auto")

    v2_smoke = sub.add_parser("stain-v2-consistency-smoke")
    v2_smoke.add_argument(
        "--roi-index", default="reports/d4c1/roi_cache/index.parquet"
    )
    v2_smoke.add_argument("--data-config", default="configs/stain_detection_v2.yaml")
    v2_smoke.add_argument("--train-config", default="configs/stain_train_v2.yaml")
    v2_smoke.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v2_smoke.add_argument("--output", default="runs/input_guard/d4c1b/stain_v2")
    v2_smoke.add_argument("--device", default="auto")

    v2_train = sub.add_parser("stain-v2-train")
    v2_train.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v2_train.add_argument(
        "--roi-index", default="reports/d4c1/roi_cache/index.parquet"
    )
    v2_train.add_argument("--data-config", default="configs/stain_detection_v2.yaml")
    v2_train.add_argument("--train-config", default="configs/stain_train_v2.yaml")
    v2_train.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v2_train.add_argument("--output", default="runs/input_guard/d4c1b/stain_v2")
    v2_train.add_argument("--device", default="auto")
    v2_train.add_argument("--max-epochs", type=int, default=None)

    v2_audit = sub.add_parser("stain-v2-audit")
    v2_audit.add_argument("--run-dir", default="runs/input_guard/d4c1b/stain_v2")
    v2_audit.add_argument("--reports-dir", default="reports/d4c1b")
    v2_audit.add_argument("--device", default="auto")

    # D4-C.1-C：representation domain invariance v3
    v3_pre = sub.add_parser("stain-domain-v3-preflight")
    v3_pre.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v3_pre.add_argument("--data-config", default="configs/stain_detection_v3.yaml")
    v3_pre.add_argument("--train-config", default="configs/stain_train_v3.yaml")
    v3_pre.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v3_pre.add_argument("--output", default="runs/input_guard/d4c1c/preflight")
    v3_pre.add_argument("--device", default="auto")

    v3_train = sub.add_parser("stain-domain-v3-train")
    v3_train.add_argument("--candidate", required=True, choices=["c1", "c2", "c3"])
    v3_train.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v3_train.add_argument(
        "--roi-index", default="reports/d4c1/roi_cache/index.parquet"
    )
    v3_train.add_argument("--data-config", default="configs/stain_detection_v3.yaml")
    v3_train.add_argument("--train-config", default="configs/stain_train_v3.yaml")
    v3_train.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v3_train.add_argument("--output-root", default="runs/input_guard/d4c1c")
    v3_train.add_argument("--device", default="auto")
    v3_train.add_argument("--max-epochs", type=int, default=None)
    v3_train.add_argument(
        "--allow-c3",
        action="store_true",
        help="仅当 C1/C2 已有 meaningful robustness signal 时允许",
    )

    v3_audit = sub.add_parser("stain-domain-v3-robustness-audit")
    v3_audit.add_argument("--candidate", required=True, choices=["c1", "c2", "c3"])
    v3_audit.add_argument("--output-root", default="runs/input_guard/d4c1c")
    v3_audit.add_argument("--reports-dir", default="reports/d4c1c")
    v3_audit.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v3_audit.add_argument(
        "--roi-index", default="reports/d4c1/roi_cache/index.parquet"
    )
    v3_audit.add_argument("--data-config", default="configs/stain_detection_v3.yaml")
    v3_audit.add_argument("--train-config", default="configs/stain_train_v3.yaml")
    v3_audit.add_argument(
        "--style-contract",
        default="reports/d4c1b/style_augmentation_contract.json",
    )
    v3_audit.add_argument("--device", default="auto")
    v3_audit.add_argument("--v2-domain-probe", type=float, default=None)
    v3_audit.add_argument("--v2-style-sensitivity", type=float, default=None)

    v3_cal = sub.add_parser("stain-domain-v3-calibrate")
    v3_cal.add_argument("--candidate", required=True, choices=["c1", "c2", "c3"])
    v3_cal.add_argument("--output-root", default="runs/input_guard/d4c1c")
    v3_cal.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    v3_cal.add_argument("--data-config", default="configs/stain_detection_v3.yaml")
    v3_cal.add_argument("--train-config", default="configs/stain_train_v3.yaml")
    v3_cal.add_argument("--device", default="auto")

    v3_src = sub.add_parser("stain-domain-v3-source-test")
    v3_src.add_argument(
        "--help-note",
        action="store_true",
        help="占位：仅 candidate PASS 后启用",
    )

    v3_known = sub.add_parser("stain-domain-v3-known-audit")
    v3_unified = sub.add_parser("stain-domain-v3-unified-recovery")

    # D4-C.1-D：dataset confounding audit（无 CNN 训练）
    d4c1d = sub.add_parser("stain-confounding-audit")
    d4c1d.add_argument(
        "--stain-manifest", default="data/stain/v1/stain_manifest.parquet"
    )
    d4c1d.add_argument("--reports-dir", default="reports/d4c1d")
    d4c1d.add_argument("--docs-dir", default="docs")
    d4c1d.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="若已有 source_confounding_manifest.parquet 则复用",
    )

    args = parser.parse_args()
    if args.cmd == "validate-contract":
        errors, warnings = validate_contract(
            args.ontology, args.mappings_dir, args.strict
        )
        raise SystemExit(emit(errors, warnings))
    if args.cmd == "build":
        samples, labels, spatial, meta = ManifestBuilder(args.config).build(args.output)
        print(
            f"samples={len(samples)} labels={len(labels)} "
            f"spatial={len(spatial)} warnings={meta['warnings_count']}"
        )
        return
    if args.cmd == "validate-manifest":
        errors, warnings = validate_manifest(args.manifest_dir)
        raise SystemExit(emit(errors, warnings))
    if args.cmd == "clean":
        meta = CleaningBuilder(args.policy).build(
            args.manifest_dir, args.output, args.report_dir
        )
        print(
            f"samples_before={meta['samples_before']} samples_after={meta['samples_after']} "
            f"aliases={meta['duplicate_aliases']} label_conflicts={meta['label_conflicts']} "
            f"cross_dataset_duplicates={meta['cross_dataset_duplicates']}"
        )
        return
    if args.cmd == "validate-clean":
        errors, warnings = validate_clean(args.processed_dir, args.policy)
        raise SystemExit(emit(errors, warnings))
    if args.cmd == "build-groups":
        audit = SplitBuilder(args.policy).build_groups(
            args.processed_dir, args.manifest_dir, args.output, args.report_dir
        )
        print(
            f"samples={audit['total_canonical_samples']} "
            f"groups={audit['total_leakage_groups']} "
            f"missing_patient={audit.get('tonguedx_missing_patient_id_count', 0)}"
        )
        return
    if args.cmd == "split":
        result = SplitBuilder(args.policy).build_split(
            args.processed_dir, args.output, args.report_dir, args.groups
        )
        report = result["split_report"]
        print(
            f"train={report['train']} val={report['val']} test={report['test']} "
            f"external_holdout={report['external_holdout']} "
            f"leakage={result['leakage']}"
        )
        return
    if args.cmd == "build-splits":
        result = SplitBuilder(args.policy).build_all(
            args.processed_dir, args.manifest_dir, args.output, args.report_dir
        )
        report = result["split_report"]
        print(
            f"groups={result['group_audit']['total_leakage_groups']} "
            f"train={report['train']} val={report['val']} test={report['test']} "
            f"external_holdout={report['external_holdout']} "
            f"leakage={result['leakage']}"
        )
        return
    if args.cmd == "validate-split":
        errors, warnings = validate_split(
            args.split_dir, args.processed_dir, args.policy
        )
        raise SystemExit(emit(errors, warnings))
    if args.cmd == "build-segmentation-manifest":
        result = SegmentationBuilder(args.config).build(
            args.processed_dir,
            args.split_dir,
            args.output,
            args.report_dir,
            run_smoke=not args.no_smoke,
        )
        meta = result["metadata"]
        print(
            f"samples={meta['total_samples']} "
            f"train={meta['per_split']['train']} "
            f"val={meta['per_split']['val']} "
            f"test={meta['per_split']['test']} "
            f"smoke={'PASS' if (result.get('smoke') or {}).get('ok', False) else 'SKIP/FAIL'}"
        )
        return
    if args.cmd == "validate-segmentation":
        errors, warnings = validate_segmentation(
            args.segmentation_dir, args.config, args.split_dir
        )
        raise SystemExit(emit(errors, warnings))
    if args.cmd == "segmentation-smoke-test":
        smoke = smoke_test_dataset(
            Path(args.segmentation_dir) / "segmentation_manifest.parquet"
            if (Path(args.segmentation_dir) / "segmentation_manifest.parquet").exists()
            else args.segmentation_dir,
            args.config,
        )
        print(json.dumps(smoke, ensure_ascii=False))
        raise SystemExit(0 if smoke.get("ok") else 1)
    if args.cmd == "segmentation-train-smoke":
        metadata = run_smoke_training(
            args.segmentation_dir,
            args.data_config,
            args.train_config,
            args.output,
        )
        print(
            f"device={metadata['device']} amp={metadata['amp']} "
            f"epoch={metadata['final_epoch']} "
            f"best_val_dice={metadata['best_val_dice']:.4f} "
            f"resume={metadata['resume_test']['result']}"
        )
        return
    if args.cmd == "segmentation-overfit":
        metadata = run_tiny_overfit(
            args.segmentation_dir,
            args.data_config,
            args.train_config,
            args.output,
        )
        print(
            f"samples={metadata['sample_count']} steps={metadata['steps']} "
            f"loss={metadata['initial_loss']:.4f}->{metadata['final_loss']:.4f} "
            f"dice={metadata['final_dice']:.4f} result={metadata['result']}"
        )
        return
    if args.cmd == "segmentation-preflight":
        preflight = preflight_full_training(
            args.segmentation_dir, args.data_config, args.train_config
        )
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if args.cmd == "segmentation-train":
        metadata = run_full_training(
            args.segmentation_dir,
            args.data_config,
            args.train_config,
            args.output,
            resume_from=args.resume,
        )
        print(
            f"run_id={metadata.get('run_id')} "
            f"epochs={metadata['actual_epochs']}/{metadata['planned_epochs']} "
            f"early_stop={metadata['early_stopped']} "
            f"best_epoch={metadata['best_epoch']} "
            f"best_val_dice={metadata['best_val_dice']:.4f} "
            f"frozen={metadata.get('baseline_frozen')}"
        )
        return
    if args.cmd == "segmentation-evaluate":
        if args.split == "test" and not args.allow_test:
            raise SystemExit(
                "ERROR: test evaluation requires --allow-test after best checkpoint freeze"
            )
        result = evaluate_checkpoint_on_split(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            split=args.split,
            output_dir=args.output,
            allow_test=bool(args.allow_test or args.split != "test"),
        )
        overall = result["overall"]["dice"]["mean"]
        gate = (result.get("baseline_gate") or {}).get("baseline_status")
        print(
            f"split={result['split']} dice={overall:.4f} "
            f"biohit={result['biohit']['dice']['mean']:.4f} "
            f"tongueset3={result['tongueset3']['dice']['mean']:.4f} "
            f"gap={result['domain_gap_dice']:.4f} gate={gate}"
        )
        return
    if args.cmd == "segmentation-infer":
        from .segmentation.inference import (
            TongueSegmentationInference,
            format_console_summary,
            load_rgb_image,
            save_inference_outputs,
        )

        engine = TongueSegmentationInference(
            checkpoint_path=args.checkpoint,
            data_config=args.data_config,
            train_config=args.train_config,
            device=args.device,
            roi_margin_ratio=float(args.roi_margin_ratio),
        )
        result = engine.predict(args.image, sample_id=args.sample_id)
        original_rgb, _mode = load_rgb_image(args.image)
        sample_name = args.sample_id or Path(args.image).stem
        sample_dir = Path(args.output) / str(sample_name).replace("::", "__")
        save_inference_outputs(result, sample_dir, original_rgb=original_rgb)
        print(format_console_summary(result))
        print(f"output: {sample_dir}")
        return
    if args.cmd == "segmentation-infer-regression":
        if not args.allow_test:
            raise SystemExit(
                "ERROR: D3-E regression on frozen test requires --allow-test "
                "(engineering regression only; do not tune model)"
            )
        from .segmentation.regression_d3e import run_d3e_test_regression

        report = run_d3e_test_regression(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"total={report['total']} "
            f"d3e_dice={report['overall_original_resolution_dice']:.4f} "
            f"delta={report['difference_vs_d3c']['overall']:+.4f} "
            f"biohit={report['biohit_original_resolution_dice']:.4f} "
            f"tongueset3={report['tongueset3_original_resolution_dice']:.4f} "
            f"invalid_bbox={report['invalid_bbox']} empty_roi={report['empty_roi']}"
        )
        return
    if args.cmd == "validate-input-guard":
        from .input_guard.validators import emit_validate, validate_input_guard_contract

        errors, warnings = validate_input_guard_contract(args.policy)
        raise SystemExit(emit_validate(errors, warnings))
    if args.cmd == "input-guard-smoke":
        from .input_guard.smoke import run_input_guard_contract_smoke

        summary = run_input_guard_contract_smoke(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"contract_status={summary['contract_status']} "
            f"samples={summary['sample_count']} "
            f"evaluation_complete_all_false={summary['all_evaluation_complete_false']} "
            f"defined={summary['defined_checks_count']} "
            f"implemented={summary['implemented_checks_count']}"
        )
        raise SystemExit(0 if summary["contract_status"] == "PASS" else 1)
    if args.cmd == "input-guard-feature-audit":
        from .input_guard.calibration import (
            build_feature_distribution,
            collect_calibration_rows,
        )
        import json
        from pathlib import Path as _Path

        rows = collect_calibration_rows(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            device=args.device,
        )
        distribution = build_feature_distribution(rows)
        out = _Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        reports = _Path("reports/d4")
        reports.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(distribution, ensure_ascii=False, indent=2)
        (out / "d4b_feature_distribution.json").write_text(payload, encoding="utf-8")
        (reports / "d4b_feature_distribution.json").write_text(payload, encoding="utf-8")
        print(f"samples={distribution['sample_count']} output={out}")
        return
    if args.cmd == "input-guard-calibrate":
        from .input_guard.calibration import run_calibration_pipeline

        result = run_calibration_pipeline(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_dir=args.output,
            device=args.device,
            write_policy=True,
        )
        print(
            f"calibrated samples={result['sample_count']} "
            f"policy_version={result['policy_version']} "
            f"policy={result['policy_path']}"
        )
        return
    if args.cmd == "input-guard-run":
        from .input_guard.runtime import InputGuardRuntime, format_runtime_summary

        runtime = InputGuardRuntime(
            checkpoint_path=args.checkpoint,
            data_config=args.data_config,
            train_config=args.train_config,
            policy_path=args.policy,
            device=args.device,
            stain_checkpoint=args.stain_checkpoint,
            stain_data_config=args.stain_data_config,
            stain_train_config=args.stain_train_config,
            stain_thresholds=args.stain_thresholds,
        )
        result = runtime.evaluate(args.image, sample_id=args.sample_id)
        print(format_runtime_summary(result))
        return
    if args.cmd == "input-guard-test-audit":
        if not args.allow_test:
            raise SystemExit(
                "ERROR: test audit requires --allow-test after threshold freeze "
                "(engineering audit only; do not retune)"
            )
        from .input_guard.audit import run_test_engineering_audit

        report = run_test_engineering_audit(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"total={report['total']} "
            f"pass={report['decision_counts']['pass']} "
            f"warning={report['decision_counts']['warning']} "
            f"retake={report['decision_counts']['retake']} "
            f"review={report['calibration_review_required']}"
        )
        return
    if args.cmd == "stain-build-manifest":
        from .stain.manifest import build_stain_base_frame, class_balance_report
        from pathlib import Path as _Path
        import json as _json

        frame = build_stain_base_frame(
            args.processed_dir, args.split_dir, args.data_config
        )
        out = _Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / "stain_base_manifest.parquet", index=False)
        balance = class_balance_report(frame)
        (out / "class_balance.json").write_text(
            _json.dumps(balance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"samples={len(frame)} "
            f"train={(frame.split=='train').sum()} "
            f"val={(frame.split=='val').sum()} "
            f"test={(frame.split=='test').sum()} "
            f"pos={(frame.label==1).sum()} neg={(frame.label==0).sum()}"
        )
        return
    if args.cmd == "stain-preflight":
        from .stain.manifest import build_stain_base_frame, run_d3e_roi_preflight

        base = build_stain_base_frame(
            args.processed_dir, args.split_dir, args.data_config
        )
        result = run_d3e_roi_preflight(
            base,
            checkpoint_path=args.checkpoint,
            data_config_path=args.seg_data_config,
            train_config_path=args.seg_train_config,
            stain_data_config=args.data_config,
            output_dir=args.output,
            device=args.device,
        )
        audit = result["audit"]
        print(
            f"roi_success_rate={audit['roi_success_rate']:.4f} "
            f"gate={audit['roi_success_gate']} "
            f"excluded={len(audit['excluded_samples'])}"
        )
        return
    if args.cmd == "stain-train-smoke":
        from .stain.train import run_stain_training

        meta = run_stain_training(
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            output_dir=args.output,
            device=args.device,
            smoke=True,
        )
        print(
            f"smoke ok best_val_auroc={meta['best_val_auroc']} "
            f"epochs={meta['actual_epochs']}"
        )
        return
    if args.cmd == "stain-overfit":
        from .stain.train import run_tiny_overfit

        report = run_tiny_overfit(
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"overfit={report['status']} acc={report['final_accuracy']:.3f} "
            f"loss_drop={report['loss_drop']:.4f}"
        )
        return
    if args.cmd == "stain-train":
        from .stain.train import run_stain_training

        meta = run_stain_training(
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            output_dir=args.output,
            device=args.device,
            smoke=False,
        )
        print(
            f"train done best_epoch={meta['best_epoch']} "
            f"best_val_auroc={meta['best_val_auroc']} "
            f"actual_epochs={meta['actual_epochs']}"
        )
        return
    if args.cmd == "stain-calibrate":
        from .stain.evaluate import run_val_calibration
        from .stain.detector import update_policy_with_stain_thresholds

        calibration = run_val_calibration(
            run_dir=args.run_dir,
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            device=args.device,
        )
        if args.update_policy:
            update_policy_with_stain_thresholds(
                args.update_policy,
                t_clear=calibration["t_clear"],
                t_retake=calibration["t_retake"],
                policy_version="1.2",
            )
        print(
            f"t_clear={calibration['t_clear']} t_retake={calibration['t_retake']} "
            f"constraint_not_met={calibration['constraint_not_met']} "
            f"uncertain_rate={calibration.get('uncertain_rate')}"
        )
        return
    if args.cmd == "stain-evaluate":
        if not args.allow_test:
            raise SystemExit(
                "ERROR: stain-evaluate requires --allow-test after freeze "
                "(test once; do not retune thresholds)"
            )
        from .stain.evaluate import run_frozen_test_evaluation

        report = run_frozen_test_evaluation(
            run_dir=args.run_dir,
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            device=args.device,
            allow_test=True,
            d4b_audit_path=args.d4b_audit,
        )
        print(
            f"baseline_status={report['baseline_status']} "
            f"auroc={report['ranking']['auroc']} "
            f"stain_precision={report['three_state']['confident_stain_precision']} "
            f"coverage={report['three_state']['confident_coverage']}"
        )
        return
    if args.cmd == "stain-infer":
        from .segmentation.inference import TongueSegmentationInference, load_rgb_image
        from .stain.detector import StainDetector
        from pathlib import Path as _Path

        rgb, _mode = load_rgb_image(args.image)
        seg = TongueSegmentationInference(
            checkpoint_path=args.seg_checkpoint,
            data_config=args.seg_data_config,
            train_config=args.seg_train_config,
            device=args.device,
            return_masked_roi=False,
        )
        seg_result = seg.predict(rgb, sample_id=args.sample_id)
        thresholds = args.thresholds or str(
            _Path(args.stain_checkpoint).parent / "thresholds.json"
        )
        detector = StainDetector(
            checkpoint_path=args.stain_checkpoint,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            thresholds_path=thresholds,
            device=args.device,
        )
        check = detector.predict(rgb, seg_result)
        print(
            f"finding={check.finding} score={check.score} "
            f"decision_effect={check.decision_effect} "
            f"reason={check.reason_code} source={check.source}"
        )
        return
    if args.cmd == "input-guard-d4d-calibrate":
        from .input_guard.d4d_calibration import run_d4d_calibration_pipeline

        result = run_d4d_calibration_pipeline(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_dir=args.output,
            d4d_config_path=args.d4d_config,
            device=args.device,
            write_policy=True,
            max_samples=args.max_samples,
        )
        print(
            f"color_cast={result['color_cast_status']} "
            f"occlusion={result['occlusion_status']} "
            f"samples={result['sample_count']} "
            f"policy={result['policy_path']}"
        )
        return
    if args.cmd == "input-guard-d4d-synthetic-audit":
        from .input_guard.d4d_calibration import (
            collect_d4d_calibration_rows,
            load_d4d_config,
        )
        from .input_guard.d4d_synthetic import (
            run_color_cast_synthetic_audit,
            run_occlusion_synthetic_audit,
        )
        from .input_guard.policy import InputGuardPolicy
        import json as _json
        from pathlib import Path as _Path

        d4d_cfg = load_d4d_config(args.d4d_config)
        rows = collect_d4d_calibration_rows(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            d4d_config=d4d_cfg,
            device=args.device,
            max_samples=args.max_samples,
        )
        policy = InputGuardPolicy(args.policy)
        cast = run_color_cast_synthetic_audit(rows, policy=policy, d4d_cfg=d4d_cfg)
        occ = run_occlusion_synthetic_audit(rows, policy=policy, d4d_cfg=d4d_cfg)
        out = _Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "d4d_synthetic_color_cast.json").write_text(
            _json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "d4d_synthetic_occlusion.json").write_text(
            _json.dumps(occ, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"cast_severe={cast.get('severe_detection_rate')} "
            f"occ_severe={occ.get('severe_detection_rate')} "
            f"occ_small_retake={occ.get('small_retake_rate')}"
        )
        return
    if args.cmd == "input-guard-unified-run":
        from .input_guard.runtime import InputGuardRuntime, format_runtime_summary

        runtime = InputGuardRuntime(
            checkpoint_path=args.seg_checkpoint,
            data_config=args.data_config,
            train_config=args.train_config,
            policy_path=args.policy,
            device=args.device,
            stain_checkpoint=args.stain_checkpoint,
            stain_thresholds=args.stain_thresholds,
        )
        result = runtime.evaluate(args.image, sample_id=args.sample_id)
        print(format_runtime_summary(result))
        return
    if args.cmd == "input-guard-unified-audit":
        if not args.allow_test:
            raise SystemExit(
                "ERROR: unified audit requires --allow-test after threshold freeze"
            )
        from .input_guard.d4d_audit import run_unified_test_audit

        report = run_unified_test_audit(
            checkpoint_path=args.checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_dir=args.output,
            stain_checkpoint=args.stain_checkpoint,
            stain_thresholds=args.stain_thresholds,
            device=args.device,
            allow_test=True,
        )
        print(
            f"total={report['total']} "
            f"pass={report['decision_counts'].get('pass')} "
            f"warning={report['decision_counts'].get('warning')} "
            f"retake={report['decision_counts'].get('retake')} "
            f"eval_complete_true={report['evaluation_complete'].get('true')}"
        )
        return
    if args.cmd == "input-guard-integration-audit":
        from .input_guard.d4d1_integration_audit import run_integration_audit

        stats = run_integration_audit(
            checkpoint_path=args.seg_checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            stain_checkpoint=args.stain_checkpoint,
            stain_thresholds=args.stain_thresholds,
            output_dir=args.output,
            device=args.device,
            d4c_test_predictions=args.d4c_test_predictions,
        )
        ablation = stats["ablation"]["counts"]
        newly = stats["newly_rejected"]
        rec = stats["recommendation"]
        print(
            f"n={stats['n_samples']} "
            f"A={ablation['A_d4b_only']} "
            f"B={ablation['B_d4b_stain']} "
            f"C={ablation['C_d4b_stain_cast']} "
            f"D={ablation['D_full']} "
            f"new_retake={newly['n']} "
            f"stain_only_new={newly['stain_only']} "
            f"recommendation={rec['recommendation']}"
        )
        return
    if args.cmd == "input-guard-d4e-production-audit":
        from .input_guard.d4e_audit import (
            run_d4e_production_unified_audit,
            write_d4_final_docs,
        )

        audit = run_d4e_production_unified_audit(
            checkpoint_path=args.seg_checkpoint,
            segmentation_dir=args.segmentation_dir,
            data_config_path=args.data_config,
            train_config_path=args.train_config,
            policy_path=args.policy,
            output_path=args.output,
            stain_checkpoint=args.stain_checkpoint,
            stain_thresholds=args.stain_thresholds,
            device=args.device,
        )
        write_d4_final_docs(audit=audit, docs_dir="docs")
        print(
            f"n={audit['samples']} "
            f"pass={audit['pass']} warning={audit['warning']} retake={audit['retake']} "
            f"stain_inv={audit['stain_model_invocations']} "
            f"guard_ready={audit['guard_ready']} "
            f"full_cov={audit['full_capability_coverage']}"
        )
        return
    if args.cmd == "stain-domain-diagnose":
        from .stain.d4c1a_diagnosis import run_d4c1a_diagnosis

        stats = run_d4c1a_diagnosis(
            stain_manifest=args.stain_manifest,
            segmentation_dir=args.segmentation_dir,
            seg_checkpoint=args.seg_checkpoint,
            seg_data_config=args.seg_data_config,
            seg_train_config=args.seg_train_config,
            stain_checkpoint=args.stain_checkpoint,
            stain_data_config=args.stain_data_config,
            stain_train_config=args.stain_train_config,
            stain_thresholds=args.stain_thresholds,
            output_dir=args.output,
            device=args.device,
        )
        rec = stats["recommendation"]
        rep = stats["representation_ablation"]
        print(
            f"n={stats['n_manifest']} "
            f"preprocess_ok={stats['preprocessing_equivalence']['pass']} "
            f"identity_acc={stats['dataset_identity_audit'].get('cv_accuracy')} "
            f"ts3_black_med={rep.get('tongueset3', {}).get('black', {}).get('median')} "
            f"ts3_gray_med={rep.get('tongueset3', {}).get('gray', {}).get('median')} "
            f"primary={stats['shortcut_evidence']['primary_shortcut_hypothesis']} "
            f"recommendation={rec['recommendation']}"
        )
        return
    if args.cmd == "stain-style-contract":
        from .stain.style_augment import estimate_style_ranges_from_train

        contract = estimate_style_ranges_from_train(
            stain_manifest=args.stain_manifest,
            external_roi_index=args.roi_index,
            output_path=args.output,
        )
        print(
            f"style_contract={args.output} "
            f"gain={contract['channel_gain_ranges']} "
            f"gamma={contract['gamma_range']}"
        )
        return
    if args.cmd == "stain-v2-overfit":
        from .stain.robust_train import run_tiny_source_overfit

        result = run_tiny_source_overfit(
            stain_manifest=args.stain_manifest,
            data_config=args.data_config,
            train_config=args.train_config,
            style_contract=args.style_contract,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"overfit_passed={result['passed']} "
            f"acc={result['final_accuracy']} epochs={result['epochs_run']}"
        )
        return
    if args.cmd == "stain-v2-consistency-smoke":
        from .stain.robust_train import run_external_consistency_smoke

        result = run_external_consistency_smoke(
            roi_index=args.roi_index,
            data_config=args.data_config,
            train_config=args.train_config,
            style_contract=args.style_contract,
            output_dir=args.output,
            device=args.device,
        )
        print(
            f"consistency_smoke_passed={result['passed']} loss={result['loss']}"
        )
        return
    if args.cmd == "stain-v2-train":
        from .stain.robust_train import train_stain_v2

        result = train_stain_v2(
            stain_manifest=args.stain_manifest,
            roi_index=args.roi_index,
            data_config=args.data_config,
            train_config=args.train_config,
            style_contract_path=args.style_contract,
            output_dir=args.output,
            device=args.device,
            max_epochs=args.max_epochs,
        )
        meta = result["metadata"]
        print(
            f"best_epoch={meta['best_epoch']} "
            f"val_auroc={meta['best_source_val_auroc']} "
            f"t_clear={meta['t_clear_v2']} t_retake={meta['t_retake_v2']}"
        )
        return
    if args.cmd == "stain-v2-audit":
        from .stain.robust_audit import run_post_train_pipeline

        stats = run_post_train_pipeline(
            output_run=Path(args.run_dir),
            reports_dir=Path(args.reports_dir),
            device=args.device,
        )
        decision = stats["decision"]
        print(
            f"status={decision['baseline_status']} "
            f"recommendation={decision['recommendation']} "
            f"policy_activated={decision['policy_activated']}"
        )
        return
    if args.cmd == "stain-domain-v3-preflight":
        import json as _json

        from .stain.v3_train import (
            run_grl_unit_smoke,
            run_mixstyle_unit_smoke,
            run_v3_tiny_overfit,
        )

        grl = run_grl_unit_smoke()
        mix = run_mixstyle_unit_smoke()
        overfit = run_v3_tiny_overfit(
            stain_manifest=args.stain_manifest,
            data_config=args.data_config,
            train_config=args.train_config,
            style_contract=args.style_contract,
            output_dir=args.output,
            device=args.device,
        )
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"grl": grl, "mixstyle": mix, "tiny_overfit": overfit}
        (out / "preflight.json").write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"grl={grl['passed']} mixstyle={mix['passed']} "
            f"overfit={overfit['passed']} acc={overfit['final_accuracy']}"
        )
        return
    if args.cmd == "stain-domain-v3-train":
        from .stain.v3_train import CANDIDATE_RUN_DIRS, train_stain_v3_candidate

        result = train_stain_v3_candidate(
            candidate=args.candidate,
            stain_manifest=args.stain_manifest,
            roi_index=args.roi_index,
            data_config=args.data_config,
            train_config=args.train_config,
            style_contract_path=args.style_contract,
            output_root=args.output_root,
            device=args.device,
            max_epochs=args.max_epochs,
            allow_c3=bool(args.allow_c3),
        )
        meta = result["metadata"]
        print(
            f"candidate={args.candidate} "
            f"run={CANDIDATE_RUN_DIRS[args.candidate]} "
            f"best_epoch={meta['best_epoch']} "
            f"val_auroc={meta['best_source_val_auroc']}"
        )
        return
    if args.cmd == "stain-domain-v3-robustness-audit":
        from .stain.v3_audit import run_candidate_full_audit
        from .stain.v3_train import CANDIDATE_RUN_DIRS

        ckpt = (
            Path(args.output_root)
            / CANDIDATE_RUN_DIRS[args.candidate]
            / "best.pt"
        )
        report = run_candidate_full_audit(
            candidate=args.candidate,
            ckpt_path=ckpt,
            stain_manifest=Path(args.stain_manifest),
            roi_index=Path(args.roi_index),
            data_config=Path(args.data_config),
            train_config=Path(args.train_config),
            style_contract_path=Path(args.style_contract),
            reports_dir=Path(args.reports_dir),
            device=args.device,
            v2_domain_probe=args.v2_domain_probe,
            v2_style_sensitivity=args.v2_style_sensitivity,
        )
        acc = report["acceptance"]
        print(
            f"candidate={args.candidate} "
            f"auroc={report['source_val_auroc']} "
            f"gap_red={report['gap_reduction_vs_v2']} "
            f"ts3_high={report['tongueset3_highscore_rate']} "
            f"status={acc['status']} signal={report['meaningful_signal']}"
        )
        return
    if args.cmd == "stain-domain-v3-calibrate":
        from .stain.v3_train import (
            CANDIDATE_RUN_DIRS,
            calibrate_v3_thresholds,
            load_v3_checkpoint,
        )
        from .stain.train import resolve_device

        run_dir = Path(args.output_root) / CANDIDATE_RUN_DIRS[args.candidate]
        final_dir = Path(args.output_root) / "final"
        device_t = resolve_device(args.device)
        model, _ckpt = load_v3_checkpoint(
            run_dir / "best.pt",
            candidate=args.candidate,
            train_config=args.train_config,
            map_location=device_t,
        )
        model = model.to(device_t)
        thr = calibrate_v3_thresholds(
            model=model,
            stain_manifest=args.stain_manifest,
            data_config=args.data_config,
            train_config=args.train_config,
            output_dir=final_dir,
            device=device_t,
        )
        print(
            f"t_clear_v3={thr['t_clear']} t_retake_v3={thr['t_retake']} "
            f"out={final_dir / 'thresholds.json'}"
        )
        return
    if args.cmd in {
        "stain-domain-v3-source-test",
        "stain-domain-v3-known-audit",
        "stain-domain-v3-unified-recovery",
    }:
        print(
            "BLOCKED: source-test / known-audit / unified-recovery "
            "仅在至少一名 candidate 通过 acceptance gate 后启用；"
            "当前请先完成 robustness-audit。"
        )
        return
    if args.cmd == "stain-confounding-audit":
        from .stain.d4c1d_audit import run_full_d4c1d_audit

        stats = run_full_d4c1d_audit(
            stain_manifest=args.stain_manifest,
            reports_dir=args.reports_dir,
            docs_dir=args.docs_dir,
            rebuild_manifest=not bool(args.reuse_manifest),
        )
        decision = stats["decision"]
        print(
            f"n={stats['n_audit_samples']} "
            f"pos={stats['n_positive']} neg={stats['n_negative']} "
            f"acq_auroc={stats['all_acquisition']['auroc']:.4f} "
            f"match_rate={stats['matching']['positive_match_rate']:.3f} "
            f"confirmed={decision['SOURCE_CONFOUNDING_CONFIRMED']} "
            f"level={decision['SOURCE_CONFOUNDING_LEVEL']} "
            f"rescuable={decision['EXISTING_DATA_RESCUABLE']} "
            f"action={decision['RECOMMENDED_DATA_ACTION']}"
        )
        return


if __name__ == "__main__":
    main()
