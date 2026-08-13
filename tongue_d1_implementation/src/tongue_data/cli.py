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


if __name__ == "__main__":
    main()
