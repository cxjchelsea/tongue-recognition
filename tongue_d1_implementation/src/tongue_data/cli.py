import argparse
import json
from pathlib import Path

from .manifest import ManifestBuilder
from .validators import validate_contract, validate_manifest
from .cleaning import CleaningBuilder, validate_clean
from .splitting import SplitBuilder, validate_split
from .segmentation import SegmentationBuilder, validate_segmentation
from .segmentation.dataset import smoke_test_dataset


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


if __name__ == "__main__":
    main()
