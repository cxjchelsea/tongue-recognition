import argparse
from .manifest import ManifestBuilder
from .validators import validate_contract, validate_manifest
from .cleaning import CleaningBuilder, validate_clean


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


if __name__ == "__main__":
    main()
