import argparse
from .manifest import ManifestBuilder
from .validators import validate_contract,validate_manifest

def emit(errors,warnings):
    for x in warnings: print(f"[WARN] {x}")
    for x in errors: print(f"[ERROR] {x}")
    if not errors: print("OK")
    return 1 if errors else 0

def main():
    p=argparse.ArgumentParser(prog="tongue-data")
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("validate-contract")
    a.add_argument("--ontology",default="ontology/tongue_phenotype_v1.yaml")
    a.add_argument("--mappings-dir",default="ontology/mappings")
    a.add_argument("--strict",action="store_true")
    b=sub.add_parser("build")
    b.add_argument("--config",required=True); b.add_argument("--output",required=True)
    c=sub.add_parser("validate-manifest")
    c.add_argument("--manifest-dir",required=True)
    args=p.parse_args()
    if args.cmd=="validate-contract":
        e,w=validate_contract(args.ontology,args.mappings_dir,args.strict)
        raise SystemExit(emit(e,w))
    if args.cmd=="build":
        s,l,sp,m=ManifestBuilder(args.config).build(args.output)
        print(f"samples={len(s)} labels={len(l)} spatial={len(sp)} warnings={m['warnings_count']}")
        return
    if args.cmd=="validate-manifest":
        e,w=validate_manifest(args.manifest_dir)
        raise SystemExit(emit(e,w))
if __name__=="__main__": main()
