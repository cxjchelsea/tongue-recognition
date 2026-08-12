from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import yaml
from .ontology import Ontology
from .mapping import MappingRegistry
from .adapters import ADAPTERS
from .schema import SAMPLE_COLUMNS, LABEL_COLUMNS, SPATIAL_COLUMNS

class ManifestBuilder:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        project_root = self.config_path.parent.parent
        op = Path(self.cfg["ontology"])
        mp = Path(self.cfg["mappings_dir"])
        self.ontology_path = op if op.is_absolute() else project_root/op
        self.mappings_dir = mp if mp.is_absolute() else project_root/mp
        self.ontology = Ontology(self.ontology_path)
        self.registry = MappingRegistry(self.ontology,self.mappings_dir)

    def build(self, output_dir):
        out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        all_s,all_l,all_sp,warnings = [],[],[],[]
        dataset_stats,mapping_stats = {},{}
        build_cfg = self.cfg.get("build",{})

        for dcfg in self.cfg.get("datasets",[]):
            if not dcfg.get("enabled",True): continue
            mdoc = self.registry.load(dcfg["mapping"])
            errors,ws = self.registry.validate_doc(
                mdoc,strict=build_cfg.get("fail_on_needs_review",False)
            )
            if errors:
                raise ValueError(f"{dcfg['name']} mapping invalid: {errors}")
            warnings.extend([f"{dcfg['name']}: {x}" for x in ws])

            adapter = ADAPTERS[dcfg["adapter"]](dcfg,mdoc)
            src_errors = adapter.validate_source()
            if src_errors:
                if build_cfg.get("fail_on_missing_dataset",True):
                    raise FileNotFoundError("; ".join(src_errors))
                warnings.extend(src_errors); continue

            s,l,sp,ws = adapter.collect()
            all_s.extend(x.to_dict() for x in s)
            all_l.extend(x.to_dict() for x in l)
            all_sp.extend(x.to_dict() for x in sp)
            warnings.extend(ws)
            dataset_stats[dcfg["name"]] = {
                "samples":len(s),"labels":len(l),"spatial_annotations":len(sp),
                "unique_md5":len({x.md5 for x in s}),
                "patient_ids":len({x.patient_id for x in s if x.patient_id_available and x.patient_id})
            }
            st = {}
            for item in mdoc.get("mappings",{}).values():
                k=item.get("status","missing"); st[k]=st.get(k,0)+1
            mapping_stats[dcfg["name"]] = st

        sdf = pd.DataFrame(all_s,columns=SAMPLE_COLUMNS)
        ldf = pd.DataFrame(all_l,columns=LABEL_COLUMNS)
        spdf = pd.DataFrame(all_sp,columns=SPATIAL_COLUMNS)
        sdf.to_parquet(out/"samples.parquet",index=False)
        ldf.to_parquet(out/"labels.parquet",index=False)
        spdf.to_parquet(out/"spatial_annotations.parquet",index=False)

        (out/"dataset_statistics.json").write_text(json.dumps(dataset_stats,ensure_ascii=False,indent=2),encoding="utf-8")
        (out/"mapping_statistics.json").write_text(json.dumps(mapping_stats,ensure_ascii=False,indent=2),encoding="utf-8")
        meta = {
            "manifest_version":str(build_cfg.get("manifest_version","1.0")),
            "ontology_version":self.ontology.version,
            "build_timestamp":datetime.now(timezone.utc).isoformat(),
            "config_path":str(self.config_path),
            "warnings_count":len(warnings),
            "warnings":warnings[:1000],
        }
        (out/"build_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        return sdf,ldf,spdf,meta
