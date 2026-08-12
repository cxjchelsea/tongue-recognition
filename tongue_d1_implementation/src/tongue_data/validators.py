from pathlib import Path
import pandas as pd
from .schema import SAMPLE_COLUMNS,LABEL_COLUMNS,SPATIAL_COLUMNS
from .ontology import Ontology
from .mapping import MappingRegistry

def validate_contract(ontology_path,mappings_dir,strict=False):
    ontology=Ontology(ontology_path)
    errors=ontology.validate(); warnings=[]
    reg=MappingRegistry(ontology,mappings_dir)
    for p in sorted(Path(mappings_dir).glob("*.yaml")):
        e,w=reg.validate_doc(reg.load(p.name),strict=strict)
        errors.extend([f"{p.name}: {x}" for x in e])
        warnings.extend([f"{p.name}: {x}" for x in w])
    return errors,warnings

def validate_manifest(manifest_dir):
    root=Path(manifest_dir); errors=[]; warnings=[]
    paths=[root/"samples.parquet",root/"labels.parquet",root/"spatial_annotations.parquet"]
    for p in paths:
        if not p.exists(): errors.append(f"missing manifest file: {p}")
    if errors: return errors,warnings
    s=pd.read_parquet(paths[0]); l=pd.read_parquet(paths[1]); sp=pd.read_parquet(paths[2])

    for name,df,cols in [("samples",s,SAMPLE_COLUMNS),("labels",l,LABEL_COLUMNS),("spatial",sp,SPATIAL_COLUMNS)]:
        miss=[c for c in cols if c not in df.columns]
        if miss: errors.append(f"{name}: missing columns {miss}")

    if s["sample_id"].duplicated().any(): errors.append("samples: duplicate sample_id")
    ids=set(s["sample_id"].astype(str))
    if len(l) and (~l["sample_id"].astype(str).isin(ids)).any(): errors.append("labels reference missing sample")
    if len(sp) and (~sp["sample_id"].astype(str).isin(ids)).any(): errors.append("spatial annotations reference missing sample")

    if len(l):
        if (l["label_available"]!=True).any(): errors.append("persisted label row with label_available != true")
        te_l2=l[(l["source_dataset"]=="tonguexpert") & l["source_field"].isin(
            ["coating_label","tai_label","zhi_label","fissure_label","tooth_mk_label"]
        )]
        if len(te_l2):
            if (te_l2["label_source"]=="human").any(): errors.append("TonguExpert L2 marked human")
            if (te_l2["supervision_tier"]=="gold_candidate").any(): errors.append("TonguExpert L2 marked gold_candidate")

    if len(sp):
        # 向量化校验 bbox，避免上万框时逐行极慢
        bbox = sp[sp["annotation_type"] == "bbox"].copy()
        if len(bbox):
            dims = s.set_index("sample_id")[["width", "height"]]
            joined = bbox.join(dims, on="sample_id", how="left")
            bad_x = joined[
                joined["width"].notna()
                & ~(
                    (joined["x_min"] >= 0)
                    & (joined["x_min"] < joined["x_max"])
                    & (joined["x_max"] <= joined["width"])
                )
            ]
            bad_y = joined[
                joined["height"].notna()
                & ~(
                    (joined["y_min"] >= 0)
                    & (joined["y_min"] < joined["y_max"])
                    & (joined["y_max"] <= joined["height"])
                )
            ]
            for annotation_id in bad_x["annotation_id"].head(20):
                errors.append(f"bad bbox x: {annotation_id}")
            for annotation_id in bad_y["annotation_id"].head(20):
                errors.append(f"bad bbox y: {annotation_id}")
        masks = sp[sp["annotation_type"] == "mask"]
        for mask_path in masks["mask_path"].dropna().unique():
            if not Path(str(mask_path)).exists():
                errors.append(f"mask missing: {mask_path}")
    return errors, warnings
