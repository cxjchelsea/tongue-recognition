from pathlib import Path
from .base import DatasetAdapter
from ..schema import SampleRecord, SpatialRecord
from ..utils import is_image, read_table_auto, normalize_na

class TonguExpertAdapter(DatasetAdapter):
    def _id_col(self,df):
        cfg = self.cfg.get("id_column")
        if cfg and cfg in df.columns: return cfg
        for c in df.columns:
            if str(c).lower() in {"id","sample_id","sampleid","image","image_id","name"}:
                return c
        return df.columns[0]

    def collect(self):
        samples,labels,spatial,warnings = [],[],[],[]
        raw_dir = self.root / self.cfg["raw_dir"]
        mask_dir = self.root / self.cfg["mask_dir"]
        if not raw_dir.exists():
            return samples,labels,spatial,[f"{self.name}: raw_dir missing: {raw_dir}"]

        stem_to_sid = {}
        for img in [p for p in raw_dir.rglob("*") if p.is_file() and is_image(p)]:
            source_id = str(img.relative_to(raw_dir)).replace("\\","/")
            sid = self.make_sample_id(source_id)
            stem_to_sid[img.stem] = sid
            md5,wid,hei = self.basic_image_meta(img)
            samples.append(SampleRecord(
                sid,self.name,source_id,str(img),md5,wid,hei,
                patient_id=img.stem, patient_id_available=False
            ))

        if mask_dir.exists():
            for mask in [p for p in mask_dir.rglob("*") if p.is_file() and is_image(p)]:
                sid = stem_to_sid.get(mask.stem)
                if sid:
                    spatial.append(SpatialRecord(
                        sid,f"{sid}::mask","segmentation.tongue","tongue","mask",
                        None,None,None,None,str(mask),self.name,
                        "tongue_mask_unverified_origin","unknown_mask_origin",
                        "weak","compatible",str(self.mapping_doc.get("version","")),
                        "mask origin not confirmed as human ground truth"
                    ))

        specs = [
            ("L1",self.cfg.get("l1_file"),["labels_tai","labels_zhi","labels_fissure","labels_tooth_mk"]),
            ("L2",self.cfg.get("l2_file"),["coating_label","tai_label","zhi_label","fissure_label","tooth_mk_label"]),
        ]
        for group,rel,fields in specs:
            if not rel: continue
            p = self.root / rel
            if not p.exists():
                warnings.append(f"{self.name}: {group} file missing: {p}")
                continue
            df = read_table_auto(p)
            id_col = self._id_col(df)
            for _,row in df.iterrows():
                rid = normalize_na(row.get(id_col))
                if rid is None: continue
                stem = Path(str(rid)).stem
                sid = stem_to_sid.get(stem)
                if sid is None:
                    matches = [v for k,v in stem_to_sid.items() if k == stem or stem in k or k in stem]
                    sid = matches[0] if len(matches)==1 else None
                if sid is None:
                    warnings.append(f"{self.name}: {group} id unmatched: {rid}")
                    continue
                for field in fields:
                    if field not in row.index: continue
                    val = normalize_na(row[field])
                    if val is None:
                        continue  # NA => unavailable, never negative
                    recs,ws = self.mapping_to_label_records(
                        sid,field,str(val),source_group=group,value=1
                    )
                    labels.extend(recs); warnings.extend(ws)
        return samples,labels,spatial,warnings
