import yaml
from .base import DatasetAdapter
from ..schema import SampleRecord, SpatialRecord, LabelRecord
from ..mapping import resolve_mapping_entry
from ..utils import infer_split

class TMCYoloAdapter(DatasetAdapter):
    def _names(self):
        candidates = []
        if self.cfg.get("classes_yaml"):
            candidates.append(self.root/self.cfg["classes_yaml"])
        candidates.extend(self.root.rglob("*.yaml"))
        for p in candidates:
            if not p.exists(): continue
            try:
                d = yaml.safe_load(p.read_text(encoding="utf-8"))
                names = d.get("names")
                if isinstance(names,list): return {i:n for i,n in enumerate(names)}
                if isinstance(names,dict): return {int(k):v for k,v in names.items()}
            except Exception:
                pass
        return {}

    def collect(self):
        samples,labels,spatial,warnings = [],[],[],[]
        names = self._names()
        if not names:
            warnings.append(f"{self.name}: class names YAML not found")
        label_idx = {}
        for p in self.root.glob(self.cfg.get("labels_glob","**/labels/**/*.txt")):
            if p.is_file():
                label_idx.setdefault(p.stem,[]).append(p)
        allowed = set(x.lower() for x in self.cfg.get("allowed_image_ext",[".jpg",".jpeg",".png"]))
        imgs = [p for p in self.root.glob(self.cfg.get("images_glob","**/images/**/*"))
                if p.is_file() and p.suffix.lower() in allowed]
        for img in imgs:
            source_id = str(img.relative_to(self.root)).replace("\\","/")
            sid = self.make_sample_id(source_id)
            md5,wid,hei = self.basic_image_meta(img)
            samples.append(SampleRecord(
                sid,self.name,source_id,str(img),md5,wid,hei,
                source_split=infer_split(img)
            ))
            choices = label_idx.get(img.stem,[])
            if not choices: continue
            same = [p for p in choices if infer_split(p)==infer_split(img)]
            lab = same[0] if same else choices[0]
            for line_no,line in enumerate(lab.read_text(encoding="utf-8",errors="ignore").splitlines(),1):
                parts = line.split()
                if len(parts)<5: continue
                try:
                    cls = int(float(parts[0])); xc,yc,bw,bh = map(float,parts[1:5])
                except Exception:
                    warnings.append(f"{self.name}: bad YOLO row {lab}:{line_no}")
                    continue
                source_label = names.get(cls,str(cls))
                item = resolve_mapping_entry(self.mapping_doc,str(source_label))
                if item is None:
                    warnings.append(f"{self.name}: unmapped class {source_label}")
                    continue
                if item["status"] in {"excluded","needs_review"}:
                    warnings.append(f"{self.name}: {source_label} -> {item['status']}")
                    continue
                x1=max(0,(xc-bw/2)*wid); y1=max(0,(yc-bh/2)*hei)
                x2=min(wid,(xc+bw/2)*wid); y2=min(hei,(yc+bh/2)*hei)
                spatial.append(SpatialRecord(
                    sid,f"{sid}::{line_no}",item["canonical_task"],item["canonical_label"],
                    "bbox",x1,y1,x2,y2,None,self.name,source_label,
                    item.get("label_source","dataset_annotation"),
                    item.get("supervision_tier","silver"),item["status"],
                    item.get("mapping_version",""),item.get("note")
                ))
                labels.append(LabelRecord(
                    sid,item["canonical_task"],item["canonical_label"],1,True,
                    self.name,"yolo_bbox",source_label,"derived_image_level_from_bbox",
                    item.get("label_source","dataset_annotation"),
                    item.get("supervision_tier","silver"),item["status"],
                    item.get("mapping_version",""),None,
                    "positive evidence derived from bbox; absence is NOT inferred"
                ))
        return samples,labels,spatial,warnings
