from pathlib import Path
import pandas as pd
from .base import DatasetAdapter
from ..schema import SampleRecord
from ..utils import normalize_na

class TongueDxAdapter(DatasetAdapter):
    def collect(self):
        samples,labels,spatial,warnings = [],[],[],[]
        csv_path = self.root / self.cfg["csv"]
        if not csv_path.exists():
            return samples,labels,spatial,[f"{self.name}: csv missing: {csv_path}"]
        df = pd.read_csv(csv_path)
        image_col = self.cfg.get("image_path_column","image_path")
        patient_col = self.cfg.get("patient_id_column","id")
        seen = set()
        for _,row in df.iterrows():
            raw = normalize_na(row.get(image_col))
            if raw is None:
                continue
            img = Path(str(raw))
            if not img.is_absolute():
                img = self.root / img
            if not img.exists():
                warnings.append(f"{self.name}: image missing: {img}")
                continue
            source_id = str(img.relative_to(self.root)).replace("\\","/") if self.root in img.parents else img.name
            sid = self.make_sample_id(source_id)
            if sid not in seen:
                md5,wid,hei = self.basic_image_meta(img)
                patient = normalize_na(row.get(patient_col))
                samples.append(SampleRecord(
                    sid,self.name,source_id,str(img),md5,wid,hei,
                    patient_id=None if patient is None else str(patient),
                    patient_id_available=patient is not None
                ))
                seen.add(sid)
            for col in self.cfg.get("label_columns",[]):
                v = normalize_na(row.get(col))
                if v is None:
                    continue
                recs,ws = self.mapping_to_label_records(sid,col,col,value=v)
                labels.extend(recs); warnings.extend(ws)
        return samples,labels,spatial,warnings
