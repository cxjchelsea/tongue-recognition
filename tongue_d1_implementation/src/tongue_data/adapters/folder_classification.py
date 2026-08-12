from .base import DatasetAdapter
from ..schema import SampleRecord
from ..utils import is_image, infer_split

class FolderClassificationAdapter(DatasetAdapter):
    def collect(self):
        samples, labels, spatial, warnings = [], [], [], []
        for source_label, rel_dir in self.cfg.get("class_dirs",{}).items():
            class_dir = self.root / rel_dir
            if not class_dir.exists():
                warnings.append(f"{self.name}: class dir missing: {class_dir}")
                continue
            for img in class_dir.rglob("*"):
                if not img.is_file() or not is_image(img):
                    continue
                source_id = str(img.relative_to(self.root)).replace("\\","/")
                sid = self.make_sample_id(source_id)
                md5,wid,hei = self.basic_image_meta(img)
                samples.append(SampleRecord(
                    sid,self.name,source_id,str(img),md5,wid,hei,
                    source_split=infer_split(img)
                ))
                recs,ws = self.mapping_to_label_records(
                    sid,None,str(source_label),value=1
                )
                labels.extend(recs); warnings.extend(ws)
        return samples,labels,spatial,warnings
