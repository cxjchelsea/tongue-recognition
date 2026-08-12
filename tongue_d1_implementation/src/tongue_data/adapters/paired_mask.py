from .base import DatasetAdapter
from ..schema import SampleRecord, SpatialRecord
from ..mapping import resolve_mapping_entry
from ..utils import is_image, infer_split

class PairedMaskAdapter(DatasetAdapter):
    def collect(self):
        samples, labels, spatial, warnings = [], [], [], []
        images_dir = self.root / self.cfg["images_dir"]
        masks_dir = self.root / self.cfg["masks_dir"]
        if not images_dir.exists() or not masks_dir.exists():
            return samples, labels, spatial, [f"{self.name}: images_dir or masks_dir missing"]
        mapping = resolve_mapping_entry(self.mapping_doc, "tongue_mask")
        image_files = [p for p in images_dir.rglob("*") if p.is_file() and is_image(p)]
        mask_by_stem = {p.stem:p for p in masks_dir.rglob("*") if p.is_file() and is_image(p)}
        for img in image_files:
            source_id = str(img.relative_to(images_dir)).replace("\\","/")
            sid = self.make_sample_id(source_id)
            md5,wid,hei = self.basic_image_meta(img)
            samples.append(SampleRecord(
                sid,self.name,source_id,str(img),md5,wid,hei,
                source_split=infer_split(img)
            ))
            mask = mask_by_stem.get(img.stem)
            if not mask:
                warnings.append(f"{self.name}: missing mask for {img.name}")
                continue
            spatial.append(SpatialRecord(
                sample_id=sid, annotation_id=f"{sid}::mask",
                annotation_task=mapping.get("canonical_task"),
                canonical_label=mapping.get("canonical_label"),
                annotation_type="mask",
                x_min=None,y_min=None,x_max=None,y_max=None,
                mask_path=str(mask), source_dataset=self.name,
                source_label="tongue_mask",
                label_source=mapping.get("label_source","human_mask"),
                supervision_tier=mapping.get("supervision_tier","gold_candidate"),
                mapping_status=mapping.get("status","exact"),
                mapping_version=mapping.get("mapping_version","")
            ))
        return samples,labels,spatial,warnings
