from pathlib import Path
from abc import ABC, abstractmethod
from ..mapping import resolve_mapping_entry
from ..schema import LabelRecord
from ..utils import file_md5, image_size

class DatasetAdapter(ABC):
    def __init__(self, cfg, mapping_doc):
        self.cfg = cfg
        self.name = cfg["name"]
        self.root = Path(cfg["root"])
        self.mapping_doc = mapping_doc
    def validate_source(self):
        return [] if self.root.exists() else [f"{self.name}: root does not exist: {self.root}"]
    def make_sample_id(self, source_id):
        return f"{self.name}::{str(source_id).replace(chr(92), '/')}"
    def basic_image_meta(self, path):
        w, h = image_size(path)
        return file_md5(path), w, h
    @abstractmethod
    def collect(self):
        raise NotImplementedError
    def mapping_to_label_records(self, sample_id, source_field, source_label, source_group=None, value=None):
        source_key = f"{source_group}.{source_field}.{source_label}" if source_group else (
            str(source_label) if source_field is None else str(source_field)
        )
        item = resolve_mapping_entry(self.mapping_doc, source_key, source_group)
        if item is None:
            return [], [f"{self.name}: unmapped source label: {source_key}"]
        if item["status"] in {"excluded","needs_review"}:
            return [], [f"{self.name}: {source_key} -> {item['status']}"]
        task = item.get("canonical_task")
        label = item.get("canonical_label")
        if "positive_label" in item:
            if value is None:
                return [], [f"{self.name}: {source_field} missing value"]
            try:
                numeric = int(float(value))
            except Exception:
                numeric = 1 if str(value).strip().lower() in {"true","yes","positive"} else 0
            label = item["positive_label"] if numeric == 1 else item.get("negative_label")
            if numeric == 0 and label is None:
                return [], []
        recs = [LabelRecord(
            sample_id=sample_id, canonical_task=task, canonical_label=label, value=1,
            label_available=True, source_dataset=self.name, source_field=source_field,
            source_label=source_label, annotation_type=item.get("annotation_type","image_level"),
            label_source=item.get("label_source","dataset_annotation"),
            supervision_tier=item.get("supervision_tier","silver"),
            mapping_status=item["status"], mapping_version=item.get("mapping_version",""),
            note=item.get("note")
        )]
        for d in item.get("derive", []) or []:
            recs.append(LabelRecord(
                sample_id=sample_id, canonical_task=d["canonical_task"],
                canonical_label=d["canonical_label"], value=1, label_available=True,
                source_dataset=self.name, source_field=source_field, source_label=source_label,
                annotation_type=item.get("annotation_type","image_level"),
                label_source=item.get("label_source","dataset_annotation"),
                supervision_tier=item.get("supervision_tier","silver"),
                mapping_status=item["status"], mapping_version=item.get("mapping_version",""),
                note="derived_from_source_label"
            ))
        return recs, []
