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
        width, height = image_size(path)
        return file_md5(path), width, height

    @abstractmethod
    def collect(self):
        raise NotImplementedError

    def mapping_to_label_records(self, sample_id, source_field, source_label, source_group=None, value=None):
        """将源标签转为 canonical label rows。

        三种监督语义：
        - positive：value=1
        - explicit negative：
          * binary（配置了 negative_label）：canonical_label=negative_label, value=1
          * partial attribute（仅 positive_label）：canonical_label=positive_label, value=0
        - missing（NA）：调用方不传入有效 value，不生成记录
        """
        source_key = f"{source_group}.{source_field}.{source_label}" if source_group else (
            str(source_label) if source_field is None else str(source_field)
        )
        item = resolve_mapping_entry(self.mapping_doc, source_key, source_group)
        if item is None:
            return [], [f"{self.name}: unmapped source label: {source_key}"]
        # excluded 为预期行为，不刷 warning；needs_review 才需要人工处理
        if item["status"] == "excluded":
            return [], []
        if item["status"] == "needs_review":
            return [], [f"{self.name}: {source_key} -> needs_review"]

        task = item.get("canonical_task")
        label = item.get("canonical_label")
        record_value = 1
        note = item.get("note")

        if "positive_label" in item:
            if value is None:
                return [], [f"{self.name}: {source_field} missing value"]
            try:
                numeric = int(float(value))
            except Exception:
                numeric = 1 if str(value).strip().lower() in {"true", "yes", "positive"} else 0

            if numeric == 1:
                # 明确正监督
                label = item["positive_label"]
                record_value = 1
            elif "negative_label" in item:
                # binary task：沿用 false/true 标签，value 恒为 1 表示“断言该标签”
                label = item["negative_label"]
                record_value = 1
            else:
                # partial multiclass / attribute：保留 positive_label，value=0 表示显式否定
                # 例如 pale=0 不是 normal，yellow=0 不是 white
                label = item["positive_label"]
                record_value = 0
                note = "explicit_negative" if not note else f"{note}; explicit_negative"

        recs = [LabelRecord(
            sample_id=sample_id, canonical_task=task, canonical_label=label, value=record_value,
            label_available=True, source_dataset=self.name, source_field=source_field,
            source_label=source_label, annotation_type=item.get("annotation_type", "image_level"),
            label_source=item.get("label_source", "dataset_annotation"),
            supervision_tier=item.get("supervision_tier", "silver"),
            mapping_status=item["status"], mapping_version=item.get("mapping_version", ""),
            note=note,
        )]
        for derived in item.get("derive", []) or []:
            # derive 仅用于正向证据衍生（如 severity→present）；负监督不自动衍生
            if record_value != 1:
                continue
            recs.append(LabelRecord(
                sample_id=sample_id, canonical_task=derived["canonical_task"],
                canonical_label=derived["canonical_label"], value=1, label_available=True,
                source_dataset=self.name, source_field=source_field, source_label=source_label,
                annotation_type=item.get("annotation_type", "image_level"),
                label_source=item.get("label_source", "dataset_annotation"),
                supervision_tier=item.get("supervision_tier", "silver"),
                mapping_status=item["status"], mapping_version=item.get("mapping_version", ""),
                note="derived_from_source_label",
            ))
        return recs, []
