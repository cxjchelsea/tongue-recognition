from dataclasses import dataclass, asdict
from typing import Optional, Any

SAMPLE_COLUMNS = [
    "sample_id","dataset","source_sample_id","source_image_path","md5","width","height",
    "patient_id","patient_id_available","source_split","duplicate_group_id",
    "dataset_version","ingest_version"
]
LABEL_COLUMNS = [
    "sample_id","canonical_task","canonical_label","value","label_available",
    "source_dataset","source_field","source_label","annotation_type","label_source",
    "supervision_tier","mapping_status","mapping_version","confidence","note"
]
SPATIAL_COLUMNS = [
    "sample_id","annotation_id","annotation_task","canonical_label","annotation_type",
    "x_min","y_min","x_max","y_max","mask_path","source_dataset","source_label",
    "label_source","supervision_tier","mapping_status","mapping_version","note"
]

@dataclass
class SampleRecord:
    sample_id: str
    dataset: str
    source_sample_id: str
    source_image_path: str
    md5: str
    width: int
    height: int
    patient_id: Optional[str] = None
    patient_id_available: bool = False
    source_split: Optional[str] = None
    duplicate_group_id: Optional[str] = None
    dataset_version: Optional[str] = None
    ingest_version: str = "1.1"
    def to_dict(self): return asdict(self)

@dataclass
class LabelRecord:
    sample_id: str
    canonical_task: Optional[str]
    canonical_label: Any
    value: Any
    label_available: bool
    source_dataset: str
    source_field: Optional[str]
    source_label: Any
    annotation_type: str
    label_source: str
    supervision_tier: str
    mapping_status: str
    mapping_version: str
    confidence: Optional[float] = None
    note: Optional[str] = None
    def to_dict(self): return asdict(self)

@dataclass
class SpatialRecord:
    sample_id: str
    annotation_id: str
    annotation_task: Optional[str]
    canonical_label: Any
    annotation_type: str
    x_min: Optional[float]
    y_min: Optional[float]
    x_max: Optional[float]
    y_max: Optional[float]
    mask_path: Optional[str]
    source_dataset: str
    source_label: Any
    label_source: str
    supervision_tier: str
    mapping_status: str
    mapping_version: str
    note: Optional[str] = None
    def to_dict(self): return asdict(self)
