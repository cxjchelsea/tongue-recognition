from pathlib import Path
import yaml
from .ontology import ALLOWED_MAPPING_STATUS, ALLOWED_SUPERVISION

class MappingRegistry:
    def __init__(self, ontology, mappings_dir):
        self.ontology = ontology
        self.mappings_dir = Path(mappings_dir)
    def load(self, filename):
        p = self.mappings_dir / filename
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    def validate_doc(self, data, strict=False):
        errors, warnings = [], []
        if not str(data.get("version","")):
            errors.append("mapping version missing")
        d = data.get("defaults", {})
        if d.get("supervision_tier") and d["supervision_tier"] not in ALLOWED_SUPERVISION:
            errors.append(f"illegal supervision_tier={d['supervision_tier']}")
        for name, src in data.get("sources", {}).items():
            if src.get("supervision_tier") not in ALLOWED_SUPERVISION:
                errors.append(f"{name}: illegal supervision_tier={src.get('supervision_tier')}")
        for source_key, item in data.get("mappings", {}).items():
            status = item.get("status")
            if status not in ALLOWED_MAPPING_STATUS:
                errors.append(f"{source_key}: illegal mapping status={status}")
                continue
            if status == "needs_review":
                (errors if strict else warnings).append(f"{source_key}: needs_review")
                continue
            if status == "excluded":
                continue
            task = item.get("canonical_task")
            if not task or not self.ontology.has_task(task):
                errors.append(f"{source_key}: unknown canonical_task={task}")
                continue
            labels = []
            for k in ("canonical_label","positive_label","negative_label"):
                if k in item:
                    labels.append(item[k])
            for label in labels:
                if not self.ontology.has_label(task, label):
                    errors.append(f"{source_key}: invalid label={label} for task={task}")
            for d in item.get("derive", []) or []:
                dt, dl = d.get("canonical_task"), d.get("canonical_label")
                if not self.ontology.has_task(dt) or not self.ontology.has_label(dt, dl):
                    errors.append(f"{source_key}: invalid derived target {dt}={dl}")
        return errors, warnings

def resolve_mapping_entry(doc, source_key, source_group=None):
    item = doc.get("mappings", {}).get(source_key)
    if item is None:
        return None
    merged = dict(doc.get("defaults", {}))
    if source_group:
        merged.update(doc.get("sources", {}).get(source_group, {}))
    merged.update(item)
    merged["mapping_version"] = str(doc.get("version",""))
    return merged
