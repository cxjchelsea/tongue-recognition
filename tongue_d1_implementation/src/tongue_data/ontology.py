from pathlib import Path
import yaml

ALLOWED_TASK_TYPES = {"binary","multilabel","multiclass_partial","ordinal","binary_segmentation"}
ALLOWED_MAPPING_STATUS = {"exact","compatible","partial","needs_review","excluded"}
ALLOWED_SUPERVISION = {"gold_candidate","silver","pseudo","weak","excluded"}

class Ontology:
    def __init__(self, path):
        self.path = Path(path)
        self.data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.tasks = self.data.get("tasks", {})
    @property
    def version(self):
        return str(self.data.get("version", ""))
    def has_task(self, task):
        return task in self.tasks
    def has_label(self, task, label):
        return task in self.tasks and label in self.tasks[task].get("labels", [])
    def validate(self):
        errors = []
        if not self.version:
            errors.append("ontology version missing")
        for task, spec in self.tasks.items():
            if spec.get("task_type") not in ALLOWED_TASK_TYPES:
                errors.append(f"{task}: illegal task_type={spec.get('task_type')}")
            labels = spec.get("labels")
            if not isinstance(labels, list) or not labels:
                errors.append(f"{task}: labels must be non-empty list")
            elif len(set(map(str, labels))) != len(labels):
                errors.append(f"{task}: duplicate labels")
        return errors
