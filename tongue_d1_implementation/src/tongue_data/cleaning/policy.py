from pathlib import Path
import yaml


class CleaningPolicy:
    """配置化清洗策略；避免数据集规则散落在 if/else。"""

    def __init__(self, path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.global_cfg = dict(self.doc.get("global", {}))
        self.datasets = dict(self.doc.get("datasets", {}))

    def dataset_cfg(self, dataset_name: str) -> dict:
        if dataset_name not in self.datasets:
            raise KeyError(f"dataset missing in cleaning policy: {dataset_name}")
        return dict(self.datasets[dataset_name])

    def group_id(self, dataset_name: str, md5: str) -> str:
        template = self.global_cfg.get(
            "duplicate_group_id_format", "dup::{dataset}::{md5}"
        )
        return template.format(dataset=dataset_name, md5=md5)

    def tie_breakers(self):
        return list(
            self.global_cfg.get(
                "canonical_tie_breakers",
                ["source_image_path", "source_sample_id", "sample_id"],
            )
        )
