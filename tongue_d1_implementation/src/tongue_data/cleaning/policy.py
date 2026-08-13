from pathlib import Path
import yaml

# D2-A.1 当前明确支持的标签冲突策略
ALLOWED_CONFLICT_POLICIES = {"drop_conflicted_facts_from_clean"}
ALLOWED_SPATIAL_GEOMETRY_POLICIES = {"multi_instance_keep"}


class CleaningPolicy:
    """配置化清洗策略；避免数据集规则散落在 if/else。"""

    def __init__(self, path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.global_cfg = dict(self.doc.get("global", {}))
        self.datasets = dict(self.doc.get("datasets", {}))
        self.validate_supported_policies()

    def validate_supported_policies(self):
        """未知 policy 必须 fail-fast，禁止静默回退。"""
        conflict_policy = self.conflict_policy()
        if conflict_policy not in ALLOWED_CONFLICT_POLICIES:
            raise ValueError(
                f"unsupported conflict_policy={conflict_policy!r}; "
                f"allowed={sorted(ALLOWED_CONFLICT_POLICIES)}"
            )
        spatial_policy = self.spatial_different_geometry_policy()
        if spatial_policy not in ALLOWED_SPATIAL_GEOMETRY_POLICIES:
            raise ValueError(
                f"unsupported spatial_different_geometry_policy={spatial_policy!r}; "
                f"allowed={sorted(ALLOWED_SPATIAL_GEOMETRY_POLICIES)}"
            )

    def conflict_policy(self) -> str:
        return str(
            self.global_cfg.get("conflict_policy")
            or self.global_cfg.get("label_conflict_policy")
            or ""
        )

    def spatial_different_geometry_policy(self) -> str:
        return str(
            self.global_cfg.get("spatial_different_geometry_policy", "multi_instance_keep")
        )

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
