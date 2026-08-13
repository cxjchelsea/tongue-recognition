from pathlib import Path
import yaml


class SplitPolicy:
    """配置化 split 策略。"""

    def __init__(self, path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.global_cfg = dict(self.doc.get("global", {}))
        self.datasets = dict(self.doc.get("datasets", {}))

    @property
    def seed(self) -> int:
        return int(self.global_cfg.get("seed", 0))

    def target_ratios(self) -> dict:
        return {str(key): float(value) for key, value in dict(self.global_cfg.get("target_ratios", {})).items()}

    def dataset_cfg(self, dataset_name: str) -> dict:
        if dataset_name not in self.datasets:
            raise KeyError(f"dataset missing in split policy: {dataset_name}")
        return dict(self.datasets[dataset_name])

    def core_tasks(self) -> list[str]:
        return list(self.global_cfg.get("core_tasks", []))

    def exclude_pools(self) -> set[str]:
        strat = self.global_cfg.get("stratification", {})
        return set(strat.get("exclude_pools", ["pseudo", "external_holdout"]))

    def missing_patient_policy(self) -> str:
        return str(self.global_cfg.get("missing_patient_policy", "fallback_to_sample_and_warn"))
