"""Segmentation contract 配置加载。"""
from __future__ import annotations

from pathlib import Path

import yaml


class SegmentationConfig:
    """集中管理 D3-A 分割数据与训练契约。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.data = dict(self.doc.get("data", {}))
        self.image = dict(self.doc.get("image", {}))
        self.input = dict(self.doc.get("input", {}))
        self.resize = dict(self.doc.get("resize", {}))
        self.normalization = dict(self.doc.get("normalization", {}))
        self.mask = dict(self.doc.get("mask", {}))
        self.augmentation = dict(self.doc.get("augmentation", {}))
        self.reproducibility = dict(self.doc.get("reproducibility", {}))
        self.metrics = dict(self.doc.get("metrics", {}))
        self.loss = dict(self.doc.get("loss", {}))
        self.training_contract = dict(self.doc.get("training_contract", {}))
        self.acceptance_targets = dict(self.doc.get("acceptance_targets_d3bc", {}))
        self.foreground_ratio_warning = dict(self.doc.get("foreground_ratio_warning", {}))

    @property
    def seed(self) -> int:
        return int(self.reproducibility.get("seed", 0))

    @property
    def datasets(self) -> list[str]:
        return list(self.data.get("datasets", []))

    @property
    def task(self) -> str:
        return str(self.data.get("task", "segmentation.tongue"))

    @property
    def annotation_type(self) -> str:
        return str(self.data.get("annotation_type", "mask"))

    @property
    def input_height(self) -> int:
        return int(self.input.get("height", 384))

    @property
    def input_width(self) -> int:
        return int(self.input.get("width", 384))

    @property
    def mask_threshold(self) -> float:
        return float(self.metrics.get("mask_threshold", self.mask.get("threshold", 0.5)))

    @property
    def foreground_rule(self) -> str:
        return str(self.data.get("foreground_rule", "mask > 0"))
