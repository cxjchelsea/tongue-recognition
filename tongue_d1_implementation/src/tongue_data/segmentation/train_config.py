"""D3-B 训练契约加载与 config hash。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


class TrainConfig:
    """segmentation_train_v1 配置。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_text(encoding="utf-8")
        self.doc = yaml.safe_load(raw)
        self.version = str(self.doc.get("version", ""))
        self.model = dict(self.doc.get("model", {}))
        self.loss = dict(self.doc.get("loss", {}))
        self.optimizer = dict(self.doc.get("optimizer", {}))
        self.scheduler = dict(self.doc.get("scheduler", {}))
        self.training = dict(self.doc.get("training", {}))
        self.early_stopping = dict(self.doc.get("early_stopping", {}))
        self.checkpoint = dict(self.doc.get("checkpoint", {}))
        self.metrics = dict(self.doc.get("metrics", {}))
        self.reproducibility = dict(self.doc.get("reproducibility", {}))
        self.debug = dict(self.doc.get("debug", {}))
        self.smoke = dict(self.doc.get("smoke", {}))
        self.data_contract = str(self.doc.get("data_contract", "configs/segmentation_v1.yaml"))
        self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        # 规范化 YAML 结构后哈希，保证 deterministic
        normalized = json.dumps(self.doc, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._hash

    @property
    def seed(self) -> int:
        return int(self.reproducibility.get("seed", 0))

    @property
    def mask_threshold(self) -> float:
        return float(self.metrics.get("mask_threshold", 0.5))

    @property
    def device(self) -> str:
        return str(self.training.get("device", "auto"))
