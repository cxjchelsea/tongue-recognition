"""Stain detection / training 配置加载与 hash。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


class StainDataConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.task = str(self.doc.get("task", "quality.stain_suspected"))
        self.dataset = str(self.doc.get("dataset", "stained_coating"))
        self.input = dict(self.doc.get("input", {}))
        self.roi = dict(self.doc.get("roi", {}))
        self.labels = dict(self.doc.get("labels", {}))
        self._hash = hashlib.sha256(
            json.dumps(self.doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._hash

    @property
    def input_size(self) -> int:
        return int(self.input.get("size", 224))

    @property
    def mask_fill(self) -> int:
        return int(self.input.get("mask_outside_fill", 0))

    @property
    def min_roi_success_rate(self) -> float:
        return float(self.roi.get("min_roi_success_rate", 0.99))


class StainTrainConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.doc = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.doc.get("version", ""))
        self.data_contract = str(self.doc.get("data_contract", "configs/stain_detection_v1.yaml"))
        self.model = dict(self.doc.get("model", {}))
        self.loss = dict(self.doc.get("loss", {}))
        self.optimizer = dict(self.doc.get("optimizer", {}))
        self.scheduler = dict(self.doc.get("scheduler", {}))
        self.training = dict(self.doc.get("training", {}))
        self.early_stopping = dict(self.doc.get("early_stopping", {}))
        self.checkpoint = dict(self.doc.get("checkpoint", {}))
        self.augmentation = dict(self.doc.get("augmentation", {}))
        self.calibration = dict(self.doc.get("calibration", {}))
        self.gates = dict(self.doc.get("gates", {}))
        self.smoke = dict(self.doc.get("smoke", {}))
        self.overfit = dict(self.doc.get("overfit", {}))
        self.reproducibility = dict(self.doc.get("reproducibility", {}))
        self._hash = hashlib.sha256(
            json.dumps(self.doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._hash

    @property
    def seed(self) -> int:
        return int(self.reproducibility.get("seed", 20260813))

    @property
    def device(self) -> str:
        return str(self.training.get("device", "auto"))
