"""Training history 序列化。"""
from __future__ import annotations

import json
from pathlib import Path


class TrainingHistory:
    def __init__(self):
        self.epochs: list[dict] = []

    def append(self, record: dict):
        self.epochs.append(dict(record))

    def to_list(self) -> list[dict]:
        return list(self.epochs)

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps({"epochs": self.epochs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        history = cls()
        history.epochs = list(doc.get("epochs", []))
        return history
