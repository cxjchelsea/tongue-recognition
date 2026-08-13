"""可复现性：seed / device / 环境记录。"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    """覆盖 Python / NumPy / PyTorch / CUDA / 环境 hash seed。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """device=auto 时优先 CUDA。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def dataloader_worker_init_fn(worker_id: int, base_seed: int = 0):
    """DataLoader worker 种子。"""
    worker_seed = int(base_seed) + int(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:
        pass


def config_hash(config_path: str | Path) -> str:
    content = Path(config_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def environment_record(config_path: str | Path, seed: int, device: str = "auto") -> dict:
    record = {
        "seed": int(seed),
        "device": resolve_device(device),
        "config_hash": config_hash(config_path),
        "numpy_version": np.__version__,
    }
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        record["torch_version"] = None
        record["cuda_available"] = False
    return record


def dumps_env(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)
