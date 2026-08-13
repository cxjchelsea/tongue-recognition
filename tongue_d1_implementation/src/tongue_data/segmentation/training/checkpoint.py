"""Checkpoint save / load / resume。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_dice: float,
    config_dict: dict,
    config_hash: str,
    seed: int,
    history: list[dict],
    extra: dict | None = None,
):
    """保存完整训练状态（不仅 model_state_dict）。"""
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_dice": float(best_val_dice),
        "config": config_dict,
        "config_hash": config_hash,
        "seed": int(seed),
        "training_history": list(history),
        "torch_version": torch.__version__,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(
    checkpoint: dict,
    *,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
) -> dict:
    """恢复 model/optimizer/scheduler/scaler；返回 epoch/best 等元信息。"""
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_val_dice": float(checkpoint.get("best_val_dice", 0.0)),
        "config_hash": checkpoint.get("config_hash"),
        "seed": checkpoint.get("seed"),
        "training_history": list(checkpoint.get("training_history", [])),
    }


def write_run_metadata(path: str | Path, metadata: dict):
    Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
