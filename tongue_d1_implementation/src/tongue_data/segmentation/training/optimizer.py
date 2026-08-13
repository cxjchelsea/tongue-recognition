"""Optimizer / Scheduler 工厂。"""
from __future__ import annotations

import torch


def build_optimizer(model, optimizer_cfg: dict, lr_override: float | None = None):
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    learning_rate = float(lr_override if lr_override is not None else optimizer_cfg.get("lr", 1e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-4))
    if name != "adamw":
        raise ValueError(f"unsupported optimizer={name!r}; D3-B only supports adamw")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_cfg: dict):
    name = str(scheduler_cfg.get("name", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name != "reduce_on_plateau":
        raise ValueError(
            f"unsupported scheduler={name!r}; D3-B supports none|reduce_on_plateau"
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get("mode", "max")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
