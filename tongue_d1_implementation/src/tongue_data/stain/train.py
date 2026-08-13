"""D4-C stain binary classifier 训练：tiny overfit / smoke / full。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import StainDataConfig, StainTrainConfig
from .dataset import StainRoiDataset, create_stain_dataloader, select_overfit_subset
from .manifest import STAIN_CONTRACT_VERSION, class_balance_report
from .metrics import pr_auc_score, roc_auc_score, summarize_ranking_metrics
from .model import build_stain_model, count_parameters


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "md5": [item["md5"] for item in batch],
    }


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        logits = model(images)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": batch["split"][index],
                    "md5": batch["md5"][index],
                    "label": float(labels[index]),
                    "logit": float(logits[index].detach().cpu()),
                    "p_stain": float(probs[index]),
                }
            )
    return pd.DataFrame(rows)


def _epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    losses: list[float] = []
    correct = 0
    total = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite train loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += int(labels.numel())
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "train_accuracy": float(correct / total) if total else 0.0,
    }


@torch.inference_mode()
def _epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    all_labels: list[float] = []
    all_scores: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            loss = criterion(logits, labels)
        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    ranking = summarize_ranking_metrics(all_labels, all_scores)
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_auroc": ranking["auroc"],
        "val_pr_auc": ranking["pr_auc"],
        "val_accuracy@0.5": ranking["at_0.5_accuracy"],
        "val_precision@0.5": ranking["at_0.5_precision"],
        "val_recall@0.5": ranking["at_0.5_recall"],
        "val_specificity@0.5": ranking["at_0.5_specificity"],
        "val_f1@0.5": ranking["at_0.5_f1"],
        "n_val": len(all_labels),
    }


def save_stain_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_auroc: float | None,
    train_config: StainTrainConfig,
    data_config: StainDataConfig,
    history: list[dict],
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val_auroc": best_val_auroc,
        "architecture": train_config.model.get("architecture", "resnet18"),
        "train_config_hash": train_config.config_hash,
        "data_config_hash": data_config.config_hash,
        "stain_contract_version": STAIN_CONTRACT_VERSION,
        "input_size": data_config.input_size,
        "seed": train_config.seed,
        "history": history,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_stain_checkpoint(
    path: str | Path,
    *,
    train_config: StainTrainConfig,
    data_config: StainDataConfig,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    expected_arch = str(train_config.model.get("architecture", "resnet18"))
    if ckpt.get("architecture") != expected_arch:
        raise ValueError(
            f"architecture mismatch ckpt={ckpt.get('architecture')} cfg={expected_arch}"
        )
    if strict:
        if ckpt.get("train_config_hash") != train_config.config_hash:
            raise ValueError(
                f"train config hash mismatch ckpt={ckpt.get('train_config_hash')} "
                f"cfg={train_config.config_hash}"
            )
        if ckpt.get("data_config_hash") != data_config.config_hash:
            raise ValueError(
                f"data config hash mismatch ckpt={ckpt.get('data_config_hash')} "
                f"cfg={data_config.config_hash}"
            )
    model = build_stain_model(train_config.model)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"strict load failed missing={missing} unexpected={unexpected}")
    return model, ckpt


def run_tiny_overfit(
    *,
    manifest_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    set_seed(train_config.seed)
    device_obj = resolve_device(device if device != "auto" else train_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    subset = select_overfit_subset(
        manifest,
        positives=int(train_config.overfit.get("positives", 8)),
        negatives=int(train_config.overfit.get("negatives", 8)),
    )
    # 临时把子集当成 train split
    subset = subset.copy()
    subset["split"] = "train"
    dataset = StainRoiDataset(
        subset,
        data_config,
        train_config,
        split="train",
        disable_augmentation=True,
        seed=train_config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(16, len(dataset)),
        shuffle=True,
        collate_fn=_collate,
    )
    model = build_stain_model(train_config.model).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_config.optimizer.get("weight_decay", 1e-4)),
    )
    epochs = int(train_config.overfit.get("epochs", 40))
    history = []
    first_loss = None
    for epoch in range(1, epochs + 1):
        stats = _epoch_train(
            model, loader, optimizer, criterion, device_obj, amp=False
        )
        if first_loss is None:
            first_loss = stats["train_loss"]
        history.append({"epoch": epoch, **stats})
        if stats["train_accuracy"] >= float(train_config.overfit.get("accuracy_gate", 0.95)):
            break
    final = history[-1]
    loss_drop = (first_loss - final["train_loss"]) if first_loss is not None else 0.0
    gate = float(train_config.overfit.get("accuracy_gate", 0.95))
    passed = final["train_accuracy"] >= gate and loss_drop > 0.1
    report = {
        "status": "PASS" if passed else "FAIL",
        "epochs_ran": len(history),
        "final_accuracy": final["train_accuracy"],
        "final_loss": final["train_loss"],
        "first_loss": first_loss,
        "loss_drop": float(loss_drop),
        "accuracy_gate": gate,
        "sample_count": len(dataset),
        "history": history,
    }
    (output_dir / "overfit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            f"tiny overfit FAIL accuracy={final['train_accuracy']:.3f} "
            f"loss_drop={loss_drop:.4f}"
        )
    return report


def run_stain_training(
    *,
    manifest_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    smoke: bool = False,
    allow_test_in_training: bool = False,
) -> dict[str, Any]:
    """正式 train；test 绝不参与。"""
    if allow_test_in_training:
        raise ValueError("test must never participate in stain training")
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    set_seed(train_config.seed)
    device_obj = resolve_device(device if device != "auto" else train_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    eligible = manifest[manifest["eligible"] == True].copy()
    balance = class_balance_report(eligible)
    # 接近均衡：不使用 pos_weight
    train_rate = balance["splits"]["train"]["positive_rate"]
    if train_rate is not None and (train_rate < 0.35 or train_rate > 0.65):
        imbalance_note = "imbalance_detected_no_auto_weighting"
    else:
        imbalance_note = "near_balanced_no_pos_weight"

    train_ds = StainRoiDataset(
        eligible, data_config, train_config, split="train", seed=train_config.seed
    )
    val_ds = StainRoiDataset(
        eligible, data_config, train_config, split="val", seed=train_config.seed
    )
    # 显式禁止构建 test loader
    test_count = int(((eligible["split"] == "test") & (eligible["eligible"] == True)).sum())

    batch_size = int(train_config.training.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_config.training.get("num_workers", 0)),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_config.training.get("num_workers", 0)),
        collate_fn=_collate,
    )

    model = build_stain_model(train_config.model).to(device_obj)
    params = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_config.optimizer.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(train_config.scheduler.get("mode", "max")),
        factor=float(train_config.scheduler.get("factor", 0.5)),
        patience=int(train_config.scheduler.get("patience", 3)),
        min_lr=float(train_config.scheduler.get("min_lr", 1e-6)),
    )

    planned_epochs = int(
        train_config.smoke.get("epochs", 2)
        if smoke
        else train_config.training.get("epochs", 30)
    )
    max_train_batches = (
        int(train_config.smoke.get("max_train_batches", 5)) if smoke else None
    )
    max_val_batches = (
        int(train_config.smoke.get("max_val_batches", 5)) if smoke else None
    )
    amp = bool(train_config.training.get("amp", True)) and device_obj.type == "cuda"
    early_cfg = train_config.early_stopping
    early_enabled = bool(early_cfg.get("enabled", True)) and not smoke
    patience = int(early_cfg.get("patience", 7))

    history: list[dict[str, Any]] = []
    best_val_auroc = -1.0
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, planned_epochs + 1):
        train_stats = _epoch_train(
            model,
            train_loader,
            optimizer,
            criterion,
            device_obj,
            amp=amp,
            max_batches=max_train_batches,
        )
        val_stats = _epoch_val(
            model,
            val_loader,
            criterion,
            device_obj,
            amp=amp,
            max_batches=max_val_batches,
        )
        row = {"epoch": epoch, **train_stats, **val_stats}
        history.append(row)
        val_auroc = val_stats["val_auroc"]
        if val_auroc is None:
            val_auroc = -1.0
        scheduler.step(float(val_auroc))

        save_stain_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_auroc=best_val_auroc if best_val_auroc >= 0 else None,
            train_config=train_config,
            data_config=data_config,
            history=history,
        )
        # best 仅由 val AUROC 决定；test 不参与
        if float(val_auroc) > best_val_auroc + 1e-12:
            best_val_auroc = float(val_auroc)
            best_epoch = epoch
            epochs_without_improve = 0
            save_stain_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_auroc=best_val_auroc,
                train_config=train_config,
                data_config=data_config,
                history=history,
                extra={"selection_metric": "val_auroc", "test_used": False},
            )
        else:
            epochs_without_improve += 1
            if early_enabled and epochs_without_improve >= patience:
                break

    if not (output_dir / "best.pt").exists():
        raise RuntimeError("best.pt missing after training")

    # val predictions from best
    best_model, best_ckpt = load_stain_checkpoint(
        output_dir / "best.pt",
        train_config=train_config,
        data_config=data_config,
        map_location=device_obj,
        strict=True,
    )
    best_model = best_model.to(device_obj)
    val_pred = collect_predictions(best_model, val_loader, device_obj)
    val_pred.to_parquet(output_dir / "val_predictions.parquet", index=False)

    metadata = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": _git_commit(),
        "stain_contract_version": STAIN_CONTRACT_VERSION,
        "data_config_hash": data_config.config_hash,
        "train_config_hash": train_config.config_hash,
        "d3_checkpoint_hash": (
            str(eligible["d3_checkpoint_hash"].dropna().iloc[0])
            if eligible["d3_checkpoint_hash"].notna().any()
            else None
        ),
        "seed": train_config.seed,
        "model": train_config.model.get("architecture"),
        "parameters": params,
        "input_size": data_config.input_size,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count_available_not_used": test_count,
        "class_balance": balance,
        "imbalance_note": imbalance_note,
        "device": str(device_obj),
        "gpu": torch.cuda.get_device_name(0) if device_obj.type == "cuda" else None,
        "planned_epochs": planned_epochs,
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_val_pr_auc": next(
            (
                item.get("val_pr_auc")
                for item in history
                if item.get("epoch") == best_epoch
            ),
            None,
        ),
        "smoke": smoke,
        "checkpoint_selection": "val_auroc",
        "test_used_in_training": False,
        "test_used_in_checkpoint_selection": False,
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
