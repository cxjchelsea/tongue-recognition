"""Frozen split evaluation：与 validation 共用 metric 定义。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import SegmentationConfig
from ..dataset import TongueSegmentationDataset, create_dataloader
from ..model import build_segmentation_model, count_parameters
from ..reproducibility import resolve_device, seed_everything
from ..train_config import TrainConfig
from .checkpoint import load_checkpoint, write_run_metadata
from .evaluation import batch_dice_iou_precision_recall
from .losses import build_loss


def _summarize_scores(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p90": None,
            "count": 0,
        }
    array = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "count": int(len(array)),
    }


def _bucket_by_foreground(ratios: list[float]) -> tuple[float, float]:
    """返回 (p33, p66) 分位阈值，用于 small/medium/large。"""
    array = np.asarray(ratios, dtype=np.float64)
    return float(np.percentile(array, 33)), float(np.percentile(array, 66))


def decide_baseline_gate(
    overall_dice: float,
    biohit_dice: float,
    tongueset3_dice: float,
    *,
    overall_target: float = 0.95,
    overall_minimum: float = 0.90,
    domain_minimum: float = 0.90,
) -> dict[str, Any]:
    """工程 gate（非临床声明）。"""
    domain_ok = biohit_dice >= domain_minimum and tongueset3_dice >= domain_minimum
    if overall_dice >= overall_target and domain_ok:
        status = "TARGET_PASS"
    elif overall_dice >= overall_minimum and domain_ok:
        status = "MINIMUM_PASS"
    elif overall_dice >= overall_minimum and not domain_ok:
        status = "OVERALL_PASS_DOMAIN_FAIL"
    else:
        status = "NEEDS_IMPROVEMENT"
    return {
        "baseline_status": status,
        "overall_dice": float(overall_dice),
        "biohit_dice": float(biohit_dice),
        "tongueset3_dice": float(tongueset3_dice),
        "gates": {
            "overall_target": overall_target,
            "overall_minimum": overall_minimum,
            "domain_minimum": domain_minimum,
        },
        "domain_ok": bool(domain_ok),
        "note": "engineering baseline gate; not a clinical claim",
    }


def verify_checkpoint_integrity(
    checkpoint: dict,
    train_config: TrainConfig,
    *,
    expected_architecture: str = "unet",
) -> list[str]:
    """best.pt 完整性检查；返回 errors。"""
    errors = []
    if "model_state_dict" not in checkpoint:
        errors.append("missing model_state_dict")
    if checkpoint.get("config_hash") != train_config.config_hash:
        errors.append(
            f"config_hash mismatch: ckpt={checkpoint.get('config_hash')} "
            f"current={train_config.config_hash}"
        )
    if int(checkpoint.get("seed", -1)) != int(train_config.seed):
        errors.append(
            f"seed mismatch: ckpt={checkpoint.get('seed')} current={train_config.seed}"
        )
    cfg = checkpoint.get("config") or {}
    model_cfg = cfg.get("model") or {}
    architecture = str(model_cfg.get("architecture", "")).lower()
    if architecture and architecture != expected_architecture:
        errors.append(f"architecture mismatch: {architecture}")
    monitor = str((cfg.get("checkpoint") or {}).get("monitor", "val_dice"))
    if monitor != "val_dice":
        errors.append(f"checkpoint monitor must be val_dice, got {monitor}")
    return errors


@torch.no_grad()
def evaluate_checkpoint_on_split(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    split: str,
    output_dir: str | Path | None = None,
    allow_test: bool = False,
) -> dict:
    """
    对指定 split 做 frozen evaluation。
    split=test 时必须 allow_test=True，且训练流程不得调用本函数。
    """
    split_name = str(split)
    if split_name == "test" and not allow_test:
        raise RuntimeError("test evaluation requires allow_test=True after baseline freeze")
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split_name}")

    train_config = TrainConfig(train_config_path)
    data_config = SegmentationConfig(data_config_path)
    seed_everything(train_config.seed)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    integrity_errors = verify_checkpoint_integrity(checkpoint, train_config)
    if integrity_errors:
        raise ValueError(f"checkpoint integrity failed: {integrity_errors}")

    device_name = resolve_device(train_config.device)
    device = torch.device(device_name)
    model = build_segmentation_model(train_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = build_loss(train_config.loss)
    threshold = float(
        train_config.doc.get("evaluation", {}).get(
            "threshold", train_config.mask_threshold
        )
    )
    if abs(threshold - 0.5) > 1e-12:
        # D3-C：禁止偏离 0.5 的隐藏 threshold
        raise ValueError(f"D3-C baseline threshold must be 0.5, got {threshold}")

    manifest = pd.read_parquet(Path(segmentation_dir) / "segmentation_manifest.parquet")
    dataset = TongueSegmentationDataset(
        manifest,
        data_config,
        split=split_name,
        seed=train_config.seed,
        disable_augmentation=True,
    )
    batch_size = int(train_config.training.get("batch_size", 4))
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    per_image_rows = []
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        if logits.shape != masks.shape:
            raise ValueError(
                f"shape mismatch logits={tuple(logits.shape)} mask={tuple(masks.shape)}"
            )
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite eval loss: {loss}")
        metrics = batch_dice_iou_precision_recall(
            logits.float(), masks.float(), threshold=threshold
        )
        total_loss += float(loss.item())
        total_batches += 1

        batch_size_now = int(images.shape[0])
        for index in range(batch_size_now):
            probability = torch.sigmoid(logits[index].float())
            binary = (probability >= threshold).float()
            pred_ratio = float(binary.mean().cpu().item())
            sample_id = (
                batch["sample_id"][index]
                if not isinstance(batch["sample_id"], str)
                else batch["sample_id"]
            )
            dataset_name = (
                batch["dataset"][index]
                if not isinstance(batch["dataset"], str)
                else batch["dataset"]
            )
            sample_id = str(sample_id)
            dataset_name = str(dataset_name)

            original_size = batch["original_size"]
            if torch.is_tensor(original_size):
                # collate 后常见形状 [B, 2]
                original_height = int(original_size[index, 0].item())
                original_width = int(original_size[index, 1].item())
            elif isinstance(original_size, (list, tuple)) and len(original_size) == 2:
                if torch.is_tensor(original_size[0]):
                    original_height = int(original_size[0][index].item())
                    original_width = int(original_size[1][index].item())
                else:
                    original_height = int(original_size[0])
                    original_width = int(original_size[1])
            else:
                row = dataset.manifest.loc[
                    dataset.manifest["sample_id"].astype(str) == sample_id
                ].iloc[0]
                original_height = int(row["height"])
                original_width = int(row["width"])
            fg_ratio = batch["foreground_ratio"]
            if torch.is_tensor(fg_ratio):
                foreground_ratio = float(fg_ratio[index].item())
            elif isinstance(fg_ratio, (list, tuple)):
                foreground_ratio = float(fg_ratio[index])
            else:
                foreground_ratio = float(fg_ratio)

            per_image_rows.append(
                {
                    "sample_id": str(sample_id),
                    "dataset": str(dataset_name),
                    "split": split_name,
                    "dice": float(metrics["dice"][index].cpu().item()),
                    "iou": float(metrics["iou"][index].cpu().item()),
                    "precision": float(metrics["precision"][index].cpu().item()),
                    "recall": float(metrics["recall"][index].cpu().item()),
                    "foreground_ratio": foreground_ratio,
                    "pred_foreground_ratio": pred_ratio,
                    "original_height": original_height,
                    "original_width": original_width,
                    "empty_prediction": bool(pred_ratio <= 0.0),
                    "near_full_prediction": bool(pred_ratio >= 0.95),
                }
            )

    frame = pd.DataFrame(per_image_rows)
    if frame.empty:
        raise RuntimeError(f"no samples evaluated for split={split_name}")

    def _domain_block(subset: pd.DataFrame) -> dict:
        return {
            "dice": _summarize_scores(subset["dice"].tolist()),
            "iou": _summarize_scores(subset["iou"].tolist()),
            "precision": _summarize_scores(subset["precision"].tolist()),
            "recall": _summarize_scores(subset["recall"].tolist()),
            "count": int(len(subset)),
        }

    overall = _domain_block(frame)
    biohit = _domain_block(frame[frame["dataset"] == "biohit"])
    tongueset3 = _domain_block(frame[frame["dataset"] == "tongueset3"])

    overall_dice = float(overall["dice"]["mean"])
    biohit_dice = float(biohit["dice"]["mean"] or 0.0)
    tongueset3_dice = float(tongueset3["dice"]["mean"] or 0.0)
    domain_gap = abs(biohit_dice - tongueset3_dice)

    p33, p66 = _bucket_by_foreground(frame["foreground_ratio"].tolist())
    size_metrics = {}
    for name, mask in [
        ("small", frame["foreground_ratio"] < p33),
        ("medium", (frame["foreground_ratio"] >= p33) & (frame["foreground_ratio"] <= p66)),
        ("large", frame["foreground_ratio"] > p66),
    ]:
        subset = frame[mask]
        size_metrics[name] = {
            "count": int(len(subset)),
            "dice_mean": float(subset["dice"].mean()) if len(subset) else None,
            "iou_mean": float(subset["iou"].mean()) if len(subset) else None,
            "foreground_ratio_range": (
                [float(subset["foreground_ratio"].min()), float(subset["foreground_ratio"].max())]
                if len(subset)
                else None
            ),
        }

    worst_k = int(train_config.doc.get("evaluation", {}).get("worst_k", 20))
    worst = (
        frame.sort_values(["dice", "sample_id"], ascending=[True, True])
        .head(worst_k)
        .to_dict("records")
    )

    eval_cfg = train_config.doc.get("evaluation", {})
    gates = eval_cfg.get("gates", {})
    gate = decide_baseline_gate(
        overall_dice,
        biohit_dice,
        tongueset3_dice,
        overall_target=float(gates.get("overall_target", 0.95)),
        overall_minimum=float(gates.get("overall_minimum", 0.90)),
        domain_minimum=float(gates.get("domain_minimum", 0.90)),
    )

    result = {
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "config_hash": train_config.config_hash,
        "seed": train_config.seed,
        "threshold": threshold,
        "device": device_name,
        "param_counts": count_parameters(model),
        "loss_mean": float(total_loss / max(total_batches, 1)),
        "overall": overall,
        "biohit": biohit,
        "tongueset3": tongueset3,
        "domain_gap_dice": float(domain_gap),
        "foreground_size_metrics": size_metrics,
        "foreground_bucket_thresholds": {"p33": p33, "p66": p66},
        "empty_prediction_count": int(frame["empty_prediction"].sum()),
        "near_full_prediction_count": int(frame["near_full_prediction"].sum()),
        "pred_foreground_ratio": _summarize_scores(frame["pred_foreground_ratio"].tolist()),
        "worst_cases": worst,
        "baseline_gate": gate if split_name == "test" else None,
        "checkpoint_meta": {
            "epoch": checkpoint.get("epoch"),
            "best_val_dice": checkpoint.get("best_val_dice"),
            "best_epoch": checkpoint.get("best_epoch"),
            "config_hash": checkpoint.get("config_hash"),
            "seed": checkpoint.get("seed"),
        },
        "integrity_errors": integrity_errors,
        "test_access_count": 1 if split_name == "test" else 0,
        "discipline": {
            "test_used_for_training": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_threshold_tuning": False,
            "threshold_fixed": 0.5,
            "evaluated_after_best_freeze": bool(split_name == "test"),
        },
        "sample_count": int(len(frame)),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_dir / f"{split_name}_per_image_metrics.parquet", index=False)
        metrics_name = "test_metrics.json" if split_name == "test" else f"{split_name}_metrics.json"
        write_run_metadata(output_dir / metrics_name, result)
        # failure case metadata only
        (output_dir / "failure_cases").mkdir(exist_ok=True)
        write_run_metadata(output_dir / "failure_cases" / "worst_cases.json", {"cases": worst})

    return result
