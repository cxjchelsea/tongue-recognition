"""D4-C.1-B：domain-robust stain v2 训练（不覆盖 v1）。"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .calibrate import calibrate_dual_thresholds, freeze_thresholds
from .config import StainDataConfig, StainTrainConfig
from .consistency import (
    consistency_warmup_factor,
    decompose_total_loss,
    probability_consistency_loss,
    source_supervised_two_view_loss,
)
from .dataset import StainRoiDataset, select_overfit_subset
from .domain_loader import (
    ExternalConsistencyDataset,
    create_external_loader,
    create_source_loader,
)
from .metrics import summarize_ranking_metrics, three_state_metrics
from .model import build_stain_model, count_parameters
from .style_augment import estimate_style_ranges_from_train, load_style_contract
from .train import (
    _git_commit,
    collect_predictions,
    resolve_device,
    set_seed,
)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def run_tiny_source_overfit(
    *,
    stain_manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """关闭 style，验证双 view trainer 基础分类能力。"""
    train_cfg = StainTrainConfig(train_config)
    set_seed(train_cfg.seed)
    device_t = resolve_device(device)
    overfit_cfg = train_cfg.overfit
    subset = select_overfit_subset(
        pd.read_parquet(stain_manifest),
        positives=int(overfit_cfg.get("positives", 8)),
        negatives=int(overfit_cfg.get("negatives", 8)),
    )
    loader = create_source_loader(
        stain_manifest,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=min(16, len(subset)),
        disable_style=True,
        subset_ids=subset["sample_id"].tolist(),
        shuffle=True,
    )
    model = build_stain_model(train_cfg.model).to(device_t)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_cfg.optimizer.get("weight_decay", 1e-4)),
    )
    epochs = int(overfit_cfg.get("epochs", 40))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        losses = []
        for batch in loader:
            weak = batch["image_weak"].to(device_t)
            style = batch["image_style"].to(device_t)
            labels = batch["label"].to(device_t)
            optimizer.zero_grad(set_to_none=True)
            logit_weak = model(weak)
            logit_style = model(style)
            loss = source_supervised_two_view_loss(logit_weak, logit_style, labels)
            loss.backward()
            optimizer.step()
            probs = torch.sigmoid(logit_weak.reshape(-1))
            pred = (probs >= 0.5).float()
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            losses.append(float(loss.item()))
        acc = correct / max(total, 1)
        history.append({"epoch": epoch, "accuracy": acc, "loss": float(np.mean(losses))})
        if acc >= float(overfit_cfg.get("accuracy_gate", 0.95)):
            break
    result = {
        "passed": history[-1]["accuracy"] >= float(overfit_cfg.get("accuracy_gate", 0.95)),
        "final_accuracy": history[-1]["accuracy"],
        "epochs_run": len(history),
        "history": history,
        "style_disabled": True,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiny_overfit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_external_consistency_smoke(
    *,
    roi_index: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    train_cfg = StainTrainConfig(train_config)
    set_seed(train_cfg.seed)
    device_t = resolve_device(device)
    smoke = train_cfg.doc.get("consistency_smoke", {})
    frame = pd.read_parquet(roi_index)
    bio = (
        frame[(frame.dataset == "biohit") & (frame.split == "train")]
        .sort_values("sample_id")
        .head(int(smoke.get("biohit", 4)))
    )
    ts3 = (
        frame[(frame.dataset == "tongueset3") & (frame.split == "train")]
        .sort_values("sample_id")
        .head(int(smoke.get("tongueset3", 4)))
    )
    ids = bio["sample_id"].tolist() + ts3["sample_id"].tolist()
    loader = create_external_loader(
        roi_index,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=8,
        subset_ids=ids,
    )
    model = build_stain_model(train_cfg.model).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    batch = next(iter(loader))
    assert batch["has_gold_label"] is False
    assert "label" not in batch
    weak = batch["image_weak"].to(device_t)
    style = batch["image_style"].to(device_t)
    optimizer.zero_grad(set_to_none=True)
    logit_w = model(weak)
    logit_s = model(style)
    loss = probability_consistency_loss(logit_s, logit_w, stop_gradient_teacher=True)
    assert torch.isfinite(loss)
    loss.backward()
    grad_ok = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    result = {
        "passed": bool(torch.isfinite(loss).item() and grad_ok),
        "loss": float(loss.item()),
        "n_samples": len(ids),
        "pseudo_labels": False,
        "entropy_minimization": False,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "external_consistency_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _source_eval_loader(
    stain_manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    split: str,
) -> DataLoader:
    dataset = StainRoiDataset(
        stain_manifest,
        data_config,
        train_config,
        split=split,
        disable_augmentation=True,
        seed=StainTrainConfig(train_config).seed,
    )
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


@torch.inference_mode()
def evaluate_source_split(
    model: nn.Module,
    *,
    stain_manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    split: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = _source_eval_loader(stain_manifest, data_config, train_config, split)
    pred = collect_predictions(model, loader, device)
    metrics = summarize_ranking_metrics(pred["label"].to_numpy(), pred["p_stain"].to_numpy())
    return pred, metrics


@torch.inference_mode()
def evaluate_external_roi(
    model: nn.Module,
    *,
    roi_index: str | Path,
    data_config: str | Path,
    split: str,
    datasets: tuple[str, ...] = ("biohit", "tongueset3"),
    device: torch.device,
    max_per_dataset: int | None = None,
) -> pd.DataFrame:
    from .transforms import preprocess_masked_roi
    from PIL import Image

    data_cfg = StainDataConfig(data_config)
    frame = pd.read_parquet(roi_index)
    frame = frame[
        (frame["split"] == split)
        & (frame["dataset"].isin(list(datasets)))
        & frame["roi_rgb_path"].notna()
    ].sort_values("sample_id")
    if max_per_dataset is not None:
        frame = (
            frame.groupby("dataset", group_keys=False)
            .head(int(max_per_dataset))
            .reset_index(drop=True)
        )
    model.eval()
    rows = []
    for _index, row in frame.iterrows():
        rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
        mask = (np.asarray(Image.open(row["roi_mask_path"])) > 0).astype(np.uint8)
        tensor = preprocess_masked_roi(rgb, mask, data_cfg, split="val")
        batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
        logit = model(batch).reshape(-1)[0]
        prob = float(torch.sigmoid(logit).item())
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "split": split,
                "logit": float(logit.item()),
                "p_stain": prob,
                "label": None,
            }
        )
    return pd.DataFrame(rows)


def _external_summary(pred: pd.DataFrame, t_clear: float, t_retake: float) -> dict[str, Any]:
    out = {}
    for dataset_name, subset in pred.groupby("dataset"):
        probs = subset["p_stain"].to_numpy()
        logits = subset["logit"].to_numpy()
        findings = []
        for score in probs:
            if score <= t_clear:
                findings.append("clear")
            elif score >= t_retake:
                findings.append("stain")
            else:
                findings.append("uncertain")
        out[str(dataset_name)] = {
            "n": int(len(subset)),
            "median_p": float(np.median(probs)),
            "mean_p": float(np.mean(probs)),
            "p05_p": float(np.percentile(probs, 5)),
            "p95_p": float(np.percentile(probs, 95)),
            "median_logit": float(np.median(logits)),
            "mean_logit": float(np.mean(logits)),
            "highscore_rate": float((probs >= t_retake).mean()),
            "uncertain_rate": float(
                ((probs > t_clear) & (probs < t_retake)).mean()
            ),
            "band_counts": {
                "clear": findings.count("clear"),
                "uncertain": findings.count("uncertain"),
                "stain": findings.count("stain"),
            },
        }
    if "biohit" in out and "tongueset3" in out:
        out["domain_gap_median_logit"] = (
            out["tongueset3"]["median_logit"] - out["biohit"]["median_logit"]
        )
        out["domain_gap_median_p"] = (
            out["tongueset3"]["median_p"] - out["biohit"]["median_p"]
        )
    return out


def train_stain_v2(
    *,
    stain_manifest: str | Path = "data/stain/v1/stain_manifest.parquet",
    roi_index: str | Path = "reports/d4c1/roi_cache/index.parquet",
    data_config: str | Path = "configs/stain_detection_v2.yaml",
    train_config: str | Path = "configs/stain_train_v2.yaml",
    style_contract_path: str | Path = "reports/d4c1b/style_augmentation_contract.json",
    output_dir: str | Path = "runs/input_guard/d4c1b/stain_v2",
    device: str = "auto",
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """正式 v2 训练：best 仅由 source val AUROC 选择。"""
    train_cfg = StainTrainConfig(train_config)
    data_cfg = StainDataConfig(data_config)
    set_seed(train_cfg.seed)
    device_t = resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(style_contract_path).exists():
        estimate_style_ranges_from_train(
            stain_manifest=stain_manifest,
            external_roi_index=roi_index,
            output_path=style_contract_path,
        )
    style_contract = load_style_contract(style_contract_path)

    source_loader = create_source_loader(
        stain_manifest,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=int(train_cfg.training.get("batch_size", 32)),
        disable_style=False,
        shuffle=True,
    )
    external_loader = create_external_loader(
        roi_index,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=int(train_cfg.training.get("external_batch_size", 32)),
        biohit_fraction=float(train_cfg.doc.get("external", {}).get("biohit_fraction", 0.5)),
    )
    val_loader = _source_eval_loader(stain_manifest, data_config, train_config, "val")

    # 禁止从 v1 fine-tune
    if train_cfg.model.get("init_from_v1"):
        raise RuntimeError("v2 must not fine-tune from D4-C v1 checkpoint")
    model = build_stain_model(train_cfg.model).to(device_t)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_cfg.optimizer.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_cfg.scheduler.get("factor", 0.5)),
        patience=int(train_cfg.scheduler.get("patience", 3)),
        min_lr=float(train_cfg.scheduler.get("min_lr", 1e-6)),
    )
    amp = bool(train_cfg.training.get("amp", True)) and device_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_cfg = train_cfg.loss
    epochs = int(max_epochs or train_cfg.training.get("epochs", 35))
    patience = int(train_cfg.early_stopping.get("patience", 8))
    warmup_epochs = int(loss_cfg.get("consistency_warmup_epochs", 5))
    stop_grad = bool(train_cfg.doc.get("external", {}).get("stop_gradient_weak_target", True))

    history: list[dict[str, Any]] = []
    best_auroc = -1.0
    best_epoch = -1
    bad_epochs = 0
    external_iter = _cycle(external_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        warmup = consistency_warmup_factor(epoch, warmup_epochs)
        meters = {
            "supervised": [],
            "source_consistency": [],
            "external_consistency": [],
            "total": [],
        }
        for source_batch in source_loader:
            external_batch = next(external_iter)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                sw = source_batch["image_weak"].to(device_t)
                ss = source_batch["image_style"].to(device_t)
                labels = source_batch["label"].to(device_t)
                ew = external_batch["image_weak"].to(device_t)
                es = external_batch["image_style"].to(device_t)

                logit_sw = model(sw)
                logit_ss = model(ss)
                logit_ew = model(ew)
                logit_es = model(es)

                supervised = source_supervised_two_view_loss(logit_sw, logit_ss, labels)
                source_cons = probability_consistency_loss(
                    logit_ss, logit_sw, stop_gradient_teacher=stop_grad
                )
                external_cons = probability_consistency_loss(
                    logit_es, logit_ew, stop_gradient_teacher=stop_grad
                )
                parts = decompose_total_loss(
                    supervised=supervised,
                    source_consistency=source_cons,
                    external_consistency=external_cons,
                    supervised_weight=float(loss_cfg.get("supervised_weight", 1.0)),
                    source_consistency_weight=float(
                        loss_cfg.get("source_consistency_weight", 0.5)
                    ),
                    external_consistency_weight=float(
                        loss_cfg.get("external_consistency_weight", 0.5)
                    ),
                    warmup=warmup,
                )
                loss = parts["total"]
            scaler.scale(loss).backward()
            if train_cfg.training.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_cfg.training["gradient_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()
            meters["supervised"].append(float(parts["supervised"].item()))
            meters["source_consistency"].append(float(parts["source_consistency"].item()))
            meters["external_consistency"].append(
                float(parts["external_consistency"].item())
            )
            meters["total"].append(float(loss.item()))

        # source val
        val_pred = collect_predictions(model, val_loader, device_t)
        val_metrics = summarize_ranking_metrics(
            val_pred["label"].to_numpy(), val_pred["p_stain"].to_numpy()
        )
        # external val monitor（不参与选模）
        # 训练期仅抽样监控（不参与选模）；完整 external val 在 audit 阶段评估
        ext_val = evaluate_external_roi(
            model,
            roi_index=roi_index,
            data_config=data_config,
            split="val",
            device=device_t,
            max_per_dataset=40,
        )
        ext_mon = _external_summary(ext_val, 0.95, 0.96)

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_source_loss": float(np.mean(meters["supervised"])),
            "source_supervised_loss": float(np.mean(meters["supervised"])),
            "source_consistency_loss": float(np.mean(meters["source_consistency"])),
            "external_consistency_loss": float(np.mean(meters["external_consistency"])),
            "train_total_loss": float(np.mean(meters["total"])),
            "val_auroc": float(val_metrics["auroc"]),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_accuracy@0.5": float(val_metrics.get("at_0.5_accuracy", 0.0)),
            "lr": lr,
            "warmup": warmup,
            "external_val_monitor": ext_mon,
            "selection_metric": "source_val_auroc",
        }
        history.append(row)
        scheduler.step(row["val_auroc"])
        print(
            f"[v2] epoch={epoch} val_auroc={row['val_auroc']:.4f} "
            f"ext_ts3_p50={ext_mon.get('tongueset3', {}).get('median_p')} "
            f"warmup={warmup:.2f}",
            flush=True,
        )

        improved = row["val_auroc"] > best_auroc + 1e-12
        if improved:
            best_auroc = row["val_auroc"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auroc": best_auroc,
                    "train_config_hash": train_cfg.config_hash,
                    "data_config_hash": data_cfg.config_hash,
                    "architecture": train_cfg.model.get("architecture"),
                    "input_size": data_cfg.input_size,
                    "seed": train_cfg.seed,
                    "stage": "D4-C.1-B",
                    "stain_contract_version": data_cfg.version,
                    "init_from_v1": False,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output_dir / "last.pt")
        if train_cfg.early_stopping.get("enabled", True) and bad_epochs >= patience:
            print(f"[v2] early stop at epoch={epoch}", flush=True)
            break

    # reload best
    best_blob = torch.load(output_dir / "best.pt", map_location=device_t, weights_only=False)
    model.load_state_dict(best_blob["model_state_dict"])
    val_pred, val_metrics = evaluate_source_split(
        model,
        stain_manifest=stain_manifest,
        data_config=data_config,
        train_config=train_config,
        split="val",
        device=device_t,
    )
    val_pred.to_parquet(output_dir / "val_predictions.parquet", index=False)

    # source-val-only calibration
    cal = calibrate_dual_thresholds(
        val_pred["label"].to_numpy(),
        val_pred["p_stain"].to_numpy(),
        target_confident_precision=float(
            train_cfg.calibration.get("target_confident_precision", 0.90)
        ),
    )
    thr_path = output_dir / "thresholds.json"
    freeze_thresholds(cal, thr_path)
    # 附加 provenance，不覆盖 v1
    thr_doc = json.loads(thr_path.read_text(encoding="utf-8"))
    thr_doc.update(
        {
            "stage": "D4-C.1-B",
            "calibrated_on": "stained_val_only",
            "v1_thresholds_preserved": {"t_clear": 0.95, "t_retake": 0.96},
        }
    )
    thr_path.write_text(json.dumps(thr_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "stage": "D4-C.1-B",
        "stain_contract_version": data_cfg.version,
        "train_config_hash": train_cfg.config_hash,
        "data_config_hash": data_cfg.config_hash,
        "seed": train_cfg.seed,
        "planned_epochs": int(train_cfg.training.get("epochs", 35)),
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "best_source_val_auroc": best_auroc,
        "best_source_val_pr_auc": float(val_metrics["pr_auc"]),
        "t_clear_v2": cal["t_clear"],
        "t_retake_v2": cal["t_retake"],
        "params": count_parameters(model),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "external_pseudo_labels": False,
        "init_from_v1": False,
        "selection_rule": "source_val_auroc_only",
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "metadata": metadata,
        "history": history,
        "val_metrics": val_metrics,
        "thresholds": {"t_clear": cal["t_clear"], "t_retake": cal["t_retake"]},
    }
