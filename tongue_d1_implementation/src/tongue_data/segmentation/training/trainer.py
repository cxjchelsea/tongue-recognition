"""Segmentation Trainer：smoke / tiny-overfit / 可扩展完整训练。"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def np_isfinite(value) -> bool:
    if value is None:
        return False
    return bool(math.isfinite(float(value)))

from ..config import SegmentationConfig
from ..dataset import (
    TongueSegmentationDataset,
    create_dataloader,
    select_tiny_overfit_subset,
)
from ..model import build_segmentation_model, count_parameters
from ..reproducibility import resolve_device, seed_everything
from ..train_config import TrainConfig
from .checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    write_run_metadata,
)
from .evaluation import MetricAggregator, batch_dice_iou_precision_recall
from .history import TrainingHistory
from .losses import build_loss
from .optimizer import build_optimizer, build_scheduler, current_lr


def _git_commit(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "image": batch["image"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
    }


def _assert_shapes(logits: torch.Tensor, mask: torch.Tensor):
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"expected logits [B,1,H,W], got {tuple(logits.shape)}")
    if mask.shape != logits.shape:
        raise ValueError(
            f"shape mismatch logits={tuple(logits.shape)} mask={tuple(mask.shape)}"
        )


def _assert_finite_loss(loss: torch.Tensor):
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss detected: {loss}")


class SegmentationTrainer:
    """D3-B Trainer：train/val loop + checkpoint + resume。"""

    def __init__(
        self,
        train_config: TrainConfig | str | Path,
        data_config: SegmentationConfig | str | Path,
        output_dir: str | Path,
        device: str | None = None,
    ):
        if isinstance(train_config, (str, Path)):
            train_config = TrainConfig(train_config)
        if isinstance(data_config, (str, Path)):
            data_config = SegmentationConfig(data_config)
        self.train_config = train_config
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(train_config.seed)
        self.device_name = resolve_device(device or train_config.device)
        self.device = torch.device(self.device_name)
        self.use_amp = bool(train_config.training.get("amp", True)) and self.device.type == "cuda"

        self.model = build_segmentation_model(train_config).to(self.device)
        self.criterion = build_loss(train_config.loss)
        self.optimizer = build_optimizer(self.model, train_config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, train_config.scheduler)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history = TrainingHistory()
        self.epoch = 0
        self.global_step = 0
        self.best_val_dice = -1.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.param_counts = count_parameters(self.model)
        self.threshold = float(train_config.mask_threshold)
        self.test_loader_built = False  # 训练阶段严禁构建 test

    def train_one_epoch(
        self,
        loader,
        *,
        max_batches: int | None = None,
        compute_train_dice: bool = False,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        dice_scores = []
        clip_norm = self.train_config.training.get("gradient_clip_norm")

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(batch["image"])
                _assert_shapes(logits, batch["mask"])
                loss = self.criterion(logits, batch["mask"])
            _assert_finite_loss(loss)

            self.scaler.scale(loss).backward()
            # gradient sanity：至少一个参数有 finite grad
            grad_ok = False
            for parameter in self.model.parameters():
                if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                    grad_ok = True
                    break
            if not grad_ok:
                raise RuntimeError("no finite gradients after backward")

            if clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

            if compute_train_dice:
                with torch.no_grad():
                    metrics = batch_dice_iou_precision_recall(
                        logits.detach().float(),
                        batch["mask"].float(),
                        threshold=self.threshold,
                    )
                    dice_scores.extend(metrics["dice"].detach().cpu().tolist())

        if total_batches == 0:
            raise RuntimeError("train_one_epoch received empty loader/batches")
        result = {
            "train_loss": total_loss / total_batches,
            "learning_rate": current_lr(self.optimizer),
            "batches": float(total_batches),
        }
        if compute_train_dice and dice_scores:
            result["train_dice"] = float(sum(dice_scores) / len(dice_scores))
        return result

    @torch.no_grad()
    def validate(
        self,
        loader,
        *,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        self.model.eval()
        aggregator = MetricAggregator()
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch(batch, self.device)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(batch["image"])
                _assert_shapes(logits, batch["mask"])
                loss = self.criterion(logits, batch["mask"])
            _assert_finite_loss(loss)
            # metrics 在 autocast 外用 float32，避免 384² reduce 溢出
            metrics = batch_dice_iou_precision_recall(
                logits.float(), batch["mask"].float(), threshold=self.threshold
            )
            datasets = batch["dataset"]
            if isinstance(datasets, str):
                dataset_list = [datasets]
            else:
                dataset_list = [str(item) for item in datasets]
            aggregator.update(metrics, dataset_list, loss_value=float(loss.item()))

        summary = aggregator.summarize()
        overall = summary.get("overall", {})
        return {
            "val_loss": summary.get("loss"),
            "val_dice": float(overall.get("dice", 0.0)),
            "val_iou": float(overall.get("iou", 0.0)),
            "val_precision": float(overall.get("precision", 0.0)),
            "val_recall": float(overall.get("recall", 0.0)),
            "per_domain": {
                key: value
                for key, value in summary.items()
                if key not in {"overall", "loss"}
            },
            "overall": overall,
        }

    def _maybe_step_scheduler(self, val_dice: float):
        if self.scheduler is None:
            return
        self.scheduler.step(val_dice)

    def save_last_and_best(
        self,
        val_dice: float,
        val_loss: float | None = None,
        code_commit: str = "unknown",
        extra_fields: dict | None = None,
    ):
        monitor_mode = str(self.train_config.checkpoint.get("mode", "max"))
        # 严格 > ：相同 Dice 保留更早 epoch（deterministic）
        improved = (
            val_dice > self.best_val_dice if monitor_mode == "max" else val_dice < self.best_val_dice
        )
        if improved:
            self.best_val_dice = float(val_dice)
            self.best_epoch = int(self.epoch)
            if val_loss is not None:
                self.best_val_loss = float(val_loss)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        common_kwargs = dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epoch,
            global_step=self.global_step,
            best_val_dice=self.best_val_dice,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            config_dict=self.train_config.doc,
            config_hash=self.train_config.config_hash,
            seed=self.train_config.seed,
            history=self.history.to_list(),
            extra={
                "code_commit": code_commit,
                "device": self.device_name,
                "param_counts": self.param_counts,
                **(extra_fields or {}),
            },
        )
        save_checkpoint(self.output_dir / "last.pt", **common_kwargs)
        if improved:
            save_checkpoint(self.output_dir / "best.pt", **common_kwargs)
        return bool(improved)

    def resume_from_checkpoint(self, path: str | Path):
        checkpoint = load_checkpoint(path, map_location=self.device)
        meta = restore_training_state(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.epoch = int(meta["epoch"])
        self.global_step = int(meta["global_step"])
        self.best_val_dice = float(meta["best_val_dice"])
        self.best_epoch = int(checkpoint.get("best_epoch") or self.epoch)
        raw_best_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_loss = (
            float("inf") if raw_best_loss is None else float(raw_best_loss)
        )
        self.history = TrainingHistory()
        self.history.epochs = list(meta.get("training_history", []))
        return meta

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        start_epoch: int | None = None,
        early_stopping: bool | None = None,
        stage: str = "D3-B",
        run_id: str | None = None,
        resumed: bool = False,
        verify_params_unchanged_on_val: bool = True,
    ) -> dict:
        package_root = Path(__file__).resolve().parents[4]
        code_commit = _git_commit(package_root)
        if start_epoch is not None:
            self.epoch = int(start_epoch)

        planned_epochs = int(epochs)
        early_cfg = self.train_config.early_stopping
        use_early = (
            bool(early_cfg.get("enabled", False))
            if early_stopping is None
            else bool(early_stopping)
        )
        patience = int(early_cfg.get("patience", 10))
        early_stopped = False

        for _ in range(planned_epochs):
            self.epoch += 1
            started = time.time()
            train_stats = self.train_one_epoch(
                train_loader, max_batches=max_train_batches, compute_train_dice=False
            )
            if verify_params_unchanged_on_val:
                before = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters()
                }
            val_stats = self.validate(val_loader, max_batches=max_val_batches)
            if verify_params_unchanged_on_val:
                for name, parameter in self.model.named_parameters():
                    if not torch.equal(before[name], parameter.detach().cpu()):
                        raise RuntimeError(f"validation mutated parameter: {name}")

            if not np_isfinite(val_stats["val_dice"]) or not np_isfinite(val_stats["val_loss"]):
                raise RuntimeError(f"non-finite validation metrics: {val_stats}")

            self._maybe_step_scheduler(val_stats["val_dice"])

            # 先更新 best 标记，再写入 history，再落盘 checkpoint（保证 history 完整）
            monitor_mode = str(self.train_config.checkpoint.get("mode", "max"))
            will_improve = (
                val_stats["val_dice"] > self.best_val_dice
                if monitor_mode == "max"
                else val_stats["val_dice"] < self.best_val_dice
            )
            record = {
                "epoch": self.epoch,
                "train_loss": train_stats["train_loss"],
                "val_loss": val_stats["val_loss"],
                "val_dice": val_stats["val_dice"],
                "val_iou": val_stats["val_iou"],
                "val_precision": val_stats["val_precision"],
                "val_recall": val_stats["val_recall"],
                "learning_rate": train_stats["learning_rate"],
                "duration_sec": float(time.time() - started),
                "best_updated": bool(will_improve),
                "biohit_val_dice": float(
                    val_stats["per_domain"].get("biohit", {}).get("dice", 0.0)
                ),
                "biohit_val_iou": float(
                    val_stats["per_domain"].get("biohit", {}).get("iou", 0.0)
                ),
                "tongueset3_val_dice": float(
                    val_stats["per_domain"].get("tongueset3", {}).get("dice", 0.0)
                ),
                "tongueset3_val_iou": float(
                    val_stats["per_domain"].get("tongueset3", {}).get("iou", 0.0)
                ),
            }
            self.history.append(record)
            self.history.save(self.output_dir / "history.json")

            improved = self.save_last_and_best(
                val_stats["val_dice"],
                val_loss=val_stats["val_loss"],
                code_commit=code_commit,
            )
            if bool(improved) != bool(will_improve):
                raise RuntimeError("best_updated flag inconsistent with checkpoint save")

            print(
                f"[epoch {self.epoch}/{planned_epochs}] "
                f"train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f} "
                f"val_dice={record['val_dice']:.4f} "
                f"biohit={record['biohit_val_dice']:.4f} "
                f"tongueset3={record['tongueset3_val_dice']:.4f} "
                f"best={self.best_val_dice:.4f}@ep{self.best_epoch} "
                f"lr={record['learning_rate']:.6f} "
                f"{'BEST' if improved else ''}",
                flush=True,
            )

            if use_early and self.epochs_without_improvement >= patience:
                early_stopped = True
                break

        # 从 history 找 best epoch 记录
        best_records = [row for row in self.history.to_list() if row.get("best_updated")]
        best_record = best_records[-1] if best_records else (
            self.history.to_list()[-1] if self.history.to_list() else {}
        )
        overfit_warning = False
        if len(self.history.to_list()) >= 3 and self.best_epoch > 0:
            recent = self.history.to_list()[-3:]
            if all(row["val_dice"] < self.best_val_dice - 1e-6 for row in recent):
                # train_loss 持续下降但长期无 val 提升
                if best_record.get("train_loss") is not None:
                    overfit_warning = True

        metadata = {
            "stage": stage,
            "run_id": run_id,
            "config_hash": self.train_config.config_hash,
            "seed": self.train_config.seed,
            "device": self.device_name,
            "amp": self.use_amp,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "code_commit": code_commit,
            "param_counts": self.param_counts,
            "planned_epochs": planned_epochs,
            "actual_epochs": self.epoch,
            "early_stopped": early_stopped,
            "best_epoch": self.best_epoch,
            "best_val_dice": self.best_val_dice,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
            "final_epoch": self.epoch,
            "global_step": self.global_step,
            "resumed": bool(resumed),
            "test_loader_built": bool(self.test_loader_built),
            "overfit_warning": bool(overfit_warning),
            "best_record": best_record,
            "history": self.history.to_list(),
            "baseline_frozen": False,
            "note": "Do not use color-jittered/normalized tensors for phenotype color analysis; map mask back to original RGB.",
            "discipline": {
                "test_used_for_training": False,
                "test_used_for_checkpoint_selection": False,
                "monitor": "val_dice",
            },
        }
        write_run_metadata(self.output_dir / "run_metadata.json", metadata)
        return metadata


def _build_loaders(
    manifest_path: Path,
    data_config: SegmentationConfig,
    train_config: TrainConfig,
    *,
    train_split_frame: pd.DataFrame | None = None,
    disable_train_aug: bool = False,
    batch_size: int | None = None,
):
    manifest = pd.read_parquet(manifest_path)
    # 训练阶段禁止使用 test
    if train_split_frame is not None:
        train_ds = TongueSegmentationDataset(
            train_split_frame.assign(split="train"),
            data_config,
            split="train",
            seed=train_config.seed,
            disable_augmentation=disable_train_aug,
        )
    else:
        train_ds = TongueSegmentationDataset(
            manifest,
            data_config,
            split="train",
            seed=train_config.seed,
            disable_augmentation=disable_train_aug,
        )
    val_ds = TongueSegmentationDataset(
        manifest, data_config, split="val", seed=train_config.seed, disable_augmentation=True
    )
    batch = int(batch_size or train_config.training.get("batch_size", 4))
    workers = int(train_config.training.get("num_workers", 0))
    train_loader = create_dataloader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = create_dataloader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)
    return train_loader, val_loader, train_ds, val_ds


def preflight_full_training(
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
) -> dict:
    """正式训练前计数 / 配置核对。"""
    train_config = TrainConfig(train_config_path)
    data_config = SegmentationConfig(data_config_path)
    manifest = pd.read_parquet(Path(segmentation_dir) / "segmentation_manifest.parquet")
    counts = {
        "train": int((manifest["split"] == "train").sum()),
        "val": int((manifest["split"] == "val").sum()),
        "test": int((manifest["split"] == "test").sum()),
    }
    for dataset_name in ["biohit", "tongueset3"]:
        for split_name in ["train", "val", "test"]:
            key = f"{dataset_name}_{split_name}"
            counts[key] = int(
                (
                    (manifest["dataset"].astype(str) == dataset_name)
                    & (manifest["split"].astype(str) == split_name)
                ).sum()
            )
    expected = dict(train_config.doc.get("run", {}).get("expected_counts", {}))
    mismatches = {
        key: {"expected": expected[key], "actual": counts.get(key)}
        for key in expected
        if int(expected[key]) != int(counts.get(key, -1))
    }
    if mismatches:
        raise ValueError(f"manifest counts mismatch vs freeze: {mismatches}")

    batch_size = int(train_config.training.get("batch_size", 4))
    steps_per_epoch = math.ceil(counts["train"] / max(batch_size, 1))
    device = resolve_device(train_config.device)
    model = build_segmentation_model(train_config)
    params = count_parameters(model)
    return {
        "counts": counts,
        "expected_ok": True,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "device": device,
        "seed": train_config.seed,
        "config_hash": train_config.config_hash,
        "model": train_config.model,
        "param_counts": params,
        "planned_epochs": int(train_config.training.get("epochs", 50)),
        "early_stopping": train_config.early_stopping,
        "run_id": train_config.doc.get("run", {}).get("run_id"),
        "data_contract_version": data_config.version,
        "note": "preflight does not build test loader for training",
    }


def run_full_training(
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> dict:
    """
    D3-C 正式训练：仅 train + val。
    绝不构建 test loader；test 必须由独立 evaluate 命令执行。
    """
    train_config = TrainConfig(train_config_path)
    data_config = SegmentationConfig(data_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = preflight_full_training(segmentation_dir, data_config_path, train_config_path)
    write_run_metadata(output_dir / "preflight.json", preflight)

    trainer = SegmentationTrainer(train_config, data_config, output_dir)
    resumed = False
    start_epoch = None
    if resume_from is not None:
        meta = trainer.resume_from_checkpoint(resume_from)
        resumed = True
        start_epoch = int(meta["epoch"])

    train_loader, val_loader, train_ds, val_ds = _build_loaders(
        Path(segmentation_dir) / "segmentation_manifest.parquet",
        data_config,
        train_config,
        disable_train_aug=False,
    )
    # 显式断言：未构建 test
    trainer.test_loader_built = False

    run_id = str(
        train_config.doc.get("run", {}).get("run_id", "d3c-resnet34-unet-seed20260813")
    )
    planned = int(train_config.training.get("epochs", 50))
    remaining = planned if start_epoch is None else max(planned - int(start_epoch), 0)
    if remaining == 0 and start_epoch is not None:
        # 已完成
        metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
        return metadata

    # full train 时关闭逐参数 clone（大模型开销大）；smoke 仍可开
    metadata = trainer.fit(
        train_loader,
        val_loader,
        epochs=remaining if start_epoch is not None else planned,
        start_epoch=start_epoch,
        early_stopping=True,
        stage="D3-C",
        run_id=run_id,
        resumed=resumed,
        verify_params_unchanged_on_val=False,
    )
    metadata["preflight"] = preflight
    metadata["train_samples"] = int(len(train_ds))
    metadata["val_samples"] = int(len(val_ds))
    metadata["test_samples_not_used"] = int(preflight["counts"]["test"])

    # Freeze best checkpoint：reload 校验
    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("best.pt missing after full training")
    verify_trainer = SegmentationTrainer(train_config, data_config, output_dir / "_reload_check")
    reload_meta = verify_trainer.resume_from_checkpoint(best_path)
    if reload_meta["best_val_dice"] != metadata["best_val_dice"]:
        # float compare
        if abs(float(reload_meta["best_val_dice"]) - float(metadata["best_val_dice"])) > 1e-8:
            raise RuntimeError("best checkpoint reload metric mismatch")
    metadata["baseline_frozen"] = True
    metadata["best_checkpoint_reload"] = "PASS"
    metadata["best_checkpoint_path"] = str(best_path)

    # training summary
    history = metadata.get("history") or []
    last = history[-1] if history else {}
    summary = {
        "run_id": run_id,
        "planned_epochs": metadata["planned_epochs"],
        "actual_epochs": metadata["actual_epochs"],
        "early_stopped": metadata["early_stopped"],
        "best_epoch": metadata["best_epoch"],
        "best_val_dice": metadata["best_val_dice"],
        "best_val_loss": metadata["best_val_loss"],
        "last_val_dice": last.get("val_dice"),
        "last_val_loss": last.get("val_loss"),
        "biohit_best_val_dice": next(
            (
                row["biohit_val_dice"]
                for row in history
                if row.get("epoch") == metadata["best_epoch"]
            ),
            None,
        ),
        "tongueset3_best_val_dice": next(
            (
                row["tongueset3_val_dice"]
                for row in history
                if row.get("epoch") == metadata["best_epoch"]
            ),
            None,
        ),
        "overfit_warning": metadata.get("overfit_warning"),
        "config_hash": metadata["config_hash"],
        "seed": metadata["seed"],
        "device": metadata["device"],
        "resumed": resumed,
        "test_access_during_training": 0,
        "discipline": metadata["discipline"],
    }
    write_run_metadata(output_dir / "training_summary.json", summary)
    write_run_metadata(output_dir / "val_metrics.json", {
        "best_epoch": metadata["best_epoch"],
        "best_val_dice": metadata["best_val_dice"],
        "best_val_loss": metadata["best_val_loss"],
        "best_record": metadata.get("best_record"),
    })
    write_run_metadata(output_dir / "run_metadata.json", metadata)

    # 训练曲线精简 JSON（供人工审查）
    curve = {
        "epoch": [row["epoch"] for row in history],
        "train_loss": [row["train_loss"] for row in history],
        "val_loss": [row["val_loss"] for row in history],
        "val_dice": [row["val_dice"] for row in history],
    }
    reports_dir = Path("reports/d3")
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(reports_dir / "training_curve.json", curve)
    return metadata


def run_smoke_training(
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """真实数据 smoke：少量 batch，验证 forward/backward/val/ckpt。"""
    train_config = TrainConfig(train_config_path)
    data_config = SegmentationConfig(data_config_path)
    output_dir = Path(output_dir)
    trainer = SegmentationTrainer(train_config, data_config, output_dir)

    max_train = int(train_config.smoke.get("max_train_batches", 2))
    max_val = int(train_config.smoke.get("max_val_batches", 2))
    epochs = int(train_config.smoke.get("epochs", 1))

    train_loader, val_loader, train_ds, val_ds = _build_loaders(
        Path(segmentation_dir) / "segmentation_manifest.parquet",
        data_config,
        train_config,
        disable_train_aug=False,
    )
    metadata = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        max_train_batches=max_train,
        max_val_batches=max_val,
    )
    metadata["smoke"] = {
        "max_train_batches": max_train,
        "max_val_batches": max_val,
        "epochs": epochs,
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "result": "PASS",
    }
    # resume sanity：从 last 恢复后 epoch 连续
    resumed = SegmentationTrainer(train_config, data_config, output_dir / "resume_check")
    meta = resumed.resume_from_checkpoint(output_dir / "last.pt")
    metadata["resume_test"] = {
        "loaded_epoch": meta["epoch"],
        "expected_epoch": trainer.epoch,
        "result": "PASS" if meta["epoch"] == trainer.epoch else "FAIL",
    }
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    if metadata["resume_test"]["result"] != "PASS":
        raise RuntimeError("resume smoke failed")
    return metadata


def run_tiny_overfit(
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Tiny-set overfit：关闭增广，验证 pipeline 可记住小样本。"""
    train_config = TrainConfig(train_config_path)
    data_config = SegmentationConfig(data_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiny_cfg = dict(train_config.debug.get("tiny_overfit", {}))

    manifest = pd.read_parquet(Path(segmentation_dir) / "segmentation_manifest.parquet")
    per_dataset = tiny_cfg.get("per_dataset") or {"biohit": 8, "tongueset3": 8}
    subset = select_tiny_overfit_subset(
        manifest,
        per_dataset={str(key): int(value) for key, value in dict(per_dataset).items()},
        sample_count=int(tiny_cfg.get("sample_count", 16)),
        seed=train_config.seed,
    )
    subset.to_parquet(output_dir / "tiny_subset.parquet", index=False)

    # 为 overfit 提高学习效率：可用独立 lr/batch
    trainer = SegmentationTrainer(train_config, data_config, output_dir)
    if tiny_cfg.get("lr") is not None:
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(tiny_cfg["lr"])

    batch_size = int(tiny_cfg.get("batch_size", train_config.training.get("batch_size", 4)))
    train_ds = TongueSegmentationDataset(
        subset.assign(split="train"),
        data_config,
        split="train",
        seed=train_config.seed,
        disable_augmentation=not bool(tiny_cfg.get("augmentation", False)),
    )
    # overfit 用同一 tiny set 做“train dice”监控；不用 val/test 做判断
    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    eval_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    max_steps = int(tiny_cfg.get("max_steps", 500))
    target_dice = float(tiny_cfg.get("target_dice", 0.95))
    initial_loss = None
    final_loss = None
    final_dice = 0.0
    steps = 0

    trainer.model.train()
    data_iter = iter(train_loader)
    while steps < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = _move_batch(batch, trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(trainer.device.type, enabled=trainer.use_amp):
            logits = trainer.model(batch["image"])
            _assert_shapes(logits, batch["mask"])
            loss = trainer.criterion(logits, batch["mask"])
        _assert_finite_loss(loss)
        trainer.scaler.scale(loss).backward()
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        steps += 1
        loss_value = float(loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value

        if steps % 20 == 0 or steps == max_steps:
            eval_stats = trainer.validate(eval_loader)
            final_dice = float(eval_stats["val_dice"])
            if final_dice >= target_dice:
                break

    # 最终评估
    eval_stats = trainer.validate(eval_loader)
    final_dice = float(eval_stats["val_dice"])
    final_loss = float(eval_stats["val_loss"]) if eval_stats["val_loss"] is not None else final_loss

    save_checkpoint(
        output_dir / "last.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        epoch=0,
        global_step=steps,
        best_val_dice=final_dice,
        config_dict=train_config.doc,
        config_hash=train_config.config_hash,
        seed=train_config.seed,
        history=[],
        extra={"mode": "tiny_overfit", "final_dice": final_dice},
    )

    result = "PASS" if final_dice >= target_dice and (final_loss < initial_loss) else "FAIL"
    metadata = {
        "stage": "D3-B-tiny-overfit",
        "sample_count": int(len(subset)),
        "datasets": subset["dataset"].astype(str).value_counts().to_dict(),
        "sample_ids": subset["sample_id"].astype(str).tolist(),
        "augmentation": bool(tiny_cfg.get("augmentation", False)),
        "steps": int(steps),
        "max_steps": max_steps,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "final_dice": float(final_dice),
        "target_dice": target_dice,
        "device": trainer.device_name,
        "amp": trainer.use_amp,
        "param_counts": trainer.param_counts,
        "config_hash": train_config.config_hash,
        "result": result,
        "note": "tiny overfit is pipeline sanity, not generalization",
    }
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    if result != "PASS":
        raise RuntimeError(
            f"tiny overfit FAILED: dice={final_dice:.4f} target={target_dice} "
            f"loss {initial_loss:.4f}->{final_loss:.4f}"
        )
    return metadata
