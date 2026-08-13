"""Segmentation Trainer：smoke / tiny-overfit / 可扩展完整训练。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

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
        self.param_counts = count_parameters(self.model)
        self.threshold = float(train_config.mask_threshold)

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

    def save_last_and_best(self, val_dice: float, code_commit: str = "unknown"):
        common = dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epoch,
            global_step=self.global_step,
            best_val_dice=self.best_val_dice,
            config_dict=self.train_config.doc,
            config_hash=self.train_config.config_hash,
            seed=self.train_config.seed,
            history=self.history.to_list(),
            extra={
                "code_commit": code_commit,
                "device": self.device_name,
                "param_counts": self.param_counts,
            },
        )
        save_checkpoint(self.output_dir / "last.pt", **common)
        monitor_mode = str(self.train_config.checkpoint.get("mode", "max"))
        improved = val_dice > self.best_val_dice if monitor_mode == "max" else val_dice < self.best_val_dice
        if improved:
            self.best_val_dice = float(val_dice)
            common["best_val_dice"] = self.best_val_dice
            save_checkpoint(self.output_dir / "best.pt", **common)
            return True
        return False

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
    ) -> dict:
        package_root = Path(__file__).resolve().parents[4]
        code_commit = _git_commit(package_root)
        if start_epoch is not None:
            self.epoch = int(start_epoch)

        for _ in range(int(epochs)):
            self.epoch += 1
            started = time.time()
            train_stats = self.train_one_epoch(
                train_loader, max_batches=max_train_batches, compute_train_dice=False
            )
            # 验证前后参数比对：确保 validate 不改参数
            before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.model.named_parameters()
            }
            val_stats = self.validate(val_loader, max_batches=max_val_batches)
            for name, parameter in self.model.named_parameters():
                if not torch.equal(before[name], parameter.detach().cpu()):
                    raise RuntimeError(f"validation mutated parameter: {name}")

            self._maybe_step_scheduler(val_stats["val_dice"])
            improved = self.save_last_and_best(val_stats["val_dice"], code_commit=code_commit)

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
                "best_updated": bool(improved),
                "biohit_val_dice": float(
                    val_stats["per_domain"].get("biohit", {}).get("dice", 0.0)
                ),
                "tongueset3_val_dice": float(
                    val_stats["per_domain"].get("tongueset3", {}).get("dice", 0.0)
                ),
            }
            self.history.append(record)
            self.history.save(self.output_dir / "history.json")

        metadata = {
            "stage": "D3-B",
            "config_hash": self.train_config.config_hash,
            "seed": self.train_config.seed,
            "device": self.device_name,
            "amp": self.use_amp,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "code_commit": code_commit,
            "param_counts": self.param_counts,
            "best_val_dice": self.best_val_dice,
            "final_epoch": self.epoch,
            "global_step": self.global_step,
            "history": self.history.to_list(),
            "note": "Do not use color-jittered/normalized tensors for phenotype color analysis; map mask back to original RGB.",
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
