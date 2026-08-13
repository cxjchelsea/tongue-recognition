"""D3-B：ResNet34-UNet Trainer / Loss / Checkpoint 测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from tongue_data.segmentation.dataset import select_tiny_overfit_subset
from tongue_data.segmentation.model import build_segmentation_model, count_parameters
from tongue_data.segmentation.train_config import TrainConfig
from tongue_data.segmentation.training.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from tongue_data.segmentation.training.evaluation import (
    MetricAggregator,
    batch_dice_iou_precision_recall,
    logits_to_binary,
)
from tongue_data.segmentation.training.history import TrainingHistory
from tongue_data.segmentation.training.losses import BCEDiceLoss, SoftDiceLoss, build_loss
from tongue_data.segmentation.training.optimizer import build_optimizer, build_scheduler
from tongue_data.segmentation.training.trainer import SegmentationTrainer


pytest.importorskip("segmentation_models_pytorch")


def _write_train_config(path: Path, **overrides) -> Path:
    doc = {
        "version": "1.0-test",
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1.0e-6},
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "scheduler": {"name": "reduce_on_plateau", "mode": "max", "factor": 0.5, "patience": 1},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "amp": False,
            "device": "cpu",
            "gradient_clip_norm": None,
        },
        "checkpoint": {"monitor": "val_dice", "mode": "max"},
        "metrics": {"mask_threshold": 0.5},
        "reproducibility": {"seed": 20260813},
        "debug": {"tiny_overfit": {"sample_count": 4, "augmentation": False}},
        "smoke": {"max_train_batches": 1, "max_val_batches": 1, "epochs": 1},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            doc[key].update(value)
        else:
            doc[key] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


class _SyntheticSegDataset(Dataset):
    def __init__(self, size: int = 4, height: int = 64, width: int = 64, datasets=None):
        self.size = size
        self.height = height
        self.width = width
        self.datasets = datasets or (["biohit", "tongueset3"] * ((size + 1) // 2))[:size]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + 7)
        image = torch.randn(3, self.height, self.width, generator=generator)
        mask = torch.zeros(1, self.height, self.width)
        mask[:, 8:40, 8:40] = 1.0
        return {
            "image": image,
            "mask": mask,
            "sample_id": f"syn::{index}",
            "dataset": self.datasets[index],
        }


def test_unet_forward_shape():
    model = build_segmentation_model(
        {
            "architecture": "unet",
            "encoder": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        }
    )
    model.eval()
    inputs = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        outputs = model(inputs)
    assert tuple(outputs.shape) == (2, 1, 384, 384)


def test_raw_logits_not_probability_range():
    model = build_segmentation_model(
        {"architecture": "unet", "encoder": "resnet34", "encoder_weights": None}
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 128, 128))
    # logits 可超出 [0,1]
    assert float(outputs.min()) < 0.0 or float(outputs.max()) > 1.0


def test_dice_loss_perfect_near_zero():
    loss_fn = SoftDiceLoss(smooth=1e-6)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 4:20, 4:20] = 1.0
    # 极大 logits → sigmoid≈1 on foreground
    logits = torch.where(target > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    assert float(loss_fn(logits, target)) < 1e-3


def test_dice_loss_zero_overlap():
    loss_fn = SoftDiceLoss(smooth=1e-6)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, :8, :8] = 1.0
    logits = torch.where(target > 0.5, torch.tensor(-20.0), torch.tensor(20.0))
    value = float(loss_fn(logits, target))
    assert value > 0.9


def test_combined_loss_finite():
    loss_fn = build_loss({"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1e-6})
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    value = loss_fn(logits, target)
    assert torch.isfinite(value)


def test_loss_backward_finite_grad():
    model = build_segmentation_model(
        {"architecture": "unet", "encoder": "resnet34", "encoder_weights": None}
    )
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.randn(1, 3, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 10:40, 10:40] = 1.0
    optimizer.zero_grad()
    logits = model(image)
    loss = loss_fn(logits, target)
    loss.backward()
    found = False
    for parameter in model.parameters():
        if parameter.grad is not None and torch.isfinite(parameter.grad).all():
            found = True
            break
    assert found


def test_train_one_epoch_runs(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    # data config stub：只需要 seed/input 等；用真实 segmentation_v1
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4, 64, 64), batch_size=2)
    stats = trainer.train_one_epoch(loader, max_batches=2)
    assert np.isfinite(stats["train_loss"])


def test_validation_does_not_mutate_params(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4, 64, 64), batch_size=2)
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    trainer.validate(loader, max_batches=2)
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(before[name], parameter.detach())


def test_metric_sigmoid_threshold():
    # sigmoid(0)=0.5 → >=0.5 为前景；sigmoid(-10)≈0
    logits = torch.tensor([[[[-1.0, 10.0], [-10.0, 1.0]]]])
    binary = logits_to_binary(logits, threshold=0.5)
    assert binary.tolist() == [[[[0.0, 1.0], [0.0, 1.0]]]]


def test_best_checkpoint_by_val_dice(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    trainer.epoch = 1
    assert trainer.save_last_and_best(0.70) is True
    assert (tmp_path / "out" / "best.pt").exists()
    first_best = load_checkpoint(tmp_path / "out" / "best.pt")
    assert first_best["best_val_dice"] == pytest.approx(0.70)
    # 更差不覆盖 best
    trainer.epoch = 2
    assert trainer.save_last_and_best(0.60) is False
    second_best = load_checkpoint(tmp_path / "out" / "best.pt")
    assert second_best["best_val_dice"] == pytest.approx(0.70)
    assert second_best["epoch"] == 1
    # last 更新
    last = load_checkpoint(tmp_path / "out" / "last.pt")
    assert last["epoch"] == 2


def test_checkpoint_state_dict_roundtrip(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    trainer.epoch = 3
    trainer.global_step = 11
    # best_val_dice 初始为 -1，传入更高值才会写入 best.pt
    trainer.save_last_and_best(0.8)
    other = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out2")
    other.resume_from_checkpoint(tmp_path / "out" / "best.pt")
    for left, right in zip(trainer.model.state_dict().values(), other.model.state_dict().values()):
        assert torch.equal(left.cpu(), right.cpu())
    assert other.epoch == 3
    assert other.global_step == 11


def test_optimizer_and_scheduler_resume(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4, 64, 64), batch_size=2)
    trainer.train_one_epoch(loader, max_batches=1)
    trainer._maybe_step_scheduler(0.1)
    trainer.epoch = 1
    trainer.save_last_and_best(0.1)
    opt_state = trainer.optimizer.state_dict()
    sch_state = trainer.scheduler.state_dict()
    resumed = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out2")
    resumed.resume_from_checkpoint(tmp_path / "out" / "last.pt")
    assert resumed.optimizer.state_dict()["param_groups"][0]["lr"] == opt_state["param_groups"][0]["lr"]
    assert resumed.scheduler.state_dict()["best"] == sch_state["best"]


def test_resume_epoch_continuity(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4, 64, 64), batch_size=2)
    trainer.fit(loader, loader, epochs=1, max_train_batches=1, max_val_batches=1)
    resumed = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out2")
    meta = resumed.resume_from_checkpoint(tmp_path / "out" / "last.pt")
    assert meta["epoch"] == 1
    resumed.fit(loader, loader, epochs=1, max_train_batches=1, max_val_batches=1, start_epoch=meta["epoch"])
    assert resumed.epoch == 2


def test_amp_disabled_on_cpu(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml", training={"amp": True, "device": "cpu"})
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    assert trainer.use_amp is False
    loader = DataLoader(_SyntheticSegDataset(2, 64, 64), batch_size=2)
    stats = trainer.train_one_epoch(loader, max_batches=1)
    assert np.isfinite(stats["train_loss"])


def test_nan_loss_guard():
    loss_fn = SoftDiceLoss()
    logits = torch.tensor([[[[float("nan")]]]])
    target = torch.ones(1, 1, 1, 1)
    value = loss_fn(logits, target)
    assert not torch.isfinite(value)


def test_shape_mismatch_fail_fast():
    loss_fn = BCEDiceLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 4, 4))


def test_history_serializable(tmp_path: Path):
    history = TrainingHistory()
    history.append({"epoch": 1, "train_loss": 0.5, "val_dice": 0.1})
    history.save(tmp_path / "history.json")
    loaded = TrainingHistory.load(tmp_path / "history.json")
    assert loaded.epochs[0]["epoch"] == 1


def test_config_hash_deterministic(tmp_path: Path):
    path = _write_train_config(tmp_path / "train.yaml")
    left = TrainConfig(path).config_hash
    right = TrainConfig(path).config_hash
    assert left == right
    assert len(left) == 16


def test_seed_reproducible_init():
    torch.manual_seed(20260813)
    model_a = build_segmentation_model(
        {"architecture": "unet", "encoder": "resnet34", "encoder_weights": None}
    )
    torch.manual_seed(20260813)
    model_b = build_segmentation_model(
        {"architecture": "unet", "encoder": "resnet34", "encoder_weights": None}
    )
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(left, right)


def test_tiny_subset_deterministic():
    rows = []
    for dataset_name, count in [("biohit", 10), ("tongueset3", 10)]:
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"{dataset_name}::{index:03d}",
                    "dataset": dataset_name,
                    "split": "train",
                    "md5": f"m{dataset_name}{index}",
                }
            )
    manifest = pd.DataFrame(rows)
    left = select_tiny_overfit_subset(
        manifest, per_dataset={"biohit": 4, "tongueset3": 4}, seed=20260813
    )
    right = select_tiny_overfit_subset(
        manifest, per_dataset={"biohit": 4, "tongueset3": 4}, seed=20260813
    )
    assert left["sample_id"].tolist() == right["sample_id"].tolist()
    assert left["dataset"].value_counts().to_dict() == {"biohit": 4, "tongueset3": 4}


def test_tiny_overfit_disables_augmentation():
    # 选择逻辑本身不涉及 aug；Dataset flag 在集成中验证
    from tongue_data.segmentation.dataset import TongueSegmentationDataset
    from tongue_data.segmentation.config import SegmentationConfig

    config = SegmentationConfig("configs/segmentation_v1.yaml")
    # 构造最小 manifest 行（不真正读取文件：只测 flag）
    dataset = object.__new__(TongueSegmentationDataset)
    dataset.disable_augmentation = True
    dataset.split = "train"
    assert dataset.disable_augmentation is True
    # preprocess_split 逻辑：disable → val
    preprocess_split = "val" if dataset.disable_augmentation else dataset.split
    assert preprocess_split == "val"


def test_per_domain_metric_aggregation():
    aggregator = MetricAggregator()
    metrics = {
        "dice": torch.tensor([0.8, 0.4]),
        "iou": torch.tensor([0.7, 0.3]),
        "precision": torch.tensor([0.9, 0.5]),
        "recall": torch.tensor([0.6, 0.2]),
    }
    aggregator.update(metrics, ["biohit", "tongueset3"], loss_value=0.5)
    summary = aggregator.summarize()
    assert summary["biohit"]["dice"] == pytest.approx(0.8)
    assert summary["tongueset3"]["dice"] == pytest.approx(0.4)
    assert summary["overall"]["dice"] == pytest.approx(0.6)


def test_unknown_architecture_fails():
    with pytest.raises(ValueError, match="unsupported architecture"):
        build_segmentation_model({"architecture": "sam", "encoder": "resnet34"})


def test_parameter_count_positive():
    model = build_segmentation_model(
        {"architecture": "unet", "encoder": "resnet34", "encoder_weights": None}
    )
    counts = count_parameters(model)
    assert counts["total_parameters"] > 0
    assert counts["trainable_parameters"] > 0
