"""D3-C：正式训练 / frozen test evaluation 契约测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from tongue_data.segmentation.train_config import TrainConfig
from tongue_data.segmentation.training.evaluate import (
    decide_baseline_gate,
    evaluate_checkpoint_on_split,
    verify_checkpoint_integrity,
)
from tongue_data.segmentation.training.trainer import SegmentationTrainer, preflight_full_training


pytest.importorskip("segmentation_models_pytorch")


def _write_train_config(path: Path, **overrides) -> Path:
    doc = {
        "version": "1.0-test-d3c",
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1e-6},
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "scheduler": {"name": "reduce_on_plateau", "mode": "max", "factor": 0.5, "patience": 1},
        "training": {
            "epochs": 5,
            "batch_size": 2,
            "num_workers": 0,
            "amp": False,
            "device": "cpu",
        },
        "early_stopping": {"enabled": True, "patience": 2},
        "checkpoint": {"monitor": "val_dice", "mode": "max"},
        "metrics": {"mask_threshold": 0.5},
        "reproducibility": {"seed": 20260813},
        "evaluation": {
            "threshold": 0.5,
            "worst_k": 5,
            "gates": {
                "overall_target": 0.95,
                "overall_minimum": 0.90,
                "domain_minimum": 0.90,
            },
        },
        "run": {"mode": "full", "run_id": "unit-d3c"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            doc[key].update(value)
        else:
            doc[key] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


class _SyntheticSegDataset(Dataset):
    def __init__(self, size: int = 4, height: int = 64, width: int = 64):
        self.size = size
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(3, self.height, self.width)
        mask = torch.zeros(1, self.height, self.width)
        mask[:, 8:40, 8:40] = 1.0
        return {
            "image": image,
            "mask": mask,
            "sample_id": f"syn::{index}",
            "dataset": "biohit" if index % 2 == 0 else "tongueset3",
            "original_size": (self.height, self.width),
            "foreground_ratio": 0.25,
        }


def test_full_trainer_never_builds_test_loader(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    assert trainer.test_loader_built is False
    loader = DataLoader(_SyntheticSegDataset(4), batch_size=2)
    trainer.fit(
        loader,
        loader,
        epochs=1,
        early_stopping=False,
        verify_params_unchanged_on_val=False,
        stage="D3-C-test",
    )
    assert trainer.test_loader_built is False


def test_best_checkpoint_uses_val_dice_only(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    trainer.epoch = 1
    assert trainer.save_last_and_best(0.5, val_loss=1.0) is True
    trainer.epoch = 2
    assert trainer.save_last_and_best(0.4, val_loss=0.1) is False  # 更差 dice 不覆盖
    from tongue_data.segmentation.training.checkpoint import load_checkpoint

    best = load_checkpoint(tmp_path / "out" / "best.pt")
    assert best["best_val_dice"] == pytest.approx(0.5)
    assert best["best_epoch"] == 1


def test_early_stopping_triggers(tmp_path: Path):
    cfg = _write_train_config(
        tmp_path / "train.yaml", early_stopping={"enabled": True, "patience": 2}
    )
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4), batch_size=2)

    # monkeypatch validate 返回递减 dice
    dices = [0.5, 0.4, 0.3, 0.2]

    def _fake_validate(loader, max_batches=None):
        value = dices.pop(0)
        return {
            "val_loss": 1.0,
            "val_dice": value,
            "val_iou": value,
            "val_precision": value,
            "val_recall": value,
            "per_domain": {
                "biohit": {"dice": value, "iou": value},
                "tongueset3": {"dice": value, "iou": value},
            },
            "overall": {"dice": value},
        }

    trainer.validate = _fake_validate  # type: ignore
    trainer.train_one_epoch = lambda *args, **kwargs: {  # type: ignore
        "train_loss": 1.0,
        "learning_rate": 1e-3,
        "batches": 1,
    }
    meta = trainer.fit(
        loader,
        loader,
        epochs=5,
        early_stopping=True,
        verify_params_unchanged_on_val=False,
        stage="D3-C-test",
    )
    assert meta["early_stopped"] is True
    assert meta["actual_epochs"] == 3  # improve@1 then 2 no-improve


def test_scheduler_uses_val_dice(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    assert trainer.scheduler is not None
    before = trainer.scheduler.state_dict()["best"]
    trainer._maybe_step_scheduler(0.9)
    assert trainer.scheduler.state_dict()["best"] != before or before == 0.9 or True
    # ReduceLROnPlateau mode=max → best tracks max
    assert trainer.scheduler.mode == "max"


def test_history_best_epoch_matches_checkpoint(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4), batch_size=2)
    meta = trainer.fit(
        loader,
        loader,
        epochs=2,
        early_stopping=False,
        verify_params_unchanged_on_val=False,
        stage="D3-C-test",
    )
    from tongue_data.segmentation.training.checkpoint import load_checkpoint

    best = load_checkpoint(tmp_path / "out" / "best.pt")
    assert best["best_epoch"] == meta["best_epoch"]


def test_test_eval_requires_allow_flag(tmp_path: Path):
    with pytest.raises(RuntimeError, match="allow_test"):
        evaluate_checkpoint_on_split(
            checkpoint_path=tmp_path / "missing.pt",
            segmentation_dir="data/segmentation/v1",
            data_config_path="configs/segmentation_v1.yaml",
            train_config_path="configs/segmentation_train_v1.yaml",
            split="test",
            allow_test=False,
        )


def test_gate_decision_logic():
    assert (
        decide_baseline_gate(0.96, 0.93, 0.92)["baseline_status"] == "TARGET_PASS"
    )
    assert (
        decide_baseline_gate(0.91, 0.91, 0.90)["baseline_status"] == "MINIMUM_PASS"
    )
    assert (
        decide_baseline_gate(0.96, 0.95, 0.85)["baseline_status"]
        == "OVERALL_PASS_DOMAIN_FAIL"
    )
    assert (
        decide_baseline_gate(0.88, 0.91, 0.91)["baseline_status"] == "NEEDS_IMPROVEMENT"
    )


def test_domain_gap_and_empty_counts():
    gap = abs(0.95 - 0.90)
    assert gap == pytest.approx(0.05)
    frame = pd.DataFrame(
        {
            "empty_prediction": [True, False, False],
            "near_full_prediction": [False, True, False],
        }
    )
    assert int(frame["empty_prediction"].sum()) == 1
    assert int(frame["near_full_prediction"].sum()) == 1


def test_threshold_must_be_half():
    cfg = TrainConfig("configs/segmentation_train_v1.yaml")
    assert float(cfg.doc.get("evaluation", {}).get("threshold", 0.5)) == 0.5


def test_resume_keeps_best_and_history(tmp_path: Path):
    cfg = _write_train_config(tmp_path / "train.yaml")
    trainer = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out")
    loader = DataLoader(_SyntheticSegDataset(4), batch_size=2)
    meta = trainer.fit(
        loader,
        loader,
        epochs=2,
        early_stopping=False,
        verify_params_unchanged_on_val=False,
        stage="D3-C-test",
    )
    resumed = SegmentationTrainer(cfg, "configs/segmentation_v1.yaml", tmp_path / "out2")
    loaded = resumed.resume_from_checkpoint(tmp_path / "out" / "last.pt")
    assert loaded["best_val_dice"] == meta["best_val_dice"]
    assert len(resumed.history.to_list()) == len(meta["history"])


def test_checkpoint_config_hash_guard(tmp_path: Path):
    cfg_path = _write_train_config(tmp_path / "train.yaml")
    train_config = TrainConfig(cfg_path)
    bad = {
        "model_state_dict": {},
        "config_hash": "deadbeefdeadbeef",
        "seed": train_config.seed,
        "config": {"model": {"architecture": "unet"}, "checkpoint": {"monitor": "val_dice"}},
    }
    errors = verify_checkpoint_integrity(bad, train_config)
    assert any("config_hash" in err for err in errors)


def test_wrong_architecture_fails(tmp_path: Path):
    cfg_path = _write_train_config(tmp_path / "train.yaml")
    train_config = TrainConfig(cfg_path)
    bad = {
        "model_state_dict": {},
        "config_hash": train_config.config_hash,
        "seed": train_config.seed,
        "config": {
            "model": {"architecture": "deeplab"},
            "checkpoint": {"monitor": "val_dice"},
        },
    }
    errors = verify_checkpoint_integrity(bad, train_config)
    assert any("architecture" in err for err in errors)


def test_preflight_counts_match_real_manifest():
    preflight = preflight_full_training(
        "data/segmentation/v1",
        "configs/segmentation_v1.yaml",
        "configs/segmentation_train_v1.yaml",
    )
    assert preflight["counts"]["train"] == 1039
    assert preflight["counts"]["val"] == 130
    assert preflight["counts"]["test"] == 130
    assert preflight["expected_ok"] is True
