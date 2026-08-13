"""Stain：val 校准 + frozen test 一次性评估。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibrate import (
    apply_frozen_thresholds_to_frame,
    calibrate_dual_thresholds,
    freeze_thresholds,
    load_frozen_thresholds,
)
from .config import StainDataConfig, StainTrainConfig
from .dataset import StainRoiDataset
from .metrics import summarize_ranking_metrics, three_state_metrics
from .train import collect_predictions, load_stain_checkpoint, resolve_device


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "md5": [item["md5"] for item in batch],
    }


def run_val_calibration(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """仅用 val predictions 校准 t_clear / t_retake。"""
    run_dir = Path(run_dir)
    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    device_obj = resolve_device(device if device != "auto" else train_config.device)

    pred_path = run_dir / "val_predictions.parquet"
    if pred_path.exists():
        val_pred = pd.read_parquet(pred_path)
    else:
        manifest = pd.read_parquet(manifest_path)
        eligible = manifest[manifest["eligible"] == True]
        val_ds = StainRoiDataset(
            eligible, data_config, train_config, split="val", seed=train_config.seed
        )
        loader = DataLoader(
            val_ds,
            batch_size=int(train_config.training.get("batch_size", 32)),
            shuffle=False,
            collate_fn=_collate,
        )
        model, _ckpt = load_stain_checkpoint(
            run_dir / "best.pt",
            train_config=train_config,
            data_config=data_config,
            map_location=device_obj,
            strict=True,
        )
        model = model.to(device_obj)
        val_pred = collect_predictions(model, loader, device_obj)
        val_pred.to_parquet(pred_path, index=False)

    target = float(
        train_config.calibration.get("target_confident_precision", 0.90)
    )
    calibration = calibrate_dual_thresholds(
        val_pred["label"].to_numpy(),
        val_pred["p_stain"].to_numpy(),
        target_confident_precision=target,
    )
    freeze_thresholds(calibration, run_dir / "thresholds.json")
    (run_dir / "val_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 不允许写入任何 test 相关阈值重算字段
    calibration["test_used"] = False
    return calibration


def run_frozen_test_evaluation(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    device: str = "auto",
    allow_test: bool = False,
    d4b_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """模型与 thresholds 全部 freeze 后，test 只评估一次。"""
    if not allow_test:
        raise ValueError("test evaluation requires allow_test=True after freeze")
    run_dir = Path(run_dir)
    thresholds_path = run_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError("thresholds.json missing; run stain-calibrate first")
    frozen = load_frozen_thresholds(thresholds_path)
    # 禁止在本函数内重算阈值
    t_clear = float(frozen["t_clear"])
    t_retake = float(frozen["t_retake"])

    data_config = StainDataConfig(data_config_path)
    train_config = StainTrainConfig(train_config_path)
    device_obj = resolve_device(device if device != "auto" else train_config.device)

    manifest = pd.read_parquet(manifest_path)
    eligible = manifest[manifest["eligible"] == True]
    test_ds = StainRoiDataset(
        eligible, data_config, train_config, split="test", seed=train_config.seed
    )
    loader = DataLoader(
        test_ds,
        batch_size=int(train_config.training.get("batch_size", 32)),
        shuffle=False,
        collate_fn=_collate,
    )
    model, ckpt = load_stain_checkpoint(
        run_dir / "best.pt",
        train_config=train_config,
        data_config=data_config,
        map_location=device_obj,
        strict=True,
    )
    model = model.to(device_obj)
    test_pred = collect_predictions(model, loader, device_obj)
    test_pred = apply_frozen_thresholds_to_frame(
        test_pred, t_clear=t_clear, t_retake=t_retake
    )
    test_pred.to_parquet(run_dir / "test_predictions.parquet", index=False)

    ranking = summarize_ranking_metrics(test_pred["label"], test_pred["p_stain"])
    three = three_state_metrics(
        test_pred["label"],
        test_pred["p_stain"],
        t_clear=t_clear,
        t_retake=t_retake,
    )

    # confusion audit（不存原图）
    false_negatives = test_pred[
        (test_pred["label"] == 1) & (test_pred["finding"] != "true")
    ][["sample_id", "label", "p_stain", "finding"]].copy()
    false_positives = test_pred[
        (test_pred["label"] == 0) & (test_pred["finding"] == "true")
    ][["sample_id", "label", "p_stain", "finding"]].copy()

    # 可选 D4-B stratum audit
    stratum = None
    if d4b_audit_path and Path(d4b_audit_path).exists():
        d4b = pd.read_json(d4b_audit_path)
        if isinstance(d4b, pd.DataFrame) and "sample_id" in d4b.columns:
            merged = test_pred.merge(
                d4b[["sample_id", "decision"]]
                if "decision" in d4b.columns
                else d4b,
                on="sample_id",
                how="left",
            )
            if "decision" in merged.columns:
                stratum = {}
                for status in ("pass", "warning", "retake"):
                    subset = merged[merged["decision"] == status]
                    if subset.empty:
                        stratum[status] = {"n": 0}
                        continue
                    stratum[status] = {
                        "n": int(len(subset)),
                        "auroc": summarize_ranking_metrics(
                            subset["label"], subset["p_stain"]
                        )["auroc"],
                        "uncertain_rate": float(
                            (subset["finding"] == "uncertain").mean()
                        ),
                    }

    gates = train_config.gates
    target = gates.get("target", {})
    minimum = gates.get("minimum", {})

    def _meet(spec: dict) -> bool:
        checks = [
            (ranking["auroc"] or 0.0) >= float(spec.get("test_auroc", 0.85)),
            (three["confident_stain_precision"] or 0.0)
            >= float(spec.get("confident_stain_precision", 0.85)),
            (three["stain_recall"] or 0.0) >= float(spec.get("stain_recall", 0.85)),
            (three["confident_clean_purity"] or 0.0)
            >= float(spec.get("confident_clean_purity", 0.85)),
        ]
        return all(checks)

    if _meet(target):
        baseline_status = "TARGET_PASS"
    elif _meet(minimum):
        baseline_status = "MINIMUM_PASS"
    else:
        baseline_status = "NEEDS_IMPROVEMENT"

    coverage_warning = None
    if three["confident_coverage"] < 0.30:
        coverage_warning = "confident_coverage_extremely_low"

    report = {
        "stage": "D4-C",
        "split": "test",
        "thresholds_source": "val_frozen",
        "t_clear": t_clear,
        "t_retake": t_retake,
        "threshold_recomputed_on_test": False,
        "best_epoch": ckpt.get("epoch"),
        "best_val_auroc": ckpt.get("best_val_auroc"),
        "ranking": ranking,
        "three_state": {
            key: value
            for key, value in three.items()
            if key != "findings"
        },
        "false_negative_count": int(len(false_negatives)),
        "false_positive_count": int(len(false_positives)),
        "false_negatives": false_negatives.to_dict(orient="records"),
        "false_positives": false_positives.to_dict(orient="records"),
        "d4b_stratum_audit": stratum,
        "baseline_status": baseline_status,
        "coverage_warning": coverage_warning,
        "n_test": int(len(test_pred)),
    }
    (run_dir / "test_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reports_dir = Path("reports/d4")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "d4c_test_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
