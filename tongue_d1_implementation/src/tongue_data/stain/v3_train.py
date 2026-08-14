"""D4-C.1-C：MixStyle / GRL representation-invariance 训练（不覆盖 v1/v2）。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .calibrate import calibrate_dual_thresholds, freeze_thresholds
from .config import StainDataConfig, StainTrainConfig
from .consistency import (
    consistency_warmup_factor,
    probability_consistency_loss,
    source_supervised_two_view_loss,
)
from .dataset import select_overfit_subset
from .domain_balanced import create_three_domain_loader, dataset_names_to_domain_ids
from .domain_invariant_model import (
    DOMAIN_TO_ID,
    DomainInvariantStainModel,
    build_domain_invariant_model,
)
from .domain_loader import create_external_loader, create_source_loader
from .grl import GradientReversal, grad_reverse
from .metrics import summarize_ranking_metrics
from .mixstyle import MixStyle
from .model import count_parameters
from .robust_train import (
    _cycle,
    _external_summary,
    _source_eval_loader,
    evaluate_external_roi,
    evaluate_source_split,
)
from .style_augment import estimate_style_ranges_from_train, load_style_contract
from .train import _git_commit, collect_predictions, resolve_device, set_seed


CANDIDATE_RUN_DIRS = {
    "c1": "c1_mixstyle",
    "c2": "c2_grl",
    "c3": "c3_combined",
}


def grl_lambda_at_epoch(epoch: int, schedule: dict[str, Any]) -> float:
    """预注册 GRL lambda schedule；不按 test 调。"""
    schedule_type = str(schedule.get("type", "linear_warmup"))
    lambda_max = float(schedule.get("lambda_max", 0.3))
    warmup = int(schedule.get("warmup_epochs", 5))
    if schedule_type == "constant":
        return lambda_max
    if warmup <= 0:
        return lambda_max
    return float(min(1.0, max(0.0, epoch / float(warmup))) * lambda_max)


def run_grl_unit_smoke() -> dict[str, Any]:
    """GRL：forward identity；backward 符号反转。"""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, 0.5)
    assert torch.allclose(y, x)
    loss = y.sum()
    loss.backward()
    # dy/dx 应为 -0.5 * ones
    expected = -0.5 * torch.ones_like(x)
    ok = bool(torch.allclose(x.grad, expected, atol=1e-6))
    # domain CE finite + encoder adversarial gradient
    encoder = nn.Linear(8, 8)
    head = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    grl = GradientReversal(0.5)
    images = torch.randn(6, 8)
    domain_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    emb = encoder(images)
    logits = head(grl(emb))
    ce = F.cross_entropy(logits, domain_ids)
    assert torch.isfinite(ce)
    # 对比：无 GRL 时 encoder.weight.grad 方向
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    emb = encoder(images)
    logits = head(grl(emb))
    F.cross_entropy(logits, domain_ids).backward()
    grad_with_grl = encoder.weight.grad.detach().clone()
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    emb = encoder(images)
    logits = head(emb)
    F.cross_entropy(logits, domain_ids).backward()
    grad_without = encoder.weight.grad.detach().clone()
    # GRL 应对 encoder 产生反向梯度（与无 GRL 点积为负）
    cos = float(
        torch.nn.functional.cosine_similarity(
            grad_with_grl.flatten(), grad_without.flatten(), dim=0
        ).item()
    )
    reversed_ok = cos < -0.5
    return {
        "passed": bool(ok and reversed_ok and torch.isfinite(ce)),
        "forward_identity": True,
        "backward_sign_reversed": ok,
        "domain_ce_finite": True,
        "encoder_adversarial_cosine_vs_non_grl": cos,
        "encoder_gradient_reversed": reversed_ok,
    }


def run_mixstyle_unit_smoke() -> dict[str, Any]:
    """MixStyle shape/dtype/grad/deterministic/eval-off。"""
    torch.manual_seed(1)
    mix = MixStyle(p=1.0, alpha=0.1)
    mix.train()
    feature = torch.randn(6, 16, 8, 8, requires_grad=True)
    domain_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    out = mix(feature, domain_ids)
    shape_ok = out.shape == feature.shape
    dtype_ok = out.dtype == feature.dtype
    loss = out.mean()
    loss.backward()
    grad_ok = feature.grad is not None and bool(torch.isfinite(feature.grad).all())
    # deterministic under fixed seed
    torch.manual_seed(123)
    mix2 = MixStyle(p=1.0, alpha=0.1)
    mix2.train()
    base = torch.randn(4, 8, 4, 4)
    torch.manual_seed(999)
    out_a = mix2(base.clone(), torch.tensor([0, 1, 0, 1]))
    torch.manual_seed(999)
    out_b = mix2(base.clone(), torch.tensor([0, 1, 0, 1]))
    det_ok = bool(torch.allclose(out_a, out_b))
    # eval off
    mix.eval()
    feature2 = torch.randn(4, 8, 4, 4)
    out_eval = mix(feature2)
    eval_identity = bool(torch.allclose(out_eval, feature2))
    return {
        "passed": bool(shape_ok and dtype_ok and grad_ok and det_ok and eval_identity),
        "shape_preserved": shape_ok,
        "dtype_preserved": dtype_ok,
        "gradient_finite": grad_ok,
        "deterministic": det_ok,
        "eval_disabled": eval_identity,
        "label_mixing": False,
    }


def run_v3_tiny_overfit(
    *,
    stain_manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """关闭 MixStyle/GRL/consistency，验证基础 stain path。"""
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
    # 强制关闭 mixstyle/grl
    model = DomainInvariantStainModel(
        train_cfg.model,
        mixstyle_cfg={"enabled": False},
        domain_cfg={"enabled": False},
    ).to(device_t)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.optimizer.get("lr", 1e-3)),
        weight_decay=float(train_cfg.optimizer.get("weight_decay", 1e-4)),
    )
    history = []
    epochs = int(overfit_cfg.get("epochs", 40))
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        losses = []
        for batch in loader:
            weak = batch["image_weak"].to(device_t)
            style = batch["image_style"].to(device_t)
            labels = batch["label"].to(device_t)
            domain_ids = torch.zeros(weak.size(0), dtype=torch.long, device=device_t)
            optimizer.zero_grad(set_to_none=True)
            logit_weak = model.forward_stain(weak, domain_ids=domain_ids)
            logit_style = model.forward_stain(style, domain_ids=domain_ids)
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
        "mixstyle_disabled": True,
        "grl_disabled": True,
        "consistency_disabled": True,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiny_overfit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _forward_stain_split(
    model: DomainInvariantStainModel,
    source_images: torch.Tensor,
    external_images: torch.Tensor,
    external_datasets: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """拼接 source+external 一次前向，便于 cross-domain MixStyle；再拆回。"""
    n_source = source_images.size(0)
    domain_source = torch.full(
        (n_source,), DOMAIN_TO_ID["stained"], dtype=torch.long, device=source_images.device
    )
    domain_external = dataset_names_to_domain_ids(external_datasets).to(source_images.device)
    all_images = torch.cat([source_images, external_images], dim=0)
    all_domains = torch.cat([domain_source, domain_external], dim=0)
    all_logits = model.forward_stain(all_images, domain_ids=all_domains)
    return all_logits[:n_source], all_logits[n_source:]


def train_stain_v3_candidate(
    *,
    candidate: str,
    stain_manifest: str | Path = "data/stain/v1/stain_manifest.parquet",
    roi_index: str | Path = "reports/d4c1/roi_cache/index.parquet",
    data_config: str | Path = "configs/stain_detection_v3.yaml",
    train_config: str | Path = "configs/stain_train_v3.yaml",
    style_contract_path: str | Path = "reports/d4c1b/style_augmentation_contract.json",
    output_root: str | Path = "runs/input_guard/d4c1c",
    device: str = "auto",
    max_epochs: int | None = None,
    allow_c3: bool = False,
) -> dict[str, Any]:
    """
    训练单一预注册 candidate：c1|c2|c3。
    best checkpoint 仅由 source val AUROC 决定。
    """
    candidate = str(candidate).lower()
    if candidate not in {"c1", "c2", "c3"}:
        raise ValueError(f"candidate must be c1/c2/c3, got {candidate}")
    if candidate == "c3" and not allow_c3:
        raise RuntimeError("C3 requires allow_c3=True after C1/C2 robustness signal")

    train_cfg = StainTrainConfig(train_config)
    data_cfg = StainDataConfig(data_config)
    set_seed(train_cfg.seed)
    device_t = resolve_device(device)
    run_name = CANDIDATE_RUN_DIRS[candidate]
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(style_contract_path).exists():
        estimate_style_ranges_from_train(
            stain_manifest=stain_manifest,
            external_roi_index=roi_index,
            output_path=style_contract_path,
        )
    style_contract = load_style_contract(style_contract_path)

    if train_cfg.model.get("init_from_v1") or train_cfg.model.get("init_from_v2"):
        raise RuntimeError("v3 must not fine-tune from v1/v2 checkpoints")

    source_loader = create_source_loader(
        stain_manifest,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=int(train_cfg.training.get("batch_size", 24)),
        disable_style=False,
        shuffle=True,
    )
    external_loader = create_external_loader(
        roi_index,
        data_config,
        train_config,
        style_contract,
        split="train",
        batch_size=int(train_cfg.training.get("external_batch_size", 24)),
        biohit_fraction=float(train_cfg.doc.get("external", {}).get("biohit_fraction", 0.5)),
    )
    domain_loader = create_three_domain_loader(
        stain_manifest,
        roi_index,
        data_config,
        train_config,
        style_contract,
        split="train",
        per_domain=int(train_cfg.training.get("domain_per_domain", 8)),
    )
    val_loader = _source_eval_loader(stain_manifest, data_config, train_config, "val")

    model = build_domain_invariant_model(train_cfg.doc, candidate=candidate).to(device_t)
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
    domain_cfg = dict(train_cfg.doc.get("domain_adversarial", {}))
    mix_cfg = dict(train_cfg.doc.get("mixstyle", {}))
    epochs = int(max_epochs or train_cfg.training.get("epochs", 25))
    patience = int(train_cfg.early_stopping.get("patience", 6))
    warmup_epochs = int(loss_cfg.get("consistency_warmup_epochs", 5))
    stop_grad = bool(train_cfg.doc.get("external", {}).get("stop_gradient_weak_target", True))
    use_domain = bool(model.domain_enabled)
    use_mix = bool(model.mix_hooks is not None)
    domain_loss_weight = float(domain_cfg.get("domain_loss_weight", 1.0))
    lambda_schedule = dict(domain_cfg.get("lambda_schedule", {}))

    history: list[dict[str, Any]] = []
    best_auroc = -1.0
    best_epoch = -1
    bad_epochs = 0
    external_iter = _cycle(external_loader)
    domain_iter = _cycle(domain_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        warmup = consistency_warmup_factor(epoch, warmup_epochs)
        lambda_domain = grl_lambda_at_epoch(epoch, lambda_schedule) if use_domain else 0.0
        model.set_grl_lambda(lambda_domain)
        # MixStyle warmup：默认 epoch1 即开
        mix_warmup = int(mix_cfg.get("warmup_epochs", 0))
        mix_on = use_mix and epoch > mix_warmup
        model.set_mixstyle_enabled(mix_on)

        meters = {
            "supervised": [],
            "source_consistency": [],
            "external_consistency": [],
            "domain": [],
            "total": [],
            "domain_acc": [],
        }
        for source_batch in source_loader:
            external_batch = next(external_iter)
            domain_batch = next(domain_iter)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                sw = source_batch["image_weak"].to(device_t)
                ss = source_batch["image_style"].to(device_t)
                labels = source_batch["label"].to(device_t)
                ew = external_batch["image_weak"].to(device_t)
                es = external_batch["image_style"].to(device_t)
                ext_names = list(external_batch["dataset"])

                # cross-domain MixStyle：source+external 拼接前向
                logit_sw, logit_ew = _forward_stain_split(model, sw, ew, ext_names)
                logit_ss, logit_es = _forward_stain_split(model, ss, es, ext_names)

                supervised = source_supervised_two_view_loss(logit_sw, logit_ss, labels)
                source_cons = probability_consistency_loss(
                    logit_ss, logit_sw, stop_gradient_teacher=stop_grad
                )
                external_cons = probability_consistency_loss(
                    logit_es, logit_ew, stop_gradient_teacher=stop_grad
                )
                # 硬禁：external 不得进 BCE（labels 仅来自 source_batch）
                assert external_batch["has_gold_label"] is False
                assert "label" not in external_batch

                domain_loss = torch.zeros((), device=device_t)
                domain_acc = 0.0
                if use_domain:
                    domain_images = domain_batch["image"].to(device_t)
                    domain_ids = domain_batch["domain_id"].to(device_t)
                    domain_logits = model.forward_domain(
                        domain_images, domain_ids=domain_ids
                    )
                    domain_loss = F.cross_entropy(domain_logits, domain_ids)
                    pred_domain = domain_logits.argmax(dim=1)
                    domain_acc = float((pred_domain == domain_ids).float().mean().item())

                total = (
                    float(loss_cfg.get("supervised_weight", 1.0)) * supervised
                    + float(loss_cfg.get("source_consistency_weight", 0.5))
                    * warmup
                    * source_cons
                    + float(loss_cfg.get("external_consistency_weight", 0.5))
                    * warmup
                    * external_cons
                    + domain_loss_weight * domain_loss
                )

            scaler.scale(total).backward()
            if train_cfg.training.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_cfg.training["gradient_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()
            meters["supervised"].append(float(supervised.item()))
            meters["source_consistency"].append(float(source_cons.item()))
            meters["external_consistency"].append(float(external_cons.item()))
            meters["domain"].append(float(domain_loss.item()) if use_domain else 0.0)
            meters["total"].append(float(total.item()))
            meters["domain_acc"].append(domain_acc)

        # validation：强制关闭 MixStyle / style
        model.eval()
        model.set_mixstyle_enabled(False)
        val_pred = collect_predictions(model, val_loader, device_t)
        val_metrics = summarize_ranking_metrics(
            val_pred["label"].to_numpy(), val_pred["p_stain"].to_numpy()
        )
        ext_val = evaluate_external_roi(
            model,
            roi_index=roi_index,
            data_config=data_config,
            split="val",
            device=device_t,
            max_per_dataset=40,
        )
        ext_mon = _external_summary(ext_val, 0.95, 0.96)
        row = {
            "epoch": epoch,
            "candidate": candidate,
            "train_supervised_loss": float(np.mean(meters["supervised"])),
            "source_consistency_loss": float(np.mean(meters["source_consistency"])),
            "external_consistency_loss": float(np.mean(meters["external_consistency"])),
            "domain_loss": float(np.mean(meters["domain"])),
            "domain_acc_train": float(np.mean(meters["domain_acc"])),
            "train_total_loss": float(np.mean(meters["total"])),
            "val_auroc": float(val_metrics["auroc"]),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "consistency_warmup": warmup,
            "grl_lambda": lambda_domain,
            "mixstyle_on": mix_on,
            "external_val_monitor": ext_mon,
            "selection_metric": "source_val_auroc",
        }
        history.append(row)
        scheduler.step(row["val_auroc"])
        print(
            f"[v3-{candidate}] epoch={epoch} val_auroc={row['val_auroc']:.4f} "
            f"λ={lambda_domain:.3f} mix={mix_on} "
            f"ts3_p50={ext_mon.get('tongueset3', {}).get('median_p')}",
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
                    "stage": "D4-C.1-C",
                    "candidate": candidate,
                    "stain_contract_version": data_cfg.version,
                    "mixstyle_enabled": use_mix,
                    "domain_adversarial_enabled": use_domain,
                    "init_from_v1": False,
                    "init_from_v2": False,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "candidate": candidate,
            },
            output_dir / "last.pt",
        )
        # source gate early abort：若长期远低于 0.95 可继续等；正式 early stop 仍按 patience
        if train_cfg.early_stopping.get("enabled", True) and bad_epochs >= patience:
            print(f"[v3-{candidate}] early stop at epoch={epoch}", flush=True)
            break

    best_blob = torch.load(output_dir / "best.pt", map_location=device_t, weights_only=False)
    model.load_state_dict(best_blob["model_state_dict"])
    model.eval()
    model.set_mixstyle_enabled(False)
    val_pred, val_metrics = evaluate_source_split(
        model,
        stain_manifest=stain_manifest,
        data_config=data_config,
        train_config=train_config,
        split="val",
        device=device_t,
    )
    val_pred.to_parquet(output_dir / "val_predictions.parquet", index=False)

    metadata = {
        "stage": "D4-C.1-C",
        "candidate": candidate,
        "stain_contract_version": data_cfg.version,
        "train_config_hash": train_cfg.config_hash,
        "data_config_hash": data_cfg.config_hash,
        "seed": train_cfg.seed,
        "planned_epochs": int(train_cfg.training.get("epochs", 25)),
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "best_source_val_auroc": best_auroc,
        "best_source_val_pr_auc": float(val_metrics["pr_auc"]),
        "mixstyle": {
            "enabled": use_mix,
            "layers": list(mix_cfg.get("layers", ["layer1"])) if use_mix else [],
            "p": mix_cfg.get("p"),
            "alpha": mix_cfg.get("alpha"),
            "mixing_strategy": mix_cfg.get("mixing_strategy"),
        },
        "domain_adversarial": {
            "enabled": use_domain,
            "lambda_schedule": lambda_schedule if use_domain else None,
            "domain_loss_weight": domain_loss_weight if use_domain else 0.0,
        },
        "source_consistency": True,
        "external_consistency": True,
        "params": count_parameters(model),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "external_pseudo_labels": False,
        "entropy_minimization": False,
        "selection_rule": "source_val_auroc_only",
        "policy_activation": False,
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
        "model": model,
    }


def load_v3_checkpoint(
    path: str | Path,
    *,
    candidate: str,
    train_config: str | Path = "configs/stain_train_v3.yaml",
    map_location: str | torch.device = "cpu",
    strict_hash: bool = True,
) -> tuple[DomainInvariantStainModel, dict]:
    train_cfg = StainTrainConfig(train_config)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if strict_hash:
        if ckpt.get("train_config_hash") != train_cfg.config_hash:
            raise ValueError(
                f"train config hash mismatch ckpt={ckpt.get('train_config_hash')} "
                f"cfg={train_cfg.config_hash}"
            )
    model = build_domain_invariant_model(train_cfg.doc, candidate=candidate)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.set_mixstyle_enabled(False)
    return model, ckpt


def calibrate_v3_thresholds(
    *,
    model: nn.Module,
    stain_manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """仅 source VAL 校准；不得继承 v1/v2。"""
    train_cfg = StainTrainConfig(train_config)
    val_pred, _metrics = evaluate_source_split(
        model,
        stain_manifest=stain_manifest,
        data_config=data_config,
        train_config=train_config,
        split="val",
        device=device,
    )
    cal = calibrate_dual_thresholds(
        val_pred["label"].to_numpy(),
        val_pred["p_stain"].to_numpy(),
        target_confident_precision=float(
            train_cfg.calibration.get("target_confident_precision", 0.90)
        ),
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thr_path = out / "thresholds.json"
    freeze_thresholds(cal, thr_path)
    thr_doc = json.loads(thr_path.read_text(encoding="utf-8"))
    thr_doc.update(
        {
            "stage": "D4-C.1-C",
            "t_clear_v3": cal["t_clear"],
            "t_retake_v3": cal["t_retake"],
            "calibrated_on": "stained_val_only",
            "v1_thresholds_preserved": {"t_clear": 0.95, "t_retake": 0.96},
            "v2_not_overwritten": True,
        }
    )
    thr_path.write_text(json.dumps(thr_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return thr_doc
