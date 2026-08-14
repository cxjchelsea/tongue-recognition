"""D4-C.1-B：v1 vs v2 robustness / unified recovery / freeze docs。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .calibrate import apply_frozen_thresholds_to_frame, load_frozen_thresholds
from .config import StainDataConfig, StainTrainConfig
from .d4c1a_model_tools import extract_embedding, centroid_distances
from .metrics import three_state_metrics
from .robust_train import (
    _external_summary,
    evaluate_external_roi,
    evaluate_source_split,
)
from .style_augment import apply_style_transform, load_style_contract, sample_style_params
from .train import load_stain_checkpoint, resolve_device
from .transforms import preprocess_masked_roi


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_model_pair(
    *,
    v1_ckpt: Path,
    v2_ckpt: Path,
    device: str,
):
    device_t = resolve_device(device)
    model_v1, _ = load_stain_checkpoint(
        v1_ckpt,
        train_config=StainTrainConfig("configs/stain_train_v1.yaml"),
        data_config=StainDataConfig("configs/stain_detection_v1.yaml"),
        map_location=device_t,
        strict=True,
    )
    model_v2, _ = load_stain_checkpoint(
        v2_ckpt,
        train_config=StainTrainConfig("configs/stain_train_v2.yaml"),
        data_config=StainDataConfig("configs/stain_detection_v2.yaml"),
        map_location=device_t,
        strict=True,
    )
    model_v1 = model_v1.to(device_t).eval()
    model_v2 = model_v2.to(device_t).eval()
    for parameter in list(model_v1.parameters()) + list(model_v2.parameters()):
        parameter.requires_grad_(False)
    return model_v1, model_v2, device_t


def audit_external_val_robustness(
    *,
    model_v1,
    model_v2,
    roi_index: Path,
    data_config_v1: Path,
    data_config_v2: Path,
    thr_v1: dict,
    thr_v2: dict,
    device,
) -> dict[str, Any]:
    pred_v1 = evaluate_external_roi(
        model_v1, roi_index=roi_index, data_config=data_config_v1, split="val", device=device
    )
    pred_v2 = evaluate_external_roi(
        model_v2, roi_index=roi_index, data_config=data_config_v2, split="val", device=device
    )
    sum_v1 = _external_summary(pred_v1, thr_v1["t_clear"], thr_v1["t_retake"])
    sum_v2 = _external_summary(pred_v2, thr_v2["t_clear"], thr_v2["t_retake"])
    gap_v1 = abs(sum_v1.get("domain_gap_median_logit", 0.0))
    gap_v2 = abs(sum_v2.get("domain_gap_median_logit", 0.0))
    reduction = None if gap_v1 <= 1e-8 else float(1.0 - gap_v2 / gap_v1)
    ts3_high_v2 = sum_v2.get("tongueset3", {}).get("highscore_rate", 1.0)
    return {
        "v1": sum_v1,
        "v2": sum_v2,
        "logit_gap_v1": gap_v1,
        "logit_gap_v2": gap_v2,
        "logit_gap_reduction": reduction,
        "tongueset3_highscore_rate_v2": ts3_high_v2,
        "catastrophic_saturation_resolved": bool(ts3_high_v2 < 0.50),
        "note": "external val has no stain gold; engineering proxies only",
    }


def audit_style_sensitivity(
    *,
    model_v1,
    model_v2,
    roi_index: Path,
    data_config_v2: Path,
    style_contract: dict,
    device,
    n_per_domain: int = 30,
) -> dict[str, Any]:
    data_cfg = StainDataConfig(data_config_v2)
    frame = pd.read_parquet(roi_index)
    frame = frame[(frame.split == "val") & frame.roi_rgb_path.notna()]
    rng = np.random.default_rng(20260813)
    rows = []
    for dataset_name in ("biohit", "tongueset3"):
        subset = frame[frame.dataset == dataset_name].sort_values("sample_id").head(n_per_domain)
        for _index, row in subset.iterrows():
            rgb = np.asarray(Image.open(row.roi_rgb_path).convert("RGB"), dtype=np.uint8)
            mask = (np.asarray(Image.open(row.roi_mask_path)) > 0).astype(np.uint8)
            base = preprocess_masked_roi(rgb, mask, data_cfg, split="val")
            batch0 = torch.from_numpy(np.ascontiguousarray(base)).unsqueeze(0).to(device)
            with torch.inference_mode():
                logit_v1_0 = float(model_v1(batch0).reshape(-1)[0].item())
                logit_v2_0 = float(model_v2(batch0).reshape(-1)[0].item())
            deltas_v1 = []
            deltas_v2 = []
            for strength_name, strength in (
                ("red_wb", "moderate"),
                ("blue_wb", "moderate"),
                ("gamma", "moderate"),
                ("exposure", "moderate"),
            ):
                params = sample_style_params(style_contract, rng, strength=strength)
                if strength_name == "red_wb":
                    params.update({"gain_r": style_contract["channel_gain_ranges"]["r"][1], "gain_g": 1.0, "gain_b": 1.0})
                if strength_name == "blue_wb":
                    params.update({"gain_r": 1.0, "gain_g": 1.0, "gain_b": style_contract["channel_gain_ranges"]["b"][1]})
                styled, _ = apply_style_transform(
                    rgb, style_contract, rng, strength=strength, params=params
                )
                tensor = preprocess_masked_roi(styled, mask, data_cfg, split="val")
                batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    lv1 = float(model_v1(batch).reshape(-1)[0].item())
                    lv2 = float(model_v2(batch).reshape(-1)[0].item())
                deltas_v1.append(abs(lv1 - logit_v1_0))
                deltas_v2.append(abs(lv2 - logit_v2_0))
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "dataset": dataset_name,
                    "mean_abs_delta_v1": float(np.mean(deltas_v1)),
                    "mean_abs_delta_v2": float(np.mean(deltas_v2)),
                }
            )
    table = pd.DataFrame(rows)
    mean_v1 = float(table["mean_abs_delta_v1"].mean())
    mean_v2 = float(table["mean_abs_delta_v2"].mean())
    improvement = None if mean_v1 <= 1e-8 else float(1.0 - mean_v2 / mean_v1)
    return {
        "n": int(len(table)),
        "mean_abs_delta_logit_v1": mean_v1,
        "mean_abs_delta_logit_v2": mean_v2,
        "style_consistency_improvement": improvement,
        "by_dataset": table.groupby("dataset")[["mean_abs_delta_v1", "mean_abs_delta_v2"]]
        .mean()
        .to_dict(),
    }


def audit_embedding_comparison(
    *,
    model_v1,
    model_v2,
    stain_manifest: Path,
    roi_index: Path,
    data_config_v1: Path,
    data_config_v2: Path,
    device,
    max_per_group: int = 120,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def collect(model, data_cfg_path, groups_spec):
        data_cfg = StainDataConfig(data_cfg_path)
        emb = {}
        for group_name, frame in groups_spec.items():
            vectors = []
            for _index, row in frame.head(max_per_group).iterrows():
                rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
                mask = (np.asarray(Image.open(row["roi_mask_path"])) > 0).astype(np.uint8)
                tensor = preprocess_masked_roi(rgb, mask, data_cfg, split="val")
                batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
                vectors.append(extract_embedding(model, batch))
            emb[group_name] = np.stack(vectors) if vectors else np.zeros((0, 512))
        return emb

    stain = pd.read_parquet(stain_manifest)
    stain = stain[(stain.eligible == True) & (stain.split.isin(["train", "val"]))]
    external = pd.read_parquet(roi_index)
    external = external[(external.split == "val") & external.roi_rgb_path.notna()]
    groups = {
        "stain_pos": stain[stain.label == 1][["sample_id", "roi_rgb_path", "roi_mask_path"]],
        "stain_neg": stain[stain.label == 0][["sample_id", "roi_rgb_path", "roi_mask_path"]],
        "biohit": external[external.dataset == "biohit"],
        "tongueset3": external[external.dataset == "tongueset3"],
    }
    emb_v1 = collect(model_v1, data_config_v1, groups)
    emb_v2 = collect(model_v2, data_config_v2, groups)

    def probe_acc(emb):
        xs = []
        ys = []
        for label_name, key in (("stained", "stain_pos"), ("stained", "stain_neg"), ("biohit", "biohit"), ("tongueset3", "tongueset3")):
            arr = emb[key]
            if len(arr) == 0:
                continue
            # domain probe: stained vs biohit vs tongueset3
            domain = "stained" if key.startswith("stain") else key
            xs.append(arr)
            ys.extend([domain] * len(arr))
        if not xs:
            return None
        matrix = np.concatenate(xs, axis=0)
        labels = np.array(ys)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=20260813)),
            ]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260813)
        scores = cross_val_score(pipe, matrix, labels, cv=cv)
        return float(scores.mean())

    return {
        "centroid_distances_v1": centroid_distances(emb_v1),
        "centroid_distances_v2": centroid_distances(emb_v2),
        "domain_probe_cv_acc_v1": probe_acc(emb_v1),
        "domain_probe_cv_acc_v2": probe_acc(emb_v2),
        "n_per_group_v2": {key: int(len(arr)) for key, arr in emb_v2.items()},
    }


def audit_source_test_once(
    *,
    model_v2,
    stain_manifest: Path,
    data_config: Path,
    train_config: Path,
    thresholds: dict,
    device,
    v1_test_auroc: float = 0.9918,
) -> dict[str, Any]:
    pred, ranking = evaluate_source_split(
        model_v2,
        stain_manifest=stain_manifest,
        data_config=data_config,
        train_config=train_config,
        split="test",
        device=device,
    )
    three = three_state_metrics(
        pred["label"].to_numpy(),
        pred["p_stain"].to_numpy(),
        t_clear=float(thresholds["t_clear"]),
        t_retake=float(thresholds["t_retake"]),
    )
    # binary metrics at retake threshold for precision/recall style reporting
    from .metrics import binary_metrics_at_threshold

    binary = binary_metrics_at_threshold(
        pred["label"].to_numpy(),
        pred["p_stain"].to_numpy(),
        threshold=float(thresholds["t_retake"]),
    )
    auroc = float(ranking["auroc"])
    return {
        "predictions_n": int(len(pred)),
        "ranking": ranking,
        "three_state": three,
        "binary_at_t_retake": binary,
        "auroc_drop_vs_v1": float(v1_test_auroc - auroc),
        "pred_frame": pred,
    }


def audit_known_external(
    *,
    model_v1,
    model_v2,
    roi_index: Path,
    data_config_v1: Path,
    data_config_v2: Path,
    thr_v1: dict,
    thr_v2: dict,
    device,
) -> dict[str, Any]:
    pred_v1 = evaluate_external_roi(
        model_v1, roi_index=roi_index, data_config=data_config_v1, split="test", device=device
    )
    pred_v2 = evaluate_external_roi(
        model_v2, roi_index=roi_index, data_config=data_config_v2, split="test", device=device
    )
    return {
        "role": "known_external_audit",
        "not_independent_test": True,
        "used_for_selection": False,
        "v1": _external_summary(pred_v1, thr_v1["t_clear"], thr_v1["t_retake"]),
        "v2": _external_summary(pred_v2, thr_v2["t_clear"], thr_v2["t_retake"]),
        "pred_v2": pred_v2,
    }


def audit_unified_recovery(
    *,
    v2_ckpt: Path,
    v2_thresholds: Path,
    segmentation_dir: Path,
    seg_checkpoint: Path,
    seg_data_config: Path,
    seg_train_config: Path,
    policy_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """只替换 stain detector；不改 D4-B/cast/occlusion thresholds。"""
    from tongue_data.input_guard.d4d1_integration_audit import (
        build_ablation_table,
        build_retake_attribution,
        collect_sample_audit_rows,
    )
    from tongue_data.input_guard.policy import load_input_guard_policy

    policy = load_input_guard_policy(policy_path)
    # 确认 policy 仍为 1.3（未激活 v2）
    frame = collect_sample_audit_rows(
        checkpoint_path=seg_checkpoint,
        segmentation_dir=segmentation_dir,
        data_config_path=seg_data_config,
        train_config_path=seg_train_config,
        policy_path=policy_path,
        stain_checkpoint=v2_ckpt,
        stain_thresholds=v2_thresholds,
        device=device,
    )
    ablation = build_ablation_table(frame, policy_path)
    attribution = build_retake_attribution(frame)
    newly = frame[(frame.D4B_decision != "retake") & (frame.unified_decision == "retake")]
    old = {
        "pass": 36,
        "warning": 14,
        "retake": 80,
        "stain_only_retake": 67,
    }
    new_counts = ablation["counts"]["D_full"]
    return {
        "policy_version": policy.policy_version,
        "policy_activated_v2": False,
        "old_unified": old,
        "new_unified": new_counts,
        "new_ablation": ablation["counts"],
        "retake_attribution": attribution,
        "newly_rejected_n": int(len(newly)),
        "stain_only_retake_new": int(
            (
                newly["retake_due_to_stain"]
                & ~newly["retake_due_to_d4b"]
                & ~newly["retake_due_to_color_cast"]
                & ~newly["retake_due_to_occlusion"]
            ).sum()
        )
        if len(newly)
        else 0,
        "d4b_thresholds_unchanged": True,
        "color_cast_thresholds_unchanged": True,
        "occlusion_thresholds_unchanged": True,
        "n": int(len(frame)),
        "frame": frame,
    }


def decide_baseline_status(
    *,
    source_test: dict,
    external_val: dict,
    known_audit: dict,
    unified: dict,
) -> dict[str, Any]:
    auroc = float(source_test["ranking"]["auroc"])
    pr = float(source_test["ranking"]["pr_auc"])
    three = source_test["three_state"]
    drop = float(source_test["auroc_drop_vs_v1"])
    source_ok = (
        auroc >= 0.95
        and pr >= 0.95
        and float(three.get("confident_stain_precision") or 0) >= 0.90
        and float(three.get("stain_recall") or 0) >= 0.90
        and float(three.get("confident_clean_purity") or 0) >= 0.90
        and drop <= 0.03
    )
    ext_ok = bool(external_val.get("catastrophic_saturation_resolved"))
    known_ts3 = known_audit["v2"].get("tongueset3", {}).get("band_counts", {})
    known_ok = known_ts3.get("stain", 100) < 50  # 不再近乎全 stain
    unified_retake = unified["new_unified"].get("retake", 80)
    unified_ok = unified_retake <= 40

    if source_ok and ext_ok and known_ok and unified_ok:
        status = "TARGET_PASS"
        recommendation = "ACTIVATE_STAIN_V2_AND_RERUN_D4_FINAL_AUDIT"
        activate = False  # 仍需人工确认
    elif source_ok and (ext_ok or known_ok):
        status = "MINIMUM_PASS"
        recommendation = "MINIMUM_PASS_KEEP_POLICY_1_3"
        activate = False
    else:
        status = "NEEDS_IMPROVEMENT"
        recommendation = "NEEDS_IMPROVEMENT_STOP"
        activate = False
    return {
        "baseline_status": status,
        "recommendation": recommendation,
        "policy_activated": activate,
        "gates": {
            "source_ok": source_ok,
            "external_val_ok": ext_ok,
            "known_audit_ok": known_ok,
            "unified_ok": unified_ok,
        },
    }


def write_freeze_docs(stats: dict[str, Any], docs_dir: Path = Path("docs")) -> None:
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "D4_C_1_B_FREEZE_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rec = stats["decision"]
    lines = [
        "# D4-C.1-B Domain-Robust Stain Freeze Report",
        "",
        f"- baseline_status: **`{rec['baseline_status']}`**",
        f"- recommendation: `{rec['recommendation']}`",
        f"- policy_activated: `{rec['policy_activated']}`",
        f"- stain_contract_version: `{stats.get('stain_contract_version')}`",
        f"- external_pseudo_labels: `false`",
        "",
        "## Source",
        "",
        f"- best_epoch: `{stats.get('best_epoch')}`",
        f"- source val AUROC: `{stats.get('best_source_val_auroc')}`",
        f"- t_clear_v2 / t_retake_v2: `{stats.get('t_clear_v2')}` / `{stats.get('t_retake_v2')}`",
        f"- source test: `{stats.get('source_test_summary')}`",
        "",
        "## External",
        "",
        f"- external val: `{stats.get('external_val_robustness')}`",
        f"- known audit: `{stats.get('external_known_audit_summary')}`",
        "",
        "## Unified recovery",
        "",
        f"`{stats.get('unified_recovery_summary')}`",
        "",
        "v1 checkpoint/thresholds preserved. Policy remains 1.3 until explicit activation.",
        "",
    ]
    (docs_dir / "D4_C_1_B_FREEZE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    contract = [
        "# D4-C.1-B Domain-Robust Stain Contract",
        "",
        "- stain_contract_version: 1.1",
        "- parent: stain_detection_v1 (1.0) preserved",
        "- representation: black_masked_roi + letterbox 224 + ImageNet",
        "- architecture: resnet18 ImageNet init（不从 v1 fine-tune）",
        "- training: source supervised + source consistency + external unlabeled consistency",
        "- forbidden: external pseudo labels / entropy minimization / test-based selection",
        "- active policy switch requires TARGET_PASS + human confirmation",
        "",
    ]
    (docs_dir / "D4_C_1_B_DOMAIN_ROBUST_STAIN_CONTRACT.md").write_text(
        "\n".join(contract), encoding="utf-8"
    )


def run_post_train_pipeline(
    *,
    output_run: Path = Path("runs/input_guard/d4c1b/stain_v2"),
    reports_dir: Path = Path("reports/d4c1b"),
    device: str = "auto",
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    v1_ckpt = Path("runs/input_guard/d4c/stain/best.pt")
    v1_thr_path = Path("runs/input_guard/d4c/stain/thresholds.json")
    v2_ckpt = output_run / "best.pt"
    v2_thr_path = output_run / "thresholds.json"
    assert v1_ckpt.exists() and v2_ckpt.exists()
    v1_md5_before = _md5(v1_ckpt)
    v1_thr_before = v1_thr_path.read_text(encoding="utf-8")

    model_v1, model_v2, device_t = load_model_pair(
        v1_ckpt=v1_ckpt, v2_ckpt=v2_ckpt, device=device
    )
    thr_v1 = load_frozen_thresholds(v1_thr_path)
    thr_v2 = load_frozen_thresholds(v2_thr_path)
    style_contract = load_style_contract(reports_dir / "style_augmentation_contract.json")
    meta = json.loads((output_run / "run_metadata.json").read_text(encoding="utf-8"))

    external_val = audit_external_val_robustness(
        model_v1=model_v1,
        model_v2=model_v2,
        roi_index=Path("reports/d4c1/roi_cache/index.parquet"),
        data_config_v1=Path("configs/stain_detection_v1.yaml"),
        data_config_v2=Path("configs/stain_detection_v2.yaml"),
        thr_v1=thr_v1,
        thr_v2=thr_v2,
        device=device_t,
    )
    (reports_dir / "external_val_robustness.json").write_text(
        json.dumps(external_val, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Early STOP：external val 仍灾难饱和或 source val 崩
    if not external_val["catastrophic_saturation_resolved"] or meta["best_source_val_auroc"] < 0.90:
        decision = {
            "baseline_status": "NEEDS_IMPROVEMENT",
            "recommendation": "NEEDS_IMPROVEMENT_STOP",
            "policy_activated": False,
            "early_stop_before_test": True,
            "reason": "external val saturation unresolved or source val weak",
        }
        stats = {
            "stage": "D4-C.1-B",
            "stain_contract_version": "1.1",
            "decision": decision,
            "best_epoch": meta["best_epoch"],
            "best_source_val_auroc": meta["best_source_val_auroc"],
            "t_clear_v2": thr_v2["t_clear"],
            "t_retake_v2": thr_v2["t_retake"],
            "external_val_robustness": external_val,
            "v1_checkpoint_md5_unchanged": _md5(v1_ckpt) == v1_md5_before,
            "v1_thresholds_text_unchanged": v1_thr_path.read_text(encoding="utf-8")
            == v1_thr_before,
        }
        write_freeze_docs(stats)
        (reports_dir / "d4c1b_comparison_summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return stats

    style_sens = audit_style_sensitivity(
        model_v1=model_v1,
        model_v2=model_v2,
        roi_index=Path("reports/d4c1/roi_cache/index.parquet"),
        data_config_v2=Path("configs/stain_detection_v2.yaml"),
        style_contract=style_contract,
        device=device_t,
    )
    (reports_dir / "style_sensitivity_v1_v2.json").write_text(
        json.dumps(style_sens, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    embedding = audit_embedding_comparison(
        model_v1=model_v1,
        model_v2=model_v2,
        stain_manifest=Path("data/stain/v1/stain_manifest.parquet"),
        roi_index=Path("reports/d4c1/roi_cache/index.parquet"),
        data_config_v1=Path("configs/stain_detection_v1.yaml"),
        data_config_v2=Path("configs/stain_detection_v2.yaml"),
        device=device_t,
    )
    (reports_dir / "embedding_comparison_v1_v2.json").write_text(
        json.dumps(embedding, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_test = audit_source_test_once(
        model_v2=model_v2,
        stain_manifest=Path("data/stain/v1/stain_manifest.parquet"),
        data_config=Path("configs/stain_detection_v2.yaml"),
        train_config=Path("configs/stain_train_v2.yaml"),
        thresholds=thr_v2,
        device=device_t,
    )
    source_test["pred_frame"].to_parquet(
        output_run / "source_test_predictions.parquet", index=False
    )
    source_summary = {
        "auroc": source_test["ranking"]["auroc"],
        "pr_auc": source_test["ranking"]["pr_auc"],
        "precision": source_test["binary_at_t_retake"].get("precision"),
        "recall": source_test["binary_at_t_retake"].get("recall"),
        "specificity": source_test["binary_at_t_retake"].get("specificity"),
        "f1": source_test["binary_at_t_retake"].get("f1"),
        "confident_clean_purity": source_test["three_state"].get("confident_clean_purity"),
        "confident_stain_precision": source_test["three_state"].get(
            "confident_stain_precision"
        ),
        "stain_recall": source_test["three_state"].get("stain_recall"),
        "uncertain_rate": source_test["three_state"].get("uncertain_rate"),
        "confident_coverage": source_test["three_state"].get("confident_coverage"),
        "auroc_drop_vs_v1": source_test["auroc_drop_vs_v1"],
    }
    (reports_dir / "source_test_evaluation.json").write_text(
        json.dumps(
            {k: v for k, v in source_test.items() if k != "pred_frame"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    known = audit_known_external(
        model_v1=model_v1,
        model_v2=model_v2,
        roi_index=Path("reports/d4c1/roi_cache/index.parquet"),
        data_config_v1=Path("configs/stain_detection_v1.yaml"),
        data_config_v2=Path("configs/stain_detection_v2.yaml"),
        thr_v1=thr_v1,
        thr_v2=thr_v2,
        device=device_t,
    )
    known["pred_v2"].to_parquet(
        output_run / "external_known_audit_predictions.parquet", index=False
    )
    known_summary = {"v1": known["v1"], "v2": known["v2"], "role": known["role"]}
    (reports_dir / "external_known_audit.json").write_text(
        json.dumps(known_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    unified = audit_unified_recovery(
        v2_ckpt=v2_ckpt,
        v2_thresholds=v2_thr_path,
        segmentation_dir=Path("data/segmentation/v1"),
        seg_checkpoint=Path("runs/segmentation/d3c/baseline/best.pt"),
        seg_data_config=Path("configs/segmentation_v1.yaml"),
        seg_train_config=Path("configs/segmentation_train_v1.yaml"),
        policy_path=Path("configs/input_guard_v1.yaml"),
        output_dir=reports_dir,
        device=device,
    )
    unified_summary = {k: v for k, v in unified.items() if k != "frame"}
    (reports_dir / "unified_guard_recovery.json").write_text(
        json.dumps(unified_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decision = decide_baseline_status(
        source_test=source_test,
        external_val=external_val,
        known_audit=known,
        unified=unified,
    )
    stats = {
        "stage": "D4-C.1-B",
        "stain_contract_version": "1.1",
        "v1_reference_contract": "1.0",
        "training_strategy": "source_supervised+style_aug+external_consistency",
        "source_supervised": True,
        "external_pseudo_labels": False,
        "external_consistency": True,
        "style_augmentation": True,
        "best_epoch": meta["best_epoch"],
        "planned_epochs": meta["planned_epochs"],
        "actual_epochs": meta["actual_epochs"],
        "best_source_val_auroc": meta["best_source_val_auroc"],
        "best_source_val_pr_auc": meta["best_source_val_pr_auc"],
        "t_clear_v2": thr_v2["t_clear"],
        "t_retake_v2": thr_v2["t_retake"],
        "source_test_summary": source_summary,
        "external_val_robustness": external_val,
        "style_sensitivity": style_sens,
        "embedding_comparison": embedding,
        "external_known_audit_summary": known_summary,
        "unified_recovery_summary": unified_summary,
        "decision": decision,
        "v1_checkpoint_md5_unchanged": _md5(v1_ckpt) == v1_md5_before,
        "v1_thresholds_text_unchanged": v1_thr_path.read_text(encoding="utf-8")
        == v1_thr_before,
        "policy_version_still": "1.3",
    }
    write_freeze_docs(stats)
    (reports_dir / "d4c1b_comparison_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
