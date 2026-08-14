"""D4-C.1-C：dual-gate audit / candidate ranking / experiment report。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import StainDataConfig, StainTrainConfig
from .d4c1a_model_tools import centroid_distances, extract_embedding
from .robust_train import _external_summary, evaluate_external_roi, evaluate_source_split
from .style_augment import apply_style_transform, load_style_contract, sample_style_params
from .train import load_stain_checkpoint, resolve_device
from .transforms import preprocess_masked_roi
from .v3_train import load_v3_checkpoint


V2_BASELINE_GAP = 10.652472496032715
V2_BASELINE_HIGHSCORE = 0.74


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _predict_logit(model, batch: torch.Tensor) -> float:
    with torch.inference_mode():
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
        return float(out.reshape(-1)[0].item())


def audit_candidate_external_val(
    *,
    model,
    roi_index: Path,
    data_config: Path,
    device,
    t_clear: float = 0.95,
    t_retake: float = 0.96,
) -> dict[str, Any]:
    pred = evaluate_external_roi(
        model, roi_index=roi_index, data_config=data_config, split="val", device=device
    )
    summary = _external_summary(pred, t_clear, t_retake)
    gap = abs(float(summary.get("domain_gap_median_logit", 0.0)))
    reduction = None if V2_BASELINE_GAP <= 1e-8 else float(1.0 - gap / V2_BASELINE_GAP)
    ts3_high = float(summary.get("tongueset3", {}).get("highscore_rate", 1.0))
    return {
        "summary": summary,
        "domain_logit_gap": gap,
        "gap_reduction_vs_v2": reduction,
        "tongueset3_highscore_rate": ts3_high,
        "biohit_val_median_p": summary.get("biohit", {}).get("median_p"),
        "biohit_val_median_logit": summary.get("biohit", {}).get("median_logit"),
        "tongueset3_val_median_p": summary.get("tongueset3", {}).get("median_p"),
        "tongueset3_val_median_logit": summary.get("tongueset3", {}).get("median_logit"),
    }


def audit_style_sensitivity_model(
    *,
    model,
    roi_index: Path,
    data_config: Path,
    style_contract: dict,
    device,
    n_per_domain: int = 30,
) -> dict[str, Any]:
    data_cfg = StainDataConfig(data_config)
    frame = pd.read_parquet(roi_index)
    frame = frame[(frame.split == "val") & frame.roi_rgb_path.notna()]
    rng = np.random.default_rng(20260814)
    deltas = []
    for dataset_name in ("biohit", "tongueset3"):
        subset = frame[frame.dataset == dataset_name].sort_values("sample_id").head(n_per_domain)
        for _index, row in subset.iterrows():
            rgb = np.asarray(Image.open(row.roi_rgb_path).convert("RGB"), dtype=np.uint8)
            mask = (np.asarray(Image.open(row.roi_mask_path)) > 0).astype(np.uint8)
            base = preprocess_masked_roi(rgb, mask, data_cfg, split="val")
            batch0 = torch.from_numpy(np.ascontiguousarray(base)).unsqueeze(0).to(device)
            logit0 = _predict_logit(model, batch0)
            sample_deltas = []
            for strength_name in ("red_wb", "blue_wb", "gamma", "exposure"):
                params = sample_style_params(style_contract, rng, strength="moderate")
                if strength_name == "red_wb":
                    params.update(
                        {
                            "gain_r": style_contract["channel_gain_ranges"]["r"][1],
                            "gain_g": 1.0,
                            "gain_b": 1.0,
                        }
                    )
                if strength_name == "blue_wb":
                    params.update(
                        {
                            "gain_r": 1.0,
                            "gain_g": 1.0,
                            "gain_b": style_contract["channel_gain_ranges"]["b"][1],
                        }
                    )
                styled, _ = apply_style_transform(
                    rgb, style_contract, rng, strength="moderate", params=params
                )
                tensor = preprocess_masked_roi(styled, mask, data_cfg, split="val")
                batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
                sample_deltas.append(abs(_predict_logit(model, batch) - logit0))
            deltas.append(float(np.mean(sample_deltas)))
    return {
        "n": len(deltas),
        "median_abs_delta_logit": float(np.median(deltas)),
        "mean_abs_delta_logit": float(np.mean(deltas)),
    }


def _embedding_from_model(model, batch: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        if hasattr(model, "extract_embedding"):
            emb = model.extract_embedding(batch, domain_ids=None)
            return emb.detach().cpu().numpy()[0]
        return extract_embedding(model, batch)


def audit_embedding_domain_probe(
    *,
    model,
    stain_manifest: Path,
    roi_index: Path,
    data_config: Path,
    device,
    max_per_group: int = 120,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data_cfg = StainDataConfig(data_config)
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
    emb = {}
    for group_name, frame in groups.items():
        vectors = []
        for _index, row in frame.head(max_per_group).iterrows():
            rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.uint8)
            mask = (np.asarray(Image.open(row["roi_mask_path"])) > 0).astype(np.uint8)
            tensor = preprocess_masked_roi(rgb, mask, data_cfg, split="val")
            batch = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)
            vectors.append(_embedding_from_model(model, batch))
        emb[group_name] = np.stack(vectors) if vectors else np.zeros((0, 512))

    xs = []
    ys = []
    for key in ("stain_pos", "stain_neg", "biohit", "tongueset3"):
        arr = emb[key]
        if len(arr) == 0:
            continue
        domain = "stained" if key.startswith("stain") else key
        xs.append(arr)
        ys.extend([domain] * len(arr))
    matrix = np.concatenate(xs, axis=0)
    labels = np.array(ys)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=20260814)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260814)
    scores = cross_val_score(pipe, matrix, labels, cv=cv)

    # stain class separation（pos vs neg）
    stain_sep = None
    if len(emb["stain_pos"]) and len(emb["stain_neg"]):
        pos_c = emb["stain_pos"].mean(axis=0)
        neg_c = emb["stain_neg"].mean(axis=0)
        stain_sep = float(np.linalg.norm(pos_c - neg_c))
    return {
        "domain_probe_cv_acc": float(scores.mean()),
        "domain_probe_cv_std": float(scores.std()),
        "centroid_distances": centroid_distances(emb),
        "stain_class_centroid_distance": stain_sep,
        "n_per_group": {key: int(len(arr)) for key, arr in emb.items()},
    }


def audit_source_confounding(
    *,
    stain_manifest: Path,
    max_samples: int = 400,
) -> dict[str, Any]:
    """Source positive vs negative：resolution / luminance / color 粗审计。"""
    frame = pd.read_parquet(stain_manifest)
    frame = frame[(frame.eligible == True) & (frame.split == "train")].copy()
    pos = frame[frame.label == 1].sort_values("sample_id").head(max_samples // 2)
    neg = frame[frame.label == 0].sort_values("sample_id").head(max_samples // 2)

    def stats(subset: pd.DataFrame) -> dict[str, float]:
        widths = []
        heights = []
        luminances = []
        red_means = []
        blue_means = []
        for _index, row in subset.iterrows():
            rgb = np.asarray(Image.open(row["roi_rgb_path"]).convert("RGB"), dtype=np.float32)
            mask = (np.asarray(Image.open(row["roi_mask_path"])) > 0)
            heights.append(rgb.shape[0])
            widths.append(rgb.shape[1])
            if mask.any():
                pixels = rgb[mask]
                luminances.append(float(pixels.mean()))
                red_means.append(float(pixels[:, 0].mean()))
                blue_means.append(float(pixels[:, 2].mean()))
        return {
            "median_h": float(np.median(heights)),
            "median_w": float(np.median(widths)),
            "median_luminance": float(np.median(luminances)),
            "median_red": float(np.median(red_means)),
            "median_blue": float(np.median(blue_means)),
        }

    pos_s = stats(pos)
    neg_s = stats(neg)
    # 粗 proxy：若多个通道差异极大，怀疑 class↔acquisition confounding
    lum_gap = abs(pos_s["median_luminance"] - neg_s["median_luminance"])
    red_gap = abs(pos_s["median_red"] - neg_s["median_red"])
    res_gap = abs(pos_s["median_h"] * pos_s["median_w"] - neg_s["median_h"] * neg_s["median_w"])
    suspected = bool(lum_gap > 25 or red_gap > 20 or res_gap > 50000)
    return {
        "positive": pos_s,
        "negative": neg_s,
        "luminance_gap": lum_gap,
        "red_gap": red_gap,
        "resolution_area_gap": res_gap,
        "source_confounding_suspected": suspected,
        "note": "audit only; not used for training",
    }


def evaluate_candidate_acceptance(
    *,
    source_val_auroc: float,
    source_val_pr_auc: float,
    gap_reduction_vs_v2: float | None,
    tongueset3_highscore_rate: float,
    domain_probe_delta_vs_v2: float | None,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """预注册 acceptance：deterministic。"""
    source_gate = gates.get("source", {})
    domain_gate = gates.get("domain_robustness", {})
    auroc_ok = source_val_auroc >= float(source_gate.get("val_auroc_min", 0.95))
    prauc_ok = source_val_pr_auc >= float(source_gate.get("val_pr_auc_min", 0.95))
    gap_ok = (
        gap_reduction_vs_v2 is not None
        and gap_reduction_vs_v2 >= float(domain_gate.get("min_gap_reduction", 0.40))
    )
    high_ok = tongueset3_highscore_rate < float(
        domain_gate.get("max_tongueset3_highscore", 0.50)
    )
    probe_ok = (
        domain_probe_delta_vs_v2 is not None
        and domain_probe_delta_vs_v2 >= float(domain_gate.get("min_domain_probe_drop", 0.10))
    )
    passed = bool(auroc_ok and prauc_ok and gap_ok and high_ok and probe_ok)
    target = bool(
        passed
        and source_val_auroc >= float(source_gate.get("target_val_auroc", 0.97))
        and (gap_reduction_vs_v2 or 0) >= float(domain_gate.get("target_gap_reduction", 0.50))
        and tongueset3_highscore_rate < float(
            domain_gate.get("target_tongueset3_highscore", 0.30)
        )
        and (domain_probe_delta_vs_v2 or 0)
        >= float(domain_gate.get("target_domain_probe_drop", 0.20))
    )
    return {
        "source_auroc_ok": auroc_ok,
        "source_prauc_ok": prauc_ok,
        "gap_ok": gap_ok,
        "highscore_ok": high_ok,
        "probe_ok": probe_ok,
        "candidate_pass": passed,
        "candidate_target": target,
        "status": (
            "TARGET_PASS"
            if target
            else ("MINIMUM_PASS" if passed else "FAIL")
        ),
    }


def meaningful_robustness_signal(
    *,
    gap_reduction_vs_v2: float | None,
    tongueset3_highscore_rate: float,
    train_doc: dict[str, Any],
) -> bool:
    """C3 执行门槛：C1/C2 至少一项有 meaningful signal。"""
    cfg = train_doc.get("c3_execution", {})
    min_gap = float(cfg.get("min_gap_reduction_signal", 0.15))
    max_high = float(cfg.get("max_highscore_to_count_as_signal", 0.65))
    gap_signal = gap_reduction_vs_v2 is not None and gap_reduction_vs_v2 >= min_gap
    high_signal = tongueset3_highscore_rate < max_high
    return bool(gap_signal or high_signal)


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅 PASS 候选参与排序；优先 gap → highscore → probe → style。"""
    eligible = [item for item in candidates if item.get("candidate_pass")]
    eligible.sort(
        key=lambda item: (
            -float(item.get("gap_reduction_vs_v2") or -1.0),
            float(item.get("tongueset3_highscore_rate") or 1.0),
            -float(item.get("domain_probe_delta_vs_v2") or -1.0),
            float(item.get("style_sensitivity_median") or 1e9),
            -float(item.get("source_val_auroc") or 0.0),
        )
    )
    return eligible


def run_candidate_full_audit(
    *,
    candidate: str,
    ckpt_path: Path,
    stain_manifest: Path,
    roi_index: Path,
    data_config: Path,
    train_config: Path,
    style_contract_path: Path,
    reports_dir: Path,
    device: str = "auto",
    v2_domain_probe: float | None = None,
    v2_style_sensitivity: float | None = None,
) -> dict[str, Any]:
    train_cfg = StainTrainConfig(train_config)
    device_t = resolve_device(device)
    model, ckpt = load_v3_checkpoint(
        ckpt_path,
        candidate=candidate,
        train_config=train_config,
        map_location=device_t,
        strict_hash=True,
    )
    model = model.to(device_t).eval()
    model.set_mixstyle_enabled(False)

    val_pred, val_metrics = evaluate_source_split(
        model,
        stain_manifest=stain_manifest,
        data_config=data_config,
        train_config=train_config,
        split="val",
        device=device_t,
    )
    ext = audit_candidate_external_val(
        model=model,
        roi_index=roi_index,
        data_config=data_config,
        device=device_t,
    )
    style_contract = load_style_contract(style_contract_path)
    style = audit_style_sensitivity_model(
        model=model,
        roi_index=roi_index,
        data_config=data_config,
        style_contract=style_contract,
        device=device_t,
    )
    emb = audit_embedding_domain_probe(
        model=model,
        stain_manifest=stain_manifest,
        roi_index=roi_index,
        data_config=data_config,
        device=device_t,
    )
    probe_delta = None
    if v2_domain_probe is not None:
        probe_delta = float(v2_domain_probe - emb["domain_probe_cv_acc"])
    style_reduction = None
    if v2_style_sensitivity is not None and v2_style_sensitivity > 1e-8:
        style_reduction = float(
            1.0 - style["median_abs_delta_logit"] / v2_style_sensitivity
        )

    acceptance = evaluate_candidate_acceptance(
        source_val_auroc=float(val_metrics["auroc"]),
        source_val_pr_auc=float(val_metrics["pr_auc"]),
        gap_reduction_vs_v2=ext["gap_reduction_vs_v2"],
        tongueset3_highscore_rate=ext["tongueset3_highscore_rate"],
        domain_probe_delta_vs_v2=probe_delta,
        gates=train_cfg.gates,
    )
    report = {
        "candidate": candidate,
        "best_epoch": ckpt.get("epoch"),
        "checkpoint_md5": _md5(Path(ckpt_path)),
        "source_val_auroc": float(val_metrics["auroc"]),
        "source_val_pr_auc": float(val_metrics["pr_auc"]),
        "external_val": ext,
        "gap_reduction_vs_v2": ext["gap_reduction_vs_v2"],
        "tongueset3_highscore_rate": ext["tongueset3_highscore_rate"],
        "embedding_probe": emb,
        "domain_probe_accuracy": emb["domain_probe_cv_acc"],
        "domain_probe_delta_vs_v2": probe_delta,
        "style_sensitivity": style,
        "style_sensitivity_median": style["median_abs_delta_logit"],
        "style_sensitivity_reduction_vs_v2": style_reduction,
        "stain_class_separation": emb.get("stain_class_centroid_distance"),
        "candidate_pass": acceptance["candidate_pass"],
        "acceptance": acceptance,
        "meaningful_signal": meaningful_robustness_signal(
            gap_reduction_vs_v2=ext["gap_reduction_vs_v2"],
            tongueset3_highscore_rate=ext["tongueset3_highscore_rate"],
            train_doc=train_cfg.doc,
        ),
        "selection_rule": "source_val_auroc_for_ckpt__dual_gate_for_acceptance",
        "tests_accessed": False,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"candidate_{candidate}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 同步到 run dir
    run_dir = Path(ckpt_path).parent
    (run_dir / "external_val_audit.json").write_text(
        json.dumps(ext, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "embedding_probe.json").write_text(
        json.dumps(emb, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "style_sensitivity.json").write_text(
        json.dumps(style, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def write_experiment_report(
    *,
    docs_dir: Path,
    reports_dir: Path,
    candidate_reports: list[dict[str, Any]],
    source_confounding: dict[str, Any],
    final_status: str,
    selected_candidate: str | None,
    policy_activated: bool,
    pytest_result: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    reports_dir = Path(reports_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "stage": "D4-C.1-C",
        "v2_baseline_gap": V2_BASELINE_GAP,
        "v2_baseline_highscore": V2_BASELINE_HIGHSCORE,
        "candidates": candidate_reports,
        "selected_candidate": selected_candidate,
        "final_status": final_status,
        "policy_activated": policy_activated,
        "source_confounding": source_confounding,
        "pytest_result": pytest_result,
    }
    if extra:
        comparison.update(extra)
    (reports_dir / "candidate_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tradeoff = []
    for item in candidate_reports:
        tradeoff.append(
            {
                "candidate": item["candidate"],
                "source_val_auroc": item.get("source_val_auroc"),
                "domain_logit_gap": item.get("external_val", {}).get("domain_logit_gap"),
                "gap_reduction_vs_v2": item.get("gap_reduction_vs_v2"),
                "tongueset3_highscore_rate": item.get("tongueset3_highscore_rate"),
                "domain_probe_accuracy": item.get("domain_probe_accuracy"),
                "style_sensitivity_median": item.get("style_sensitivity_median"),
                "stain_class_separation": item.get("stain_class_separation"),
                "status": item.get("acceptance", {}).get("status"),
            }
        )
    (reports_dir / "representation_tradeoff.json").write_text(
        json.dumps(tradeoff, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "source_confounding_audit.json").write_text(
        json.dumps(source_confounding, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# D4-C.1-C Experiment Report",
        "",
        f"- stage: D4-C.1-C",
        f"- final_status: `{final_status}`",
        f"- selected_candidate: `{selected_candidate}`",
        f"- policy_activated: `{policy_activated}`",
        f"- pytest: `{pytest_result}`",
        f"- source_confounding_suspected: `{source_confounding.get('source_confounding_suspected')}`",
        "",
        "## Candidates",
        "",
    ]
    for item in candidate_reports:
        lines.append(
            f"- **{item['candidate']}**: AUROC={item.get('source_val_auroc')}, "
            f"gap_red={item.get('gap_reduction_vs_v2')}, "
            f"ts3_high={item.get('tongueset3_highscore_rate')}, "
            f"status={item.get('acceptance', {}).get('status')}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- v1/v2 checkpoints and thresholds preserved.",
            "- No external stain pseudo-labels.",
            "- No policy switch unless all final gates PASS and user confirms.",
            "- Source/external TEST not accessed unless a candidate PASSes acceptance.",
            "",
        ]
    )
    (docs_dir / "D4_C_1_C_EXPERIMENT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparison


def load_baseline_models_for_probe(
    *,
    device: str = "auto",
):
    """加载冻结 v1/v2 仅用于对照 probe（不训练）。"""
    device_t = resolve_device(device)
    model_v1, _ = load_stain_checkpoint(
        "runs/input_guard/d4c/stain/best.pt",
        train_config=StainTrainConfig("configs/stain_train_v1.yaml"),
        data_config=StainDataConfig("configs/stain_detection_v1.yaml"),
        map_location=device_t,
        strict=True,
    )
    # v2 训练后若 train yaml 有文档字段微调，hash 可能变化；对照 probe 允许非严格 hash
    model_v2, _ = load_stain_checkpoint(
        "runs/input_guard/d4c1b/stain_v2/best.pt",
        train_config=StainTrainConfig("configs/stain_train_v2.yaml"),
        data_config=StainDataConfig("configs/stain_detection_v2.yaml"),
        map_location=device_t,
        strict=False,
    )
    return model_v1.to(device_t).eval(), model_v2.to(device_t).eval(), device_t
