"""D4-D synthetic cast/occlusion：仅工程验证，不进正式数据。"""
from __future__ import annotations

from typing import Any

import numpy as np

from .color_cast import apply_channel_cast, compute_color_cast_features, evaluate_color_cast
from .occlusion import apply_synthetic_occlusion, compute_occlusion_features, evaluate_occlusion
from .policy import InputGuardPolicy


def select_fixed_sample_indices(total: int, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    count = min(int(count), int(total))
    if count <= 0:
        return []
    return sorted(rng.choice(total, size=count, replace=False).tolist())


def run_color_cast_synthetic_audit(
    rows: list[dict[str, Any]],
    *,
    policy: InputGuardPolicy,
    d4d_cfg: dict[str, Any],
) -> dict[str, Any]:
    synth_cfg = d4d_cfg.get("synthetic", {}).get("color_cast", {})
    seed = int(d4d_cfg.get("seed", 20260813))
    directions = list(synth_cfg.get("directions", ["red", "green", "blue", "yellow"]))
    moderate_gain = float(synth_cfg.get("moderate_gain", 1.25))
    severe_gain = float(synth_cfg.get("severe_gain", 1.55))
    count = int(synth_cfg.get("sample_count", 24))
    indices = select_fixed_sample_indices(len(rows), count, seed)

    results: dict[str, Any] = {
        "seed": seed,
        "sample_indices": indices,
        "directions": {},
        "records": [],
    }
    for direction in directions:
        results["directions"][direction] = {
            "moderate_warning_or_retake": 0,
            "severe_retake": 0,
            "n": 0,
        }

    neutral_cfg = d4d_cfg.get("color_cast", {}).get("neutral", {})
    for index in indices:
        row = rows[index]
        rgb = row["rgb"]
        mask = row.get("mask")
        sample_id = row.get("sample_id")
        base_feat = compute_color_cast_features(rgb, mask, neutral_cfg=neutral_cfg)
        for direction in directions:
            for severity, gain in (("moderate", moderate_gain), ("severe", severe_gain)):
                casted = apply_channel_cast(rgb, direction=direction, gain=gain)
                feat = compute_color_cast_features(
                    casted, mask, neutral_cfg=neutral_cfg
                )
                check = evaluate_color_cast(
                    casted, mask, policy, d4d_cfg=d4d_cfg
                )
                bucket = results["directions"][direction]
                bucket["n"] += 1 if severity == "severe" else 0
                if severity == "severe":
                    if check.decision_effect == "retake":
                        bucket["severe_retake"] += 1
                    if check.decision_effect in {"warning", "retake"}:
                        bucket["severe_detected"] = (
                            bucket.get("severe_detected", 0) + 1
                        )
                else:
                    if check.decision_effect in {"warning", "retake"}:
                        bucket["moderate_warning_or_retake"] += 1
                results["records"].append(
                    {
                        "sample_id": sample_id,
                        "direction": direction,
                        "severity": severity,
                        "gain": gain,
                        "base_cast_magnitude": base_feat.get(
                            "estimated_cast_magnitude"
                        ),
                        "cast_magnitude": feat.get("estimated_cast_magnitude"),
                        "finding": check.finding,
                        "decision_effect": check.decision_effect,
                        "evaluation_state": check.evaluation_state,
                    }
                )

    # 汇总 detection rates
    severe_total = 0
    severe_hit = 0
    moderate_total = 0
    moderate_hit = 0
    for direction, bucket in results["directions"].items():
        # recount properly from records
        pass
    severe_retake_hit = 0
    for record in results["records"]:
        if record["severity"] == "severe":
            severe_total += 1
            # detection = WARNING 或 RETAKE（与 occlusion synthetic gate 一致）
            if record["decision_effect"] in {"warning", "retake"}:
                severe_hit += 1
            if record["decision_effect"] == "retake":
                severe_retake_hit += 1
        else:
            moderate_total += 1
            if record["decision_effect"] in {"warning", "retake"}:
                moderate_hit += 1
    results["severe_detection_rate"] = (
        float(severe_hit / severe_total) if severe_total else None
    )
    results["severe_retake_rate"] = (
        float(severe_retake_hit / severe_total) if severe_total else None
    )
    results["moderate_detection_rate"] = (
        float(moderate_hit / moderate_total) if moderate_total else None
    )
    return results


def run_occlusion_synthetic_audit(
    rows: list[dict[str, Any]],
    *,
    policy: InputGuardPolicy,
    d4d_cfg: dict[str, Any],
) -> dict[str, Any]:
    synth_cfg = d4d_cfg.get("synthetic", {}).get("occlusion", {})
    seed = int(d4d_cfg.get("seed", 20260813))
    count = int(synth_cfg.get("sample_count", 24))
    indices = select_fixed_sample_indices(len(rows), count, seed)
    area_map = {
        "small": float(synth_cfg.get("small_area_ratio", 0.04)),
        "moderate": float(synth_cfg.get("moderate_area_ratio", 0.12)),
        "severe": float(synth_cfg.get("severe_area_ratio", 0.28)),
    }
    summary = {
        level: {"n": 0, "retake": 0, "warning": 0, "pass": 0, "unavailable": 0}
        for level in area_map
    }
    records = []
    for offset, index in enumerate(indices):
        row = rows[index]
        rgb = row["rgb"]
        mask = row["mask"]
        prob = row["probability"]
        for level, area in area_map.items():
            rgb_occ, occ_mask = apply_synthetic_occlusion(
                rgb,
                mask,
                area_ratio=area,
                mode="bright",
                seed=seed + offset * 17 + hash(level) % 97,
            )
            # 压低遮挡区 probability，模拟 D3 对遮挡的不确定
            prob_occ = np.asarray(prob, dtype=np.float64).copy()
            prob_occ[occ_mask] = np.minimum(prob_occ[occ_mask], 0.15)
            check = evaluate_occlusion(
                rgb_occ, mask, prob_occ, policy, d4d_cfg=d4d_cfg
            )
            bucket = summary[level]
            bucket["n"] += 1
            if check.evaluation_state != "evaluated":
                bucket["unavailable"] += 1
            elif check.decision_effect == "retake":
                bucket["retake"] += 1
            elif check.decision_effect == "warning":
                bucket["warning"] += 1
            else:
                bucket["pass"] += 1
            records.append(
                {
                    "sample_id": row.get("sample_id"),
                    "level": level,
                    "area_ratio": area,
                    "finding": check.finding,
                    "decision_effect": check.decision_effect,
                    "evaluation_state": check.evaluation_state,
                    "score": check.score,
                }
            )
    severe_n = summary["severe"]["n"]
    severe_det = summary["severe"]["retake"] + summary["severe"]["warning"]
    return {
        "seed": seed,
        "sample_indices": indices,
        "summary": summary,
        "severe_detection_rate": float(severe_det / severe_n) if severe_n else None,
        "severe_retake_rate": (
            float(summary["severe"]["retake"] / severe_n) if severe_n else None
        ),
        "small_retake_rate": (
            float(summary["small"]["retake"] / summary["small"]["n"])
            if summary["small"]["n"]
            else None
        ),
        "records": records,
    }
