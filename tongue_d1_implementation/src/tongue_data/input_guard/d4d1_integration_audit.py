"""D4-D.1：只读 Unified Guard integration audit（不改 threshold / runtime）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .decision import aggregate_decision, select_primary_reason
from .ontology import Decision, EvaluationState
from .policy import load_input_guard_policy
from .runtime import InputGuardRuntime
from .signal_checks import IMPLEMENTED_SIGNAL_CHECKS

D4B_CHECK_IDS = frozenset(check.value for check in IMPLEMENTED_SIGNAL_CHECKS)
STAIN_ID = "quality.stain_suspected"
CAST_ID = "quality.color_cast"
OCC_ID = "quality.occlusion"

STAIN_T_CLEAR = 0.95
STAIN_T_RETAKE = 0.96


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            key: None
            for key in (
                "count",
                "min",
                "p01",
                "p05",
                "p10",
                "p25",
                "median",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "mean",
                "std",
            )
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _stain_band(probability: float | None) -> str | None:
    if probability is None or (isinstance(probability, float) and np.isnan(probability)):
        return None
    score = float(probability)
    if score <= STAIN_T_CLEAR:
        return "clear"
    if score >= STAIN_T_RETAKE:
        return "stain"
    return "uncertain"


def _effect_of(check: dict | None) -> str | None:
    if not check:
        return None
    if check.get("evaluation_state") != EvaluationState.EVALUATED.value:
        return None
    return check.get("decision_effect")


def aggregate_from_checks(
    checks: dict[str, dict],
    *,
    include: set[str],
    priority: list[str],
) -> dict[str, Any]:
    effects = []
    reason_codes: list[str] = []
    retake_reasons: set[str] = set()
    warning_reasons: set[str] = set()
    for check_id, check in checks.items():
        if check_id not in include:
            continue
        effect = _effect_of(check)
        if effect is None:
            continue
        effects.append(effect)
        code = check.get("reason_code")
        if code:
            reason_codes.append(code)
            if effect == Decision.RETAKE.value:
                retake_reasons.add(code)
            elif effect == Decision.WARNING.value:
                warning_reasons.add(code)
    decision = aggregate_decision(effects)
    primary = select_primary_reason(
        reason_codes,
        priority=priority,
        retake_reasons=retake_reasons,
        warning_reasons=warning_reasons,
    )
    return {
        "decision": decision.value,
        "reason_codes": reason_codes,
        "primary_reason": primary,
    }


def _source_flags(checks: dict[str, dict]) -> dict[str, bool]:
    d4b = any(
        _effect_of(checks.get(check_id)) == Decision.RETAKE.value
        for check_id in D4B_CHECK_IDS
    )
    stain = _effect_of(checks.get(STAIN_ID)) == Decision.RETAKE.value
    cast = _effect_of(checks.get(CAST_ID)) == Decision.RETAKE.value
    occ = _effect_of(checks.get(OCC_ID)) == Decision.RETAKE.value
    return {
        "retake_due_to_d4b": d4b,
        "retake_due_to_stain": stain,
        "retake_due_to_color_cast": cast,
        "retake_due_to_occlusion": occ,
        "multi_retake_reason": sum([d4b, stain, cast, occ]) >= 2,
    }


def _attribution_bucket(flags: dict[str, bool]) -> str:
    """映射到报告用 by_source key。"""
    d4b = flags["retake_due_to_d4b"]
    stain = flags["retake_due_to_stain"]
    cast = flags["retake_due_to_color_cast"]
    occ = flags["retake_due_to_occlusion"]
    active_count = sum([d4b, stain, cast, occ])
    if active_count == 0:
        return "none"
    if active_count >= 3:
        return "three_or_more"
    if active_count == 1:
        if d4b:
            return "d4b_only"
        if stain:
            return "stain_only"
        if cast:
            return "color_cast_only"
        return "occlusion_only"
    # 恰好两个来源
    if d4b and stain:
        return "d4b_plus_stain"
    if d4b and cast:
        return "d4b_plus_cast"
    if d4b and occ:
        return "d4b_plus_occlusion"
    if stain and cast:
        return "stain_plus_cast"
    if stain and occ:
        return "stain_plus_occlusion"
    if cast and occ:
        return "cast_plus_occlusion"
    return "other"


def collect_sample_audit_rows(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    stain_checkpoint: str | Path,
    stain_thresholds: str | Path,
    device: str = "auto",
) -> pd.DataFrame:
    """对 frozen test 130 跑一次完整 inference（只读 audit，不写 policy）。"""
    from tongue_data.segmentation.inference import load_rgb_image

    manifest = pd.read_parquet(
        Path(segmentation_dir) / "segmentation_manifest.parquet"
    )
    test = manifest[manifest["split"].astype(str) == "test"].copy()
    test = test.sort_values("sample_id").reset_index(drop=True)
    policy = load_input_guard_policy(policy_path)
    runtime = InputGuardRuntime(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        policy_path=policy_path,
        device=device,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
    )
    rows: list[dict[str, Any]] = []
    for _index, row in test.iterrows():
        sample_id = str(row["sample_id"])
        rgb, _mode = load_rgb_image(str(row["image_path"]))
        # 保护：不修改调用方数组契约由 runtime 内部保证
        result = runtime.evaluate(rgb.copy(), sample_id=sample_id)
        checks = {key: check.to_dict() for key, check in result.checks.items()}
        stain = checks.get(STAIN_ID, {})
        cast = checks.get(CAST_ID, {})
        occ = checks.get(OCC_ID, {})
        d4b_abl = aggregate_from_checks(
            checks, include=set(D4B_CHECK_IDS), priority=policy.primary_reason_priority
        )
        flags = _source_flags(checks)
        p_stain = stain.get("score")
        features = result.features or {}
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": str(row["dataset"]),
                "split": "test",
                "image_path": str(row["image_path"]),
                "D3_status": result.segmentation_reference.get("status"),
                "D4B_decision": d4b_abl["decision"],
                "D4B_reason_codes": "|".join(d4b_abl["reason_codes"]),
                "stain_probability": p_stain,
                "stain_finding": stain.get("finding"),
                "stain_decision_effect": stain.get("decision_effect"),
                "stain_evaluation_state": stain.get("evaluation_state"),
                "stain_band": _stain_band(p_stain),
                "color_cast_evaluation_state": cast.get("evaluation_state"),
                "color_cast_finding": cast.get("finding"),
                "color_cast_decision_effect": cast.get("decision_effect"),
                "color_cast_score": cast.get("score"),
                "occlusion_evaluation_state": occ.get("evaluation_state"),
                "occlusion_finding": occ.get("finding"),
                "occlusion_decision_effect": occ.get("decision_effect"),
                "occlusion_score": occ.get("score"),
                "unified_decision": result.decision,
                "primary_reason": result.primary_reason,
                "all_reason_codes": "|".join(result.reason_codes),
                "evaluation_complete": bool(result.evaluation_complete),
                "guard_ready": bool(result.guard_ready),
                "retake_due_to_d4b": flags["retake_due_to_d4b"],
                "retake_due_to_stain": flags["retake_due_to_stain"],
                "retake_due_to_color_cast": flags["retake_due_to_color_cast"],
                "retake_due_to_occlusion": flags["retake_due_to_occlusion"],
                "multi_retake_reason": flags["multi_retake_reason"],
                "attribution_bucket": _attribution_bucket(flags),
                "foreground_ratio": features.get("foreground_ratio"),
                "mean_luminance": features.get("mean_luminance"),
                "roi_blur_score": features.get("roi_blur_score"),
                "effective_short_side_px": features.get("effective_short_side_px"),
                "tight_bbox_width_px": features.get("tight_bbox_width_px"),
                "tight_bbox_height_px": features.get("tight_bbox_height_px"),
                "_checks_json": json.dumps(checks, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_table(frame: pd.DataFrame, policy_path: str | Path) -> dict[str, Any]:
    policy = load_input_guard_policy(policy_path)
    priority = policy.primary_reason_priority
    layers = {
        "A_d4b_only": set(D4B_CHECK_IDS),
        "B_d4b_stain": set(D4B_CHECK_IDS) | {STAIN_ID},
        "C_d4b_stain_cast": set(D4B_CHECK_IDS) | {STAIN_ID, CAST_ID},
        "D_full": set(D4B_CHECK_IDS) | {STAIN_ID, CAST_ID, OCC_ID},
    }
    out: dict[str, Any] = {"n": int(len(frame)), "sample_ids": frame["sample_id"].tolist()}
    counts = {}
    for layer_name, include in layers.items():
        decisions = []
        for _index, row in frame.iterrows():
            checks = json.loads(row["_checks_json"])
            agg = aggregate_from_checks(checks, include=include, priority=priority)
            decisions.append(agg["decision"])
        series = pd.Series(decisions)
        counts[layer_name] = {
            "pass": int((series == "pass").sum()),
            "warning": int((series == "warning").sum()),
            "retake": int((series == "retake").sum()),
        }
    out["counts"] = counts
    out["delta_retake"] = {
        "B_minus_A": counts["B_d4b_stain"]["retake"] - counts["A_d4b_only"]["retake"],
        "C_minus_B": counts["C_d4b_stain_cast"]["retake"]
        - counts["B_d4b_stain"]["retake"],
        "D_minus_C": counts["D_full"]["retake"] - counts["C_d4b_stain_cast"]["retake"],
    }
    # 一致性：D_full 应等于 unified_decision
    mismatch = int((frame["unified_decision"] != frame.apply(
        lambda row: aggregate_from_checks(
            json.loads(row["_checks_json"]),
            include=layers["D_full"],
            priority=priority,
        )["decision"],
        axis=1,
    )).sum()) if len(frame) else 0
    out["full_vs_unified_mismatch"] = mismatch
    return out


def build_retake_attribution(frame: pd.DataFrame) -> dict[str, Any]:
    retakes = frame[frame["unified_decision"] == "retake"].copy()
    report_by_source = {
        "d4b_only": 0,
        "stain_only": 0,
        "color_cast_only": 0,
        "occlusion_only": 0,
        "d4b_plus_stain": 0,
        "d4b_plus_cast": 0,
        "d4b_plus_occlusion": 0,
        "stain_plus_cast": 0,
        "stain_plus_occlusion": 0,
        "cast_plus_occlusion": 0,
        "three_or_more": 0,
        "none": 0,
        "other": 0,
    }
    raw_bucket_counts: dict[str, int] = {}
    for bucket in retakes["attribution_bucket"]:
        raw_bucket_counts[bucket] = raw_bucket_counts.get(bucket, 0) + 1
        if bucket not in report_by_source:
            report_by_source["other"] += 1
        else:
            report_by_source[bucket] += 1

    return {
        "total_retake": int(len(retakes)),
        "by_source": report_by_source,
        "unique_trigger_count": {
            "D4B": int(retakes["retake_due_to_d4b"].sum()) if len(retakes) else 0,
            "stain": int(retakes["retake_due_to_stain"].sum()) if len(retakes) else 0,
            "color_cast": int(retakes["retake_due_to_color_cast"].sum())
            if len(retakes)
            else 0,
            "occlusion": int(retakes["retake_due_to_occlusion"].sum())
            if len(retakes)
            else 0,
        },
        "raw_bucket_counts": raw_bucket_counts,
    }


def verify_aggregation_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    bugs = []
    for _index, row in frame.iterrows():
        checks = json.loads(row["_checks_json"])
        has_retake = any(
            _effect_of(check) == Decision.RETAKE.value for check in checks.values()
        )
        if row["unified_decision"] == "retake" and not has_retake:
            bugs.append(
                {
                    "sample_id": row["sample_id"],
                    "type": "retake_without_trigger",
                }
            )
        if row["unified_decision"] == "pass" and has_retake:
            bugs.append(
                {
                    "sample_id": row["sample_id"],
                    "type": "pass_with_retake_trigger",
                }
            )
    return {"bug_count": len(bugs), "bugs": bugs}


def verify_stain_mapping(frame: pd.DataFrame) -> dict[str, Any]:
    bugs = []
    for _index, row in frame.iterrows():
        probability = row["stain_probability"]
        if probability is None or (isinstance(probability, float) and np.isnan(probability)):
            continue
        expected = _stain_band(float(probability))
        finding = row["stain_finding"]
        effect = row["stain_decision_effect"]
        mapping = {
            "clear": ("false", None),
            "uncertain": ("uncertain", "warning"),
            "stain": ("true", "retake"),
        }
        exp_finding, exp_effect = mapping[expected]
        if finding != exp_finding or effect != exp_effect:
            # effect None vs missing
            if not (exp_effect is None and effect in {None, "pass"} and finding == exp_finding):
                # false may have decision_effect pass
                if expected == "clear" and finding == "false" and effect in {None, "pass"}:
                    continue
                bugs.append(
                    {
                        "sample_id": row["sample_id"],
                        "p_stain": float(probability),
                        "expected_band": expected,
                        "finding": finding,
                        "effect": effect,
                    }
                )
    return {"bug_count": len(bugs), "bugs": bugs[:20]}


def stain_distribution_report(frame: pd.DataFrame) -> dict[str, Any]:
    def _block(subset: pd.DataFrame) -> dict[str, Any]:
        probs = subset["stain_probability"].dropna().astype(float).to_numpy()
        bands = subset["stain_band"].fillna("missing")
        return {
            "n": int(len(subset)),
            "quantiles": _quantiles(probs),
            "band_counts": {
                "clear": int((bands == "clear").sum()),
                "uncertain": int((bands == "uncertain").sum()),
                "stain": int((bands == "stain").sum()),
                "missing": int((bands == "missing").sum()),
            },
            "rate_p_le_clear": float((probs <= STAIN_T_CLEAR).mean()) if probs.size else None,
            "rate_uncertain_band": float(
                ((probs > STAIN_T_CLEAR) & (probs < STAIN_T_RETAKE)).mean()
            )
            if probs.size
            else None,
            "rate_p_ge_retake": float((probs >= STAIN_T_RETAKE).mean())
            if probs.size
            else None,
        }

    overall = _block(frame)
    biohit = _block(frame[frame["dataset"] == "biohit"])
    tongueset3 = _block(frame[frame["dataset"] == "tongueset3"])
    # domain shift heuristic：stain true rate >> D4-C negative prevalence
    possible_shift = bool(
        (biohit["rate_p_ge_retake"] or 0) > 0.20
        or (tongueset3["rate_p_ge_retake"] or 0) > 0.20
    )
    identity_shift = False
    if biohit["rate_p_ge_retake"] is not None and tongueset3["rate_p_ge_retake"] is not None:
        identity_shift = abs(
            biohit["rate_p_ge_retake"] - tongueset3["rate_p_ge_retake"]
        ) >= 0.25
    return {
        "overall": overall,
        "biohit": biohit,
        "tongueset3": tongueset3,
        "possible_cross_domain_probability_shift": possible_shift,
        "dataset_identity_shift_suspected": identity_shift,
        "uncertain_band_utilization": {
            "overall_uncertain_count": overall["band_counts"]["uncertain"],
            "note": "known limitation if remains near 0",
        },
    }


def compare_with_d4c_negatives(
    frame: pd.DataFrame,
    d4c_test_predictions: str | Path,
) -> dict[str, Any]:
    path = Path(d4c_test_predictions)
    if not path.exists():
        return {"available": False}
    pred = pd.read_parquet(path)
    negatives = pred[pred["label"] == 0]["p_stain"].astype(float).to_numpy()
    bio = frame.loc[frame["dataset"] == "biohit", "stain_probability"].dropna().astype(float).to_numpy()
    ts3 = frame.loc[
        frame["dataset"] == "tongueset3", "stain_probability"
    ].dropna().astype(float).to_numpy()

    def _shift(reference: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
        if reference.size == 0 or target.size == 0:
            return {"median_shift": None, "mean_shift": None, "ks": None, "wasserstein": None}
        median_shift = float(np.median(target) - np.median(reference))
        mean_shift = float(np.mean(target) - np.mean(reference))
        try:
            from scipy.stats import ks_2samp, wasserstein_distance

            ks = float(ks_2samp(reference, target).statistic)
            wass = float(wasserstein_distance(reference, target))
        except Exception:
            # 无 scipy：简易 KS 近似
            grid = np.linspace(0, 1, 101)
            ref_cdf = np.searchsorted(np.sort(reference), grid, side="right") / len(
                reference
            )
            tgt_cdf = np.searchsorted(np.sort(target), grid, side="right") / len(target)
            ks = float(np.max(np.abs(ref_cdf - tgt_cdf)))
            wass = float(abs(np.mean(target) - np.mean(reference)))
        return {
            "median_shift": median_shift,
            "mean_shift": mean_shift,
            "ks": ks,
            "wasserstein": wass,
        }

    return {
        "available": True,
        "d4c_negative_quantiles": _quantiles(negatives),
        "biohit_vs_d4c_neg": _shift(negatives, bio),
        "tongueset3_vs_d4c_neg": _shift(negatives, ts3),
        "d4c_negative_median": float(np.median(negatives)) if negatives.size else None,
        "biohit_median": float(np.median(bio)) if bio.size else None,
        "tongueset3_median": float(np.median(ts3)) if ts3.size else None,
    }


def quality_stratum_stain(frame: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for status in ("pass", "warning", "retake"):
        subset = frame[frame["D4B_decision"] == status]
        probs = subset["stain_probability"].dropna().astype(float)
        out[status] = {
            "n": int(len(subset)),
            "p_stain_median": float(probs.median()) if len(probs) else None,
            "stain_true_rate": float((subset["stain_band"] == "stain").mean())
            if len(subset)
            else None,
            "uncertain_rate": float((subset["stain_band"] == "uncertain").mean())
            if len(subset)
            else None,
        }
    return out


def check_status_breakdown(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    finding_col = f"{prefix}_finding"
    effect_col = f"{prefix}_decision_effect"
    state_col = f"{prefix}_evaluation_state"
    overall = {
        "findings": frame[finding_col].fillna("null").value_counts().to_dict(),
        "effects": frame[effect_col].fillna("null").value_counts().to_dict(),
        "states": frame[state_col].fillna("null").value_counts().to_dict(),
        "retake_count": int((frame[effect_col] == "retake").sum()),
        "warning_count": int((frame[effect_col] == "warning").sum()),
        "unavailable_count": int((frame[state_col] == "unavailable").sum()),
    }
    by_dataset = {}
    for dataset_name, subset in frame.groupby("dataset"):
        by_dataset[str(dataset_name)] = {
            "findings": subset[finding_col].fillna("null").value_counts().to_dict(),
            "effects": subset[effect_col].fillna("null").value_counts().to_dict(),
            "retake_count": int((subset[effect_col] == "retake").sum()),
            "warning_count": int((subset[effect_col] == "warning").sum()),
            "unavailable_count": int((subset[state_col] == "unavailable").sum()),
            "n": int(len(subset)),
        }
    return {"overall": overall, "by_dataset": by_dataset}


def evaluation_complete_false_reasons(frame: pd.DataFrame) -> dict[str, Any]:
    incomplete = frame[~frame["evaluation_complete"]]
    reasons = {
        "color_cast_unavailable": 0,
        "occlusion_unavailable": 0,
        "stain_unavailable": 0,
        "other": 0,
    }
    details = []
    for _index, row in incomplete.iterrows():
        flags = []
        if row["color_cast_evaluation_state"] != "evaluated":
            flags.append("color_cast_" + str(row["color_cast_evaluation_state"]))
            if row["color_cast_evaluation_state"] == "unavailable":
                reasons["color_cast_unavailable"] += 1
        if row["occlusion_evaluation_state"] != "evaluated":
            flags.append("occlusion_" + str(row["occlusion_evaluation_state"]))
            if row["occlusion_evaluation_state"] == "unavailable":
                reasons["occlusion_unavailable"] += 1
        if row["stain_evaluation_state"] != "evaluated":
            flags.append("stain_" + str(row["stain_evaluation_state"]))
            if row["stain_evaluation_state"] == "unavailable":
                reasons["stain_unavailable"] += 1
        if not flags:
            reasons["other"] += 1
            flags = ["other"]
        details.append({"sample_id": row["sample_id"], "flags": flags})
    return {"count": int(len(incomplete)), "reason_counts": reasons, "details": details}


def recommend(
    *,
    attribution: dict[str, Any],
    newly_rejected: pd.DataFrame,
    stain_report: dict[str, Any],
    cast_report: dict[str, Any],
    occ_report: dict[str, Any],
    agg_bugs: dict[str, Any],
    mapping_bugs: dict[str, Any],
) -> dict[str, Any]:
    if agg_bugs["bug_count"] or mapping_bugs["bug_count"]:
        return {
            "recommendation": "INTEGRATION_BUG_FOUND",
            "guard_ready_recommendation": "SET_FALSE_PENDING_DOMAIN_FIX",
            "rationale": "aggregation or stain mapping integrity failed",
        }

    new_n = int(len(newly_rejected))
    new_stain = int(newly_rejected["retake_due_to_stain"].sum()) if new_n else 0
    new_cast = int(newly_rejected["retake_due_to_color_cast"].sum()) if new_n else 0
    new_occ = int(newly_rejected["retake_due_to_occlusion"].sum()) if new_n else 0
    new_stain_only = int(
        (
            newly_rejected["retake_due_to_stain"]
            & ~newly_rejected["retake_due_to_d4b"]
            & ~newly_rejected["retake_due_to_color_cast"]
            & ~newly_rejected["retake_due_to_occlusion"]
        ).sum()
    ) if new_n else 0

    cast_retake = cast_report["overall"]["retake_count"]
    occ_retake = occ_report["overall"]["retake_count"]
    concerns = []
    if stain_report["possible_cross_domain_probability_shift"] and new_stain_only >= max(
        20, int(0.4 * max(new_n, 1))
    ):
        concerns.append("D4C_CROSS_DOMAIN_CONCERN")
    if cast_retake >= 20:
        concerns.append("COLOR_CAST_DOMAIN_CONCERN")
    if occ_retake >= 10:
        concerns.append("OCCLUSION_DOMAIN_CONCERN")

    if len(concerns) > 1:
        recommendation = "MULTIPLE_CONCERNS"
        guard = "SET_FALSE_PENDING_DOMAIN_FIX"
    elif len(concerns) == 1:
        recommendation = concerns[0]
        guard = "SET_FALSE_PENDING_DOMAIN_FIX"
    else:
        recommendation = "D4_FINAL_READY"
        guard = "KEEP_TRUE"

    return {
        "recommendation": recommendation,
        "guard_ready_recommendation": guard,
        "newly_rejected_n": new_n,
        "newly_rejected_stain_only": new_stain_only,
        "newly_rejected_stain_trigger": new_stain,
        "newly_rejected_cast_trigger": new_cast,
        "newly_rejected_occlusion_trigger": new_occ,
        "concerns": concerns,
    }


def run_integration_audit(
    *,
    checkpoint_path: str | Path,
    segmentation_dir: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    policy_path: str | Path,
    stain_checkpoint: str | Path,
    stain_thresholds: str | Path,
    output_dir: str | Path = "reports/d4",
    device: str = "auto",
    d4c_test_predictions: str | Path = "runs/input_guard/d4c/stain/test_predictions.parquet",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 冻结指纹（只读校验，不修改）
    policy = load_input_guard_policy(policy_path)
    stain_thr = json.loads(Path(stain_thresholds).read_text(encoding="utf-8"))
    if float(stain_thr["t_clear"]) != STAIN_T_CLEAR or float(stain_thr["t_retake"]) != STAIN_T_RETAKE:
        raise RuntimeError("stain thresholds drifted from frozen 0.95/0.96")
    policy_stain = policy.check_config("stain_suspected")["thresholds"]
    if float(policy_stain["clear"]) != STAIN_T_CLEAR or float(policy_stain["retake"]) != STAIN_T_RETAKE:
        raise RuntimeError("policy stain thresholds drifted")

    frame = collect_sample_audit_rows(
        checkpoint_path=checkpoint_path,
        segmentation_dir=segmentation_dir,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        policy_path=policy_path,
        stain_checkpoint=stain_checkpoint,
        stain_thresholds=stain_thresholds,
        device=device,
    )
    if len(frame) == 0:
        raise RuntimeError("no test samples found")

    # 保存 sample audit（去掉巨大 checks json 的 parquet 另存精简版）
    export_cols = [column for column in frame.columns if column != "_checks_json"]
    sample_path = output_dir / "d4d1_unified_sample_audit.parquet"
    frame[export_cols].to_parquet(sample_path, index=False)
    frame[export_cols].to_csv(output_dir / "d4d1_unified_sample_audit.csv", index=False)
    # 完整含 checks 供 ablation 复用
    frame.to_pickle(output_dir / "d4d1_unified_sample_audit_full.pkl")

    ablation = build_ablation_table(frame, policy_path)
    attribution = build_retake_attribution(frame)
    newly = frame[
        (frame["D4B_decision"] != "retake") & (frame["unified_decision"] == "retake")
    ].copy()
    newly_attr = {
        "n": int(len(newly)),
        "stain_only": int(
            (
                newly["retake_due_to_stain"]
                & ~newly["retake_due_to_d4b"]
                & ~newly["retake_due_to_color_cast"]
                & ~newly["retake_due_to_occlusion"]
            ).sum()
        ),
        "color_cast_only": int(
            (
                newly["retake_due_to_color_cast"]
                & ~newly["retake_due_to_d4b"]
                & ~newly["retake_due_to_stain"]
                & ~newly["retake_due_to_occlusion"]
            ).sum()
        ),
        "occlusion_only": int(
            (
                newly["retake_due_to_occlusion"]
                & ~newly["retake_due_to_d4b"]
                & ~newly["retake_due_to_stain"]
                & ~newly["retake_due_to_color_cast"]
            ).sum()
        ),
        "multiple": int(newly["multi_retake_reason"].sum()),
        "stain_trigger": int(newly["retake_due_to_stain"].sum()),
        "color_cast_trigger": int(newly["retake_due_to_color_cast"].sum()),
        "occlusion_trigger": int(newly["retake_due_to_occlusion"].sum()),
        "d4b_trigger": int(newly["retake_due_to_d4b"].sum()),
    }

    stain_report = stain_distribution_report(frame)
    domain_compare = compare_with_d4c_negatives(frame, d4c_test_predictions)
    stratum = quality_stratum_stain(frame)
    cast_report = check_status_breakdown(frame, "color_cast")
    occ_report = check_status_breakdown(frame, "occlusion")
    agg_bugs = verify_aggregation_integrity(frame)
    mapping_bugs = verify_stain_mapping(frame)
    incomplete = evaluation_complete_false_reasons(frame)

    reason_counts = (
        frame["all_reason_codes"]
        .fillna("")
        .str.split("|")
        .explode()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .to_dict()
    )
    primary_counts = frame["primary_reason"].fillna("null").value_counts().to_dict()

    retake_samples = frame[frame["unified_decision"] == "retake"][
        [
            "sample_id",
            "dataset",
            "unified_decision",
            "primary_reason",
            "all_reason_codes",
            "stain_probability",
            "stain_finding",
            "color_cast_finding",
            "occlusion_finding",
            "D4B_decision",
        ]
    ]
    retake_samples.to_csv(output_dir / "d4d1_retake_samples.csv", index=False)
    newly[
        [
            "sample_id",
            "dataset",
            "D4B_decision",
            "unified_decision",
            "primary_reason",
            "all_reason_codes",
            "stain_probability",
            "stain_finding",
            "color_cast_finding",
            "occlusion_finding",
            "retake_due_to_stain",
            "retake_due_to_color_cast",
            "retake_due_to_occlusion",
            "retake_due_to_d4b",
        ]
    ].to_csv(output_dir / "d4d1_newly_rejected_samples.csv", index=False)

    # optional top lists
    review = {}
    for label, mask in (
        (
            "stain_only",
            newly["retake_due_to_stain"]
            & ~newly["retake_due_to_color_cast"]
            & ~newly["retake_due_to_occlusion"],
        ),
        (
            "cast_only",
            newly["retake_due_to_color_cast"]
            & ~newly["retake_due_to_stain"]
            & ~newly["retake_due_to_occlusion"],
        ),
        (
            "occlusion_only",
            newly["retake_due_to_occlusion"]
            & ~newly["retake_due_to_stain"]
            & ~newly["retake_due_to_color_cast"],
        ),
        ("multiple", newly["multi_retake_reason"]),
    ):
        subset = newly[mask].sort_values("sample_id").head(10)
        review[label] = subset[["sample_id", "dataset", "image_path"]].to_dict(
            orient="records"
        )

    bio278 = frame[frame["sample_id"] == "biohit::278.bmp"]
    bio278_payload = None
    if len(bio278):
        row = bio278.iloc[0]
        bio278_payload = {
            "sample_id": row["sample_id"],
            "D4B_decision": row["D4B_decision"],
            "stain_probability": row["stain_probability"],
            "stain_finding": row["stain_finding"],
            "color_cast_finding": row["color_cast_finding"],
            "occlusion_finding": row["occlusion_finding"],
            "unified_decision": row["unified_decision"],
            "primary_reason": row["primary_reason"],
            "all_reason_codes": row["all_reason_codes"],
        }

    # shortcut correlations（只读）
    corr = {}
    numeric_cols = [
        "foreground_ratio",
        "mean_luminance",
        "roi_blur_score",
        "effective_short_side_px",
    ]
    if frame["stain_probability"].notna().sum() > 5:
        for column in numeric_cols:
            if frame[column].notna().sum() > 5:
                corr[column] = float(
                    np.corrcoef(
                        frame[column].astype(float).fillna(frame[column].median()),
                        frame["stain_probability"].astype(float).fillna(0),
                    )[0, 1]
                )

    recommendation = recommend(
        attribution=attribution,
        newly_rejected=newly,
        stain_report=stain_report,
        cast_report=cast_report,
        occ_report=occ_report,
        agg_bugs=agg_bugs,
        mapping_bugs=mapping_bugs,
    )

    attribution_path = output_dir / "d4d1_retake_attribution.json"
    attribution_path.write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "stage": "D4-D.1",
        "git_note": "working tree may contain uncommitted D4-C/D4-D; audit is read-only",
        "n_samples": int(len(frame)),
        "dataset_counts": frame["dataset"].value_counts().to_dict(),
        "ablation": ablation,
        "retake_attribution": attribution,
        "newly_rejected": newly_attr,
        "stain_distribution": stain_report,
        "stain_domain_compare": domain_compare,
        "stain_quality_stratum": stratum,
        "color_cast": cast_report,
        "occlusion": occ_report,
        "reason_code_counts": reason_counts,
        "primary_reason_counts": primary_counts,
        "evaluation_complete_false": incomplete,
        "aggregation_bugs": agg_bugs,
        "stain_mapping_bugs": mapping_bugs,
        "biohit_278": bio278_payload,
        "shortcut_correlations": corr,
        "review_samples": review,
        "frozen_checks": {
            "stain_t_clear": STAIN_T_CLEAR,
            "stain_t_retake": STAIN_T_RETAKE,
            "policy_version": policy.policy_version,
            "thresholds_modified": False,
            "runtime_modified": False,
        },
        "recommendation": recommendation,
    }
    (output_dir / "d4d1_integration_audit_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # docs copy
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "D4_D_1_INTEGRATION_AUDIT_STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_integration_audit_markdown(stats, docs / "D4_D_1_INTEGRATION_AUDIT.md")
    return stats


def write_integration_audit_markdown(stats: dict[str, Any], path: str | Path) -> None:
    """生成 D4-D.1 Integration Audit 报告（非 Freeze Report）。"""
    path = Path(path)
    ablation = stats["ablation"]["counts"]
    delta = stats["ablation"]["delta_retake"]
    attr = stats["retake_attribution"]
    newly = stats["newly_rejected"]
    stain = stats["stain_distribution"]
    cast = stats["color_cast"]["overall"]
    occ = stats["occlusion"]["overall"]
    rec = stats["recommendation"]
    bio = stats.get("biohit_278")
    lines = [
        "# D4-D.1 Unified Guard Integration Audit",
        "",
        "> 本报告用于决定是否允许 D4 Final Freeze；**不是** Freeze Report。",
        "",
        f"- stage: `{stats['stage']}`",
        f"- n_samples: `{stats['n_samples']}`",
        f"- dataset_counts: `{stats['dataset_counts']}`",
        f"- recommendation: **`{rec['recommendation']}`**",
        f"- guard_ready_recommendation: `{rec['guard_ready_recommendation']}`",
        "",
        "## 1. 为什么 RETAKE 从 13 增加到 80？",
        "",
        "四级 ablation（同一 130 sample set，复用 per-check frozen outputs）：",
        "",
        f"- A D4-B only: `{ablation['A_d4b_only']}`",
        f"- B +stain: `{ablation['B_d4b_stain']}`",
        f"- C +color_cast: `{ablation['C_d4b_stain_cast']}`",
        f"- D full: `{ablation['D_full']}`",
        "",
        f"- Δ retake B−A (stain): `{delta['B_minus_A']}`",
        f"- Δ retake C−B (color_cast): `{delta['C_minus_B']}`",
        f"- Δ retake D−C (occlusion): `{delta['D_minus_C']}`",
        "",
        "## 2. 新增 RETAKE attribution",
        "",
        f"- newly_rejected_n: `{newly['n']}`",
        f"- stain_only: `{newly['stain_only']}`",
        f"- color_cast_only: `{newly['color_cast_only']}`",
        f"- occlusion_only: `{newly['occlusion_only']}`",
        f"- multiple: `{newly['multiple']}`",
        f"- stain_trigger (any): `{newly['stain_trigger']}`",
        f"- color_cast_trigger (any): `{newly['color_cast_trigger']}`",
        f"- occlusion_trigger (any): `{newly['occlusion_trigger']}`",
        "",
        "## 3–4. Stain finding counts",
        "",
        f"- overall: `{stain['overall']['band_counts']}`",
        f"- BioHit: `{stain['biohit']['band_counts']}`",
        f"- TongueSet3: `{stain['tongueset3']['band_counts']}`",
        "",
        "## 5. Cross-domain probability shift",
        "",
        f"- possible_cross_domain_probability_shift: "
        f"`{stain['possible_cross_domain_probability_shift']}`",
        f"- dataset_identity_shift_suspected: "
        f"`{stain['dataset_identity_shift_suspected']}`",
        f"- overall p>=0.96 rate: `{stain['overall']['rate_p_ge_retake']}`",
        f"- BioHit p>=0.96 rate: `{stain['biohit']['rate_p_ge_retake']}`",
        f"- TongueSet3 p>=0.96 rate: `{stain['tongueset3']['rate_p_ge_retake']}`",
        f"- uncertain count: `{stain['uncertain_band_utilization']['overall_uncertain_count']}`",
        f"- domain compare: `{stats['stain_domain_compare']}`",
        "",
        "## 6–7. Color cast / Occlusion",
        "",
        f"- color_cast: findings=`{cast['findings']}` retake=`{cast['retake_count']}` "
        f"warning=`{cast['warning_count']}` unavailable=`{cast['unavailable_count']}`",
        f"- occlusion: findings=`{occ['findings']}` retake=`{occ['retake_count']}` "
        f"warning=`{occ['warning_count']}` unavailable=`{occ['unavailable_count']}`",
        "",
        "## 8–9. Integrity",
        "",
        f"- aggregation_bugs: `{stats['aggregation_bugs']['bug_count']}`",
        f"- stain_mapping_bugs: `{stats['stain_mapping_bugs']['bug_count']}`",
        "",
        "## 10. evaluation_complete=false",
        "",
        f"`{stats['evaluation_complete_false']}`",
        "",
        "## 11. biohit::278",
        "",
        f"`{bio}`",
        "",
        "## 12. guard_ready recommendation",
        "",
        f"- system guard_ready remains true in runtime; audit recommendation = "
        f"`{rec['guard_ready_recommendation']}`",
        "",
        "## Full retake attribution (80)",
        "",
        f"- by_source: `{attr['by_source']}`",
        f"- unique_trigger_count: `{attr['unique_trigger_count']}`",
        "",
        "## Reason / primary reason",
        "",
        f"- reason_code_counts: `{stats['reason_code_counts']}`",
        f"- primary_reason_counts: `{stats['primary_reason_counts']}`",
        "",
        "## Decision note",
        "",
        "High RETAKE rate alone is **not** an automatic FAIL. "
        "The recommendation is based on trigger attribution and "
        "cross-domain probability shift evidence.",
        "",
        "本阶段未修改 runtime / thresholds / checkpoints / splits。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
