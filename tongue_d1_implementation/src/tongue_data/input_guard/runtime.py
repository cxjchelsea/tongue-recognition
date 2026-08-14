"""D4 Unified Input Guard runtime：active checks + deferred stain semantics。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .color_cast import evaluate_color_cast
from .decision import (
    aggregate_decision,
    decision_usable,
    select_primary_reason,
)
from .guidance import capture_guidance_list, guidance_list_for_reasons
from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    EvaluationState,
    defined_checks_count,
    implemented_checks_count,
)
from .occlusion import evaluate_occlusion
from .policy import InputGuardPolicy, load_input_guard_policy
from .schema import CheckResult, InputGuardResult, make_not_evaluated_check
from .signal_checks import IMPLEMENTED_SIGNAL_CHECKS, evaluate_signal_checks
from .signal_features import enrich_features_with_signals


def _load_d4d_config(path: str | Path | None) -> dict:
    config_path = Path(path or "configs/input_guard_d4d_v1.yaml")
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def compute_evaluation_complete(
    checks: dict,
    policy: InputGuardPolicy | None = None,
) -> bool:
    """仅针对 enabled production checks；deferred/disabled 不导致 incomplete。"""
    for check_id, check in checks.items():
        if policy is not None and not policy.is_check_enabled(check_id):
            continue
        if check.evaluation_state != EvaluationState.EVALUATED.value:
            return False
    return True


def compute_system_guard_ready(policy: InputGuardPolicy) -> bool:
    """
    系统层 guard_ready：
    当前 production policy 中所有 enabled checks 均有可用实现。
    deferred checks 不阻断。
    """
    enabled = policy.active_check_ids()
    if not enabled:
        return False
    for check_id in enabled:
        meta = CHECK_DEFINITIONS[check_id]
        if not meta.get("implemented"):
            return False
        cfg = policy.check_config(check_id)
        if cfg.get("needs_calibration"):
            return False
        if check_id in {CheckId.COLOR_CAST, CheckId.OCCLUSION}:
            if cfg.get("status") != "PASS":
                return False
    return True


def compute_full_capability_coverage(policy: InputGuardPolicy) -> bool:
    """11/11 设计 capability 是否全部 production-enabled。"""
    if len(policy.deferred_check_ids()) > 0:
        return False
    return len(policy.active_check_ids()) == len(list(CheckId))


def make_deferred_check(check_id: str, *, deferred_reason: str) -> CheckResult:
    """disabled/deferred：finding=null，decision_effect=none，≠ false。"""
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.NOT_EVALUATED.value,
        finding=None,
        severity="none",
        decision_effect=None,
        score=None,
        thresholds=None,
        evidence={
            "capability_status": "deferred",
            "deferred_reason": deferred_reason,
            "evaluation_state_note": "not_evaluated",
            "decision_effect": "none",
        },
        reason_code=None,
        source="contract_skeleton",
    )


class InputGuardRuntime:
    """统一 Input Guard；policy 优先于传入 stain checkpoint。"""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        data_config: str | Path,
        train_config: str | Path,
        policy_path: str | Path = "configs/input_guard_v1.yaml",
        device: str = "auto",
        stain_detector: Any | None = None,
        stain_checkpoint: str | Path | None = None,
        stain_data_config: str | Path | None = None,
        stain_train_config: str | Path | None = None,
        stain_thresholds: str | Path | None = None,
        d4d_config_path: str | Path | None = "configs/input_guard_d4d_v1.yaml",
    ):
        from tongue_data.segmentation.inference import TongueSegmentationInference

        self.policy = load_input_guard_policy(policy_path)
        self.d4d_config = _load_d4d_config(d4d_config_path)
        self.segmentation = TongueSegmentationInference(
            checkpoint_path=checkpoint_path,
            data_config=data_config,
            train_config=train_config,
            device=device,
            return_model_space=False,
            return_probability=True,
            return_masked_roi=False,
        )
        self.stain_model_invocations = 0
        self.stain_detector = None
        # policy 优先：enabled=false 时绝不加载 / 执行 stain
        stain_enabled = self.policy.is_check_enabled(CheckId.STAIN_SUSPECTED)
        if stain_enabled:
            self.stain_detector = stain_detector
            if self.stain_detector is None and stain_checkpoint is not None:
                from tongue_data.stain.detector import StainDetector

                self.stain_detector = StainDetector(
                    checkpoint_path=stain_checkpoint,
                    data_config_path=stain_data_config
                    or "configs/stain_detection_v1.yaml",
                    train_config_path=stain_train_config or "configs/stain_train_v1.yaml",
                    thresholds_path=stain_thresholds
                    or Path(stain_checkpoint).parent / "thresholds.json",
                    device=device,
                )
        # 即使调用方传入 checkpoint，disabled 时仍保持 None
        self._stain_checkpoint_ignored = (
            (not stain_enabled) and (stain_checkpoint is not None or stain_detector is not None)
        )

    def evaluate_from_segmentation(
        self,
        original_rgb: np.ndarray,
        segmentation_result: Any,
    ) -> InputGuardResult:
        original_copy_check = (
            int(original_rgb[0, 0, 0]) if original_rgb.size else None
        )
        features = enrich_features_with_signals(original_rgb, segmentation_result)
        if original_copy_check is not None and int(original_rgb[0, 0, 0]) != original_copy_check:
            raise RuntimeError("original RGB was mutated during feature extraction")

        checks = evaluate_signal_checks(features, self.policy)

        # stain：仅 enabled 时执行；否则写 deferred placeholder（finding=null）
        stain_cfg = self.policy.check_config(CheckId.STAIN_SUSPECTED)
        if self.policy.is_check_enabled(CheckId.STAIN_SUSPECTED):
            if self.stain_detector is not None:
                stain_result = self.stain_detector.predict(
                    original_rgb, segmentation_result
                )
                self.stain_model_invocations += 1
                if "coating.color" in (stain_result.evidence or {}):
                    raise RuntimeError("stain check must not emit coating.color")
                checks[CheckId.STAIN_SUSPECTED.value] = stain_result
        else:
            checks[CheckId.STAIN_SUSPECTED.value] = make_deferred_check(
                CheckId.STAIN_SUSPECTED.value,
                deferred_reason=str(
                    stain_cfg.get("deferred_reason")
                    or "SOURCE_DATASET_CONFOUNDING_SEVERE"
                ),
            )

        no_tongue = features.segmentation_status == "no_tongue_detected" or (
            features.tongue_pixel_count is not None
            and int(features.tongue_pixel_count) <= 0
        )
        mask = getattr(segmentation_result, "original_binary_mask", None)
        prob = getattr(segmentation_result, "original_probability_mask", None)

        if not no_tongue and self.policy.is_check_enabled(CheckId.COLOR_CAST):
            if CHECK_DEFINITIONS[CheckId.COLOR_CAST].get("implemented"):
                checks[CheckId.COLOR_CAST.value] = evaluate_color_cast(
                    original_rgb, mask, self.policy, d4d_cfg=self.d4d_config
                )
        if not no_tongue and self.policy.is_check_enabled(CheckId.OCCLUSION):
            if CHECK_DEFINITIONS[CheckId.OCCLUSION].get("implemented"):
                checks[CheckId.OCCLUSION.value] = evaluate_occlusion(
                    original_rgb,
                    mask,
                    prob,
                    self.policy,
                    d4d_cfg=self.d4d_config,
                )

        effects = []
        reason_codes: list[str] = []
        retake_reasons: set[str] = set()
        warning_reasons: set[str] = set()
        for check in checks.values():
            if check.evaluation_state != EvaluationState.EVALUATED.value:
                continue
            if check.decision_effect is None:
                continue
            effects.append(check.decision_effect)
            if check.reason_code:
                reason_codes.append(check.reason_code)
                if check.decision_effect == Decision.RETAKE.value:
                    retake_reasons.add(check.reason_code)
                elif check.decision_effect == Decision.WARNING.value:
                    warning_reasons.add(check.reason_code)

        deduped: list[str] = []
        seen: set[str] = set()
        for code in reason_codes:
            if code not in seen:
                seen.add(code)
                deduped.append(code)

        final = aggregate_decision(effects)
        primary = select_primary_reason(
            deduped,
            priority=self.policy.primary_reason_priority,
            retake_reasons=retake_reasons,
            warning_reasons=warning_reasons,
        )

        evaluation_complete = compute_evaluation_complete(checks, self.policy)
        if no_tongue and final == Decision.RETAKE:
            evaluation_complete = False

        guard_ready = compute_system_guard_ready(self.policy)
        full_coverage = compute_full_capability_coverage(self.policy)
        known_limitations = list(self.policy.doc.get("known_limitations") or [])
        active = [item.value for item in self.policy.active_check_ids()]
        deferred = [item.value for item in self.policy.deferred_check_ids()]

        evaluated = [
            key
            for key, check in checks.items()
            if check.evaluation_state == EvaluationState.EVALUATED.value
        ]
        not_evaluated = [
            key
            for key, check in checks.items()
            if check.evaluation_state != EvaluationState.EVALUATED.value
        ]
        result = InputGuardResult(
            decision=final.value,
            usable=decision_usable(final),
            evaluation_complete=evaluation_complete,
            guard_ready=guard_ready,
            checks=checks,
            reason_codes=deduped,
            primary_reason=primary,
            warnings=list(warning_reasons),
            retake_guidance=guidance_list_for_reasons(deduped),
            features=features.to_dict(),
            segmentation_reference={
                "status": getattr(segmentation_result, "status", None),
                "sample_id": getattr(segmentation_result, "sample_id", None),
                "threshold": getattr(segmentation_result, "threshold", None),
            },
            quality_confidence=None,
            contract_version=INPUT_GUARD_CONTRACT_VERSION,
            full_capability_coverage=full_coverage,
            known_limitations=known_limitations,
            active_checks=active,
            deferred_checks=deferred,
            capture_guidance=capture_guidance_list(self.policy),
            stain_model_invocations=int(self.stain_model_invocations),
            notes=[
                "D4-E production runtime: 10 active QC + stain deferred.",
                f"defined={defined_checks_count()} implemented={implemented_checks_count()}",
                f"active_checks={active}",
                f"deferred_checks={deferred}",
                f"runtime_evaluated_checks={evaluated}",
                f"not_evaluated_or_unavailable={not_evaluated}",
                f"stain_model_invocations={self.stain_model_invocations}",
                f"stain_checkpoint_ignored={self._stain_checkpoint_ignored}",
                "quality_confidence intentionally null (no fake score).",
                "unavailable != pass; deferred != finding=false.",
                "evaluation_complete only considers enabled checks.",
                "guard_ready is system-level readiness for enabled checks.",
                f"full_capability_coverage={full_coverage}",
            ],
        )
        result.validate()
        return result

    def evaluate(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        sample_id: str | None = None,
    ) -> InputGuardResult:
        from tongue_data.segmentation.inference import load_rgb_image

        original_rgb, _mode = load_rgb_image(image)
        if isinstance(image, np.ndarray):
            working = original_rgb.copy()
        else:
            working = original_rgb
        seg = self.segmentation.predict(working, sample_id=sample_id)
        return self.evaluate_from_segmentation(working, seg)


# 兼容别名
UnifiedInputGuard = InputGuardRuntime


def format_runtime_summary(result: InputGuardResult) -> str:
    lines = [
        f"decision: {result.decision}",
        f"usable: {result.usable}",
        f"evaluation_complete: {result.evaluation_complete}",
        f"guard_ready: {result.guard_ready}",
        f"full_capability_coverage: {result.full_capability_coverage}",
        f"primary_reason: {result.primary_reason}",
        f"reason_codes: {result.reason_codes}",
        f"active_checks: {result.active_checks}",
        f"deferred_checks: {result.deferred_checks}",
        f"known_limitations: {result.known_limitations}",
        f"stain_model_invocations: {result.stain_model_invocations}",
        f"quality_confidence: {result.quality_confidence}",
    ]
    for check_id, check in result.checks.items():
        short = check_id.split(".", 1)[-1]
        lines.append(
            f"  {short}: state={check.evaluation_state} "
            f"finding={check.finding} effect={check.decision_effect} "
            f"reason={check.reason_code}"
        )
    lines.append(
        f"retake_guidance: {[item.get('guidance') for item in result.retake_guidance]}"
    )
    return "\n".join(lines)
