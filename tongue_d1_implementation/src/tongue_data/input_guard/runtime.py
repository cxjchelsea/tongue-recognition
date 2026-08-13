"""D4 Unified Input Guard runtime：11 checks + readiness semantics。"""
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
from .guidance import guidance_list_for_reasons
from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    EvaluationState,
    implemented_checks_count,
)
from .occlusion import evaluate_occlusion
from .policy import InputGuardPolicy, load_input_guard_policy
from .schema import InputGuardResult
from .signal_checks import IMPLEMENTED_SIGNAL_CHECKS, evaluate_signal_checks
from .signal_features import enrich_features_with_signals


def _load_d4d_config(path: str | Path | None) -> dict:
    config_path = Path(path or "configs/input_guard_d4d_v1.yaml")
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def compute_evaluation_complete(checks: dict) -> bool:
    """所有 enabled 且应执行的 check 均为 evaluated → True。"""
    for check in checks.values():
        if check.evaluation_state != EvaluationState.EVALUATED.value:
            return False
    return True


def compute_system_guard_ready(policy: InputGuardPolicy) -> bool:
    """系统层：11 项均 implemented 且 color/occlusion 正式 PASS。"""
    if implemented_checks_count() < 11:
        return False
    for check_id in CheckId:
        meta = CHECK_DEFINITIONS[check_id]
        if not meta.get("implemented"):
            return False
    color_cfg = policy.check_config(CheckId.COLOR_CAST)
    occ_cfg = policy.check_config(CheckId.OCCLUSION)
    if color_cfg.get("status") != "PASS":
        return False
    if occ_cfg.get("status") != "PASS":
        return False
    if color_cfg.get("needs_calibration"):
        return False
    if occ_cfg.get("needs_calibration"):
        return False
    return True


class InputGuardRuntime:
    """统一 Input Guard；guard_ready/evaluation_complete 按语义计算。"""

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
        self.stain_detector = stain_detector
        if self.stain_detector is None and stain_checkpoint is not None:
            from tongue_data.stain.detector import StainDetector

            self.stain_detector = StainDetector(
                checkpoint_path=stain_checkpoint,
                data_config_path=stain_data_config or "configs/stain_detection_v1.yaml",
                train_config_path=stain_train_config or "configs/stain_train_v1.yaml",
                thresholds_path=stain_thresholds
                or Path(stain_checkpoint).parent / "thresholds.json",
                device=device,
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

        # D4-C stain
        if (
            self.stain_detector is not None
            and self.policy.is_check_enabled(CheckId.STAIN_SUSPECTED)
        ):
            stain_result = self.stain_detector.predict(
                original_rgb, segmentation_result
            )
            if "coating.color" in (stain_result.evidence or {}):
                raise RuntimeError("stain check must not emit coating.color")
            checks[CheckId.STAIN_SUSPECTED.value] = stain_result

        no_tongue = features.segmentation_status == "no_tongue_detected" or (
            features.tongue_pixel_count is not None
            and int(features.tongue_pixel_count) <= 0
        )
        mask = getattr(segmentation_result, "original_binary_mask", None)
        prob = getattr(segmentation_result, "original_probability_mask", None)

        # D4-D color_cast / occlusion（no tongue 时保持 not_evaluated）
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

        # no_tongue：ROI checks not_evaluated，但决策可行动
        evaluation_complete = compute_evaluation_complete(checks)
        if no_tongue and final == Decision.RETAKE:
            # 保持 evaluation_complete=false（ROI 未评估），但不阻断 usable 决策
            evaluation_complete = False

        guard_ready = compute_system_guard_ready(self.policy)

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
            notes=[
                "D4 unified runtime: signal + stain + color_cast + occlusion.",
                f"ontology_implemented_count={implemented_checks_count()}",
                f"runtime_evaluated_checks={evaluated}",
                f"not_evaluated_or_unavailable={not_evaluated}",
                "quality_confidence intentionally null (no fake score).",
                "unavailable != pass; evaluation_complete is sample-level.",
                "guard_ready is system-level readiness.",
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
        f"primary_reason: {result.primary_reason}",
        f"reason_codes: {result.reason_codes}",
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
