"""D4-B partial runtime：segmentation + signal checks → InputGuardResult。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision import (
    aggregate_decision,
    decision_usable,
    select_primary_reason,
)
from .guidance import guidance_list_for_reasons
from .ontology import INPUT_GUARD_CONTRACT_VERSION, Decision, EvaluationState
from .policy import InputGuardPolicy, load_input_guard_policy
from .schema import InputGuardResult
from .signal_checks import IMPLEMENTED_SIGNAL_CHECKS, evaluate_signal_checks
from .signal_features import enrich_features_with_signals


class InputGuardRuntime:
    """部分运行时门禁；evaluation_complete 始终 false（缺 color/occlusion/stain）。"""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        data_config: str | Path,
        train_config: str | Path,
        policy_path: str | Path = "configs/input_guard_v1.yaml",
        device: str = "auto",
    ):
        from tongue_data.segmentation.inference import TongueSegmentationInference

        self.policy = load_input_guard_policy(policy_path)
        self.segmentation = TongueSegmentationInference(
            checkpoint_path=checkpoint_path,
            data_config=data_config,
            train_config=train_config,
            device=device,
            return_model_space=False,
            return_probability=True,
            return_masked_roi=False,
        )

    def evaluate_from_segmentation(
        self,
        original_rgb: np.ndarray,
        segmentation_result: Any,
    ) -> InputGuardResult:
        # 保护原图不被修改：先做只读校验
        original_copy_check = (
            int(original_rgb[0, 0, 0]) if original_rgb.size else None
        )
        features = enrich_features_with_signals(original_rgb, segmentation_result)
        if original_copy_check is not None and int(original_rgb[0, 0, 0]) != original_copy_check:
            raise RuntimeError("original RGB was mutated during feature extraction")

        checks = evaluate_signal_checks(features, self.policy)
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

        # 去重保序
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
        implemented = [
            check_id.value
            for check_id in IMPLEMENTED_SIGNAL_CHECKS
            if self.policy.is_check_enabled(check_id)
        ]
        not_evaluated = [
            key
            for key, check in checks.items()
            if check.evaluation_state != EvaluationState.EVALUATED.value
        ]
        result = InputGuardResult(
            decision=final.value,
            usable=decision_usable(final),
            evaluation_complete=False,
            guard_ready=False,
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
                "D4-B partial runtime: 8 signal checks implemented; "
                "color_cast/occlusion/stain not evaluated.",
                f"implemented_checks={implemented}",
                f"not_evaluated_checks={not_evaluated}",
                "thresholds are engineering heuristics, not clinical standards.",
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
        # 确保可写副本不被下游污染调用方数组：对 ndarray 输入先复制
        if isinstance(image, np.ndarray):
            working = original_rgb.copy()
        else:
            working = original_rgb
        seg = self.segmentation.predict(working, sample_id=sample_id)
        return self.evaluate_from_segmentation(working, seg)


def format_runtime_summary(result: InputGuardResult) -> str:
    implemented = [
        key
        for key, check in result.checks.items()
        if check.evaluation_state == EvaluationState.EVALUATED.value
    ]
    not_evaluated = [
        key
        for key, check in result.checks.items()
        if check.evaluation_state != EvaluationState.EVALUATED.value
    ]
    lines = [
        f"decision: {result.decision}",
        f"usable: {result.usable}",
        f"evaluation_complete: {result.evaluation_complete}",
        f"guard_ready: {result.guard_ready}",
        f"primary_reason: {result.primary_reason}",
        f"reason_codes: {result.reason_codes}",
        f"implemented_checks: {implemented}",
        f"not_evaluated_checks: {not_evaluated}",
        f"retake_guidance: {[item.get('guidance') for item in result.retake_guidance]}",
    ]
    return "\n".join(lines)
