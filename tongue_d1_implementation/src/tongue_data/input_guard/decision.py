"""Decision aggregation 与 D4-A contract skeleton evaluator。"""
from __future__ import annotations

from typing import Any

from .features import InputGuardFeatures, features_from_segmentation_result
from .guidance import guidance_list_for_reasons
from .ontology import (
    CHECK_DEFINITIONS,
    INPUT_GUARD_CONTRACT_VERSION,
    CheckId,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    Severity,
    DECISION_RANK,
    parse_decision,
    parse_reason_code,
)
from .policy import InputGuardPolicy
from .schema import CheckResult, InputGuardResult, make_not_evaluated_check


def decision_usable(decision: Decision | str) -> bool:
    parsed = parse_decision(decision)
    if parsed == Decision.RETAKE:
        return False
    return True


def aggregate_decision(effects: list[Decision | str | None]) -> Decision:
    """取最严重 action：PASS < WARNING < RETAKE；忽略 None / 未评估。"""
    final = Decision.PASS
    for effect in effects:
        if effect is None:
            continue
        candidate = parse_decision(effect)
        if DECISION_RANK[candidate] > DECISION_RANK[final]:
            final = candidate
    return final


def select_primary_reason(
    reason_codes: list[str | ReasonCode],
    *,
    priority: list[str],
    retake_reasons: set[str] | None = None,
    warning_reasons: set[str] | None = None,
) -> str | None:
    """
    deterministic primary_reason：
    1) RETAKE reasons 优先于 WARNING
    2) 同档按 policy priority
    """
    if not reason_codes:
        return None
    normalized = [parse_reason_code(code).value for code in reason_codes]
    retake_reasons = retake_reasons or set()
    warning_reasons = warning_reasons or set()

    def rank(code: str) -> tuple[int, int]:
        # 0 = retake tier, 1 = warning tier, 2 = other
        if code in retake_reasons:
            tier = 0
        elif code in warning_reasons:
            tier = 1
        else:
            tier = 2
        try:
            priority_index = priority.index(code)
        except ValueError:
            priority_index = len(priority) + normalized.index(code)
        return tier, priority_index

    return sorted(normalized, key=rank)[0]


def _segmentation_reference(result: Any) -> dict[str, Any]:
    return {
        "status": getattr(result, "status", None),
        "sample_id": getattr(result, "sample_id", None),
        "threshold": getattr(result, "threshold", None),
        "original_size": [
            getattr(result, "original_width", None),
            getattr(result, "original_height", None),
        ],
        "bbox_tight": list(result.bbox_tight)
        if getattr(result, "bbox_tight", None) is not None
        else None,
        "bbox_roi": list(result.bbox_roi)
        if getattr(result, "bbox_roi", None) is not None
        else None,
        "warnings": list(getattr(result, "warnings", []) or []),
    }


def evaluate_no_tongue(
    *,
    features: InputGuardFeatures,
    policy: InputGuardPolicy,
    segmentation_reference: dict[str, Any],
) -> InputGuardResult:
    """D3-E no_tongue_detected → 直接 RETAKE；ROI 依赖 check = not_evaluated。"""
    checks: dict[str, CheckResult] = {}
    for check_id, meta in CHECK_DEFINITIONS.items():
        short = check_id.value
        if not policy.is_check_enabled(check_id):
            checks[short] = make_not_evaluated_check(
                short, reason="check_disabled"
            )
            continue
        if check_id == CheckId.TONGUE_PRESENCE:
            checks[short] = CheckResult(
                check_id=short,
                evaluation_state=EvaluationState.EVALUATED.value,
                finding="absent",
                severity=Severity.SEVERE.value,
                decision_effect=Decision.RETAKE.value,
                score=None,
                thresholds=policy.check_config(check_id).get("thresholds"),
                evidence={
                    "segmentation_status": features.segmentation_status,
                    "foreground_ratio": features.foreground_ratio,
                },
                reason_code=ReasonCode.NO_TONGUE_DETECTED.value,
                source=EvidenceSource.SEGMENTATION_METADATA.value,
            )
        elif meta.get("depends_on_roi"):
            checks[short] = make_not_evaluated_check(
                short, reason="no_tongue_roi_unavailable"
            )
        else:
            # 非 ROI 依赖但 D4-A 未实现 → not_evaluated
            checks[short] = make_not_evaluated_check(
                short, reason="implementation_pending"
            )

    reason_codes = [ReasonCode.NO_TONGUE_DETECTED.value]
    return InputGuardResult(
        decision=Decision.RETAKE.value,
        usable=False,
        evaluation_complete=False,
        guard_ready=False,
        checks=checks,
        reason_codes=reason_codes,
        primary_reason=ReasonCode.NO_TONGUE_DETECTED.value,
        warnings=[],
        retake_guidance=guidance_list_for_reasons(reason_codes),
        features=features.to_dict(),
        segmentation_reference=segmentation_reference,
        quality_confidence=None,
        contract_version=INPUT_GUARD_CONTRACT_VERSION,
        notes=[
            "D4-A contract skeleton: only NO_TONGUE_DETECTED is evaluated "
            "from D3-E status; other checks remain not_evaluated."
        ],
    )


def build_contract_skeleton_result(
    segmentation_result: Any,
    policy: InputGuardPolicy,
) -> InputGuardResult:
    """
    D4-A smoke / adapter：
    - 映射 D3-E features
    - no_tongue → RETAKE
    - 其余 enabled 但未实现 check → not_evaluated
    - evaluation_complete=false（勿误称照片已完整 QC）
    """
    features = features_from_segmentation_result(segmentation_result)
    reference = _segmentation_reference(segmentation_result)

    if getattr(segmentation_result, "status", None) == "no_tongue_detected":
        return evaluate_no_tongue(
            features=features,
            policy=policy,
            segmentation_reference=reference,
        )

    checks: dict[str, CheckResult] = {}
    effects: list[Decision | None] = []
    reason_codes: list[str] = []
    retake_reasons: set[str] = set()
    warning_reasons: set[str] = set()

    for check_id, meta in CHECK_DEFINITIONS.items():
        short = check_id.value
        if not policy.is_check_enabled(check_id):
            checks[short] = make_not_evaluated_check(
                short, reason="check_disabled"
            )
            continue
        # D4-A：除将来可扩展外，默认全部 not_evaluated（未实现信号）
        # tongue_presence 在有舌时也标记 not_evaluated（完整规则在 D4-B）
        # 但可附带 segmentation evidence 供审计
        check = make_not_evaluated_check(
            short,
            reason=f"defined_only; implementation_stage={meta['implementation_stage']}",
        )
        if check_id == CheckId.TONGUE_PRESENCE:
            check.evidence = {
                "reason": check.evidence.get("reason"),
                "segmentation_status": features.segmentation_status,
                "foreground_ratio": features.foreground_ratio,
                "note": "presence rule baseline deferred to D4-B",
            }
        checks[short] = check

    # 未实现 check 不得影响 final decision → 保持 PASS skeleton
    final = aggregate_decision(effects)
    result = InputGuardResult(
        decision=final.value,
        usable=decision_usable(final),
        evaluation_complete=False,
        guard_ready=False,
        checks=checks,
        reason_codes=reason_codes,
        primary_reason=select_primary_reason(
            reason_codes,
            priority=policy.primary_reason_priority,
            retake_reasons=retake_reasons,
            warning_reasons=warning_reasons,
        ),
        warnings=[
            "evaluation_complete=false: D4-A contract only; "
            "do not treat decision as full quality clearance"
        ],
        retake_guidance=[],
        features=features.to_dict(),
        segmentation_reference=reference,
        quality_confidence=None,
        contract_version=INPUT_GUARD_CONTRACT_VERSION,
        notes=[
            "D4-A smoke: contract_status=PASS means schema/adapter OK, "
            "not that image quality is clinically cleared."
        ],
    )
    result.validate()
    return result


def build_result_from_check_effects(
    *,
    checks: dict[str, CheckResult],
    features: InputGuardFeatures | dict[str, Any] | None = None,
    policy: InputGuardPolicy | None = None,
    evaluation_complete: bool = True,
    segmentation_reference: dict[str, Any] | None = None,
) -> InputGuardResult:
    """
    单元测试 / 未来 D4-B 聚合入口：
    仅聚合 evaluation_state=evaluated 且有 decision_effect 的 checks。
    """
    effects: list[Decision | None] = []
    reason_codes: list[str] = []
    retake_reasons: set[str] = set()
    warning_reasons: set[str] = set()
    warning_messages: list[str] = []

    for check in checks.values():
        check.validate()
        if check.evaluation_state != EvaluationState.EVALUATED.value:
            continue
        if check.decision_effect is None:
            continue
        effect = parse_decision(check.decision_effect)
        effects.append(effect)
        if check.reason_code:
            code = parse_reason_code(check.reason_code).value
            reason_codes.append(code)
            if effect == Decision.RETAKE:
                retake_reasons.add(code)
            elif effect == Decision.WARNING:
                warning_reasons.add(code)
                warning_messages.append(code)

    # 去重保持首次出现顺序
    deduped: list[str] = []
    seen: set[str] = set()
    for code in reason_codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)

    final = aggregate_decision(effects)
    priority = policy.primary_reason_priority if policy is not None else [
        code.value for code in ReasonCode
    ]
    primary = select_primary_reason(
        deduped,
        priority=priority,
        retake_reasons=retake_reasons,
        warning_reasons=warning_reasons,
    )
    feature_dict = (
        features.to_dict()
        if isinstance(features, InputGuardFeatures)
        else dict(features or {})
    )
    result = InputGuardResult(
        decision=final.value,
        usable=decision_usable(final),
        evaluation_complete=bool(evaluation_complete),
        guard_ready=bool(evaluation_complete),
        checks=checks,
        reason_codes=deduped,
        primary_reason=primary,
        warnings=warning_messages,
        retake_guidance=guidance_list_for_reasons(deduped)
        if final == Decision.RETAKE
        else guidance_list_for_reasons(deduped),
        features=feature_dict,
        segmentation_reference=dict(segmentation_reference or {}),
        quality_confidence=None,
        contract_version=INPUT_GUARD_CONTRACT_VERSION,
    )
    result.validate()
    return result
