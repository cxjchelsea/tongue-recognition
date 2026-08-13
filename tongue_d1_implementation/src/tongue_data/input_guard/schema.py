"""Input Guard 结果 schema：CheckResult / InputGuardResult。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    Severity,
    parse_decision,
    parse_reason_code,
    parse_severity,
)


@dataclass
class CheckResult:
    """单个 QC check 的统一输出结构。"""

    check_id: str
    evaluation_state: str
    finding: str | None = None
    severity: str = Severity.NONE.value
    decision_effect: str | None = None
    score: float | None = None
    thresholds: dict[str, Any] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reason_code: str | None = None
    source: str = EvidenceSource.CONTRACT_SKELETON.value

    def validate(self) -> None:
        EvaluationState(self.evaluation_state)
        parse_severity(self.severity)
        if self.decision_effect is not None:
            parse_decision(self.decision_effect)
        if self.reason_code is not None:
            parse_reason_code(self.reason_code)
        if self.evaluation_state != EvaluationState.EVALUATED.value:
            # 未评估不得伪装成正常 finding / PASS effect
            if self.finding is not None:
                raise ValueError(
                    f"{self.check_id}: not_evaluated/unavailable cannot have finding"
                )
            if self.decision_effect == Decision.PASS.value:
                raise ValueError(
                    f"{self.check_id}: not_evaluated cannot have decision_effect=pass"
                )
        if self.decision_effect == Decision.RETAKE.value and self.reason_code is None:
            raise ValueError(f"{self.check_id}: RETAKE requires reason_code")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class InputGuardResult:
    """最终 Input Guard 结果。"""

    decision: str
    usable: bool
    evaluation_complete: bool
    checks: dict[str, CheckResult]
    reason_codes: list[str] = field(default_factory=list)
    primary_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    retake_guidance: list[dict[str, str]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    segmentation_reference: dict[str, Any] = field(default_factory=dict)
    quality_confidence: float | None = None
    contract_version: str = INPUT_GUARD_CONTRACT_VERSION
    guard_ready: bool = False
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        decision = parse_decision(self.decision)
        if decision == Decision.PASS and self.usable is not True:
            raise ValueError("PASS requires usable=true")
        if decision == Decision.WARNING and self.usable is not True:
            raise ValueError("WARNING requires usable=true")
        if decision == Decision.RETAKE and self.usable is not False:
            raise ValueError("RETAKE requires usable=false")
        if self.quality_confidence is not None:
            if not (0.0 <= float(self.quality_confidence) <= 1.0):
                raise ValueError("quality_confidence must be in [0,1] or null")
        for reason in self.reason_codes:
            parse_reason_code(reason)
        if self.primary_reason is not None:
            parse_reason_code(self.primary_reason)
            if self.primary_reason not in self.reason_codes:
                raise ValueError("primary_reason must be in reason_codes")
        for check_id, check in self.checks.items():
            if check.check_id != check_id:
                raise ValueError(
                    f"check key mismatch: key={check_id} check_id={check.check_id}"
                )
            check.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "decision": self.decision,
            "usable": self.usable,
            "evaluation_complete": self.evaluation_complete,
            "guard_ready": self.guard_ready,
            "quality_confidence": self.quality_confidence,
            "primary_reason": self.primary_reason,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "retake_guidance": list(self.retake_guidance),
            "checks": {key: value.to_dict() for key, value in self.checks.items()},
            "features": dict(self.features),
            "segmentation_reference": dict(self.segmentation_reference),
            "contract_version": self.contract_version,
            "notes": list(self.notes),
        }


def make_not_evaluated_check(
    check_id: str,
    *,
    reason: str = "implementation_pending",
    source: str = EvidenceSource.CONTRACT_SKELETON.value,
) -> CheckResult:
    """未实现 / 依赖缺失时的标准占位结果。"""
    return CheckResult(
        check_id=check_id,
        evaluation_state=EvaluationState.NOT_EVALUATED.value,
        finding=None,
        severity=Severity.NONE.value,
        decision_effect=None,
        score=None,
        thresholds=None,
        evidence={"reason": reason},
        reason_code=None,
        source=source,
    )
