"""Input Guard contract / policy / result validators。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ontology import (
    CHECK_DEFINITIONS,
    PHENOTYPE_LABELS_NOT_QC_REASONS,
    ReasonCode,
    assert_not_phenotype_as_qc_reason,
    defined_checks_count,
    implemented_checks_count,
    registered_check_ids,
    registered_reason_codes,
)
from .policy import InputGuardPolicy, load_input_guard_policy
from .schema import InputGuardResult


def validate_reason_registry() -> list[str]:
    errors: list[str] = []
    for code in ReasonCode:
        try:
            assert_not_phenotype_as_qc_reason(code.value)
        except ValueError as exc:
            errors.append(str(exc))
    # 病理标签不得出现在 registry
    registry = {code.value.lower() for code in ReasonCode}
    for phenotype in PHENOTYPE_LABELS_NOT_QC_REASONS:
        if phenotype.lower() in registry:
            errors.append(f"phenotype label registered as QC reason: {phenotype}")
    return errors


def validate_input_guard_contract(
    policy_path: str | Path,
) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []

    registry_errors = validate_reason_registry()
    errors.extend(registry_errors)

    try:
        policy = load_input_guard_policy(policy_path)
    except Exception as exc:
        errors.append(f"policy validation failed: {exc}")
        return errors, warnings

    defined = defined_checks_count()
    implemented = implemented_checks_count()
    if implemented > defined:
        errors.append("implemented_checks_count > defined_checks_count")
    if implemented == 0:
        warnings.append(
            "implemented_checks_count=0 (signal checks not marked implemented)"
        )

    # stain 必须 defined，且不得与病理苔色混淆
    from .ontology import CheckId

    if CheckId.STAIN_SUSPECTED not in CHECK_DEFINITIONS:
        errors.append("stain_suspected missing from ontology")
    if ReasonCode.STAIN_SUSPECTED.value not in registered_reason_codes():
        errors.append("STAIN_SUSPECTED missing from reason registry")

    # D4-E：deferred 合法；enabled 无实现不合法（policy.validate 已覆盖）
    deferred = policy.deferred_check_ids()
    if CheckId.STAIN_SUSPECTED in deferred:
        stain_cfg = policy.check_config(CheckId.STAIN_SUSPECTED)
        if stain_cfg.get("enabled"):
            errors.append("deferred stain must have enabled=false")
        if not stain_cfg.get("deferred_reason"):
            errors.append("deferred stain missing deferred_reason")
    elif policy.policy_version.startswith("1.4"):
        warnings.append("policy 1.4 expected stain deferred")

    # enabled checks must be implemented
    for check_id in policy.active_check_ids():
        if not CHECK_DEFINITIONS[check_id].get("implemented"):
            errors.append(f"enabled check lacks implementation: {check_id.value}")

    if not policy.primary_reason_priority:
        warnings.append("primary_reason_priority is empty")

    _ = registered_check_ids()
    return errors, warnings


def validate_result_payload(payload: dict[str, Any] | InputGuardResult) -> list[str]:
    errors: list[str] = []
    try:
        if isinstance(payload, InputGuardResult):
            payload.validate()
        else:
            # 最小字段检查
            for key in ("decision", "usable", "evaluation_complete", "checks"):
                if key not in payload:
                    errors.append(f"missing field: {key}")
            if not errors:
                # 重建轻量校验
                from .ontology import parse_decision

                decision = parse_decision(payload["decision"])
                usable = bool(payload["usable"])
                if decision.value == "retake" and usable:
                    errors.append("RETAKE requires usable=false")
                if decision.value in {"pass", "warning"} and not usable:
                    errors.append(f"{decision.value} requires usable=true")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def emit_validate(errors: list[str], warnings: list[str]) -> int:
    for item in warnings:
        print(f"[WARN] {item}")
    for item in errors:
        print(f"[ERROR] {item}")
    if not errors:
        print("OK")
    return 1 if errors else 0
