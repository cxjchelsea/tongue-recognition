"""Input Guard policy：加载 / 校验 configs/input_guard_v1.yaml。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .ontology import (
    CHECK_DEFINITIONS,
    CheckId,
    Decision,
    ReasonCode,
    parse_check_id,
    parse_decision,
    parse_reason_code,
)


class InputGuardPolicy:
    """集中配置：check 启用、阈值、primary reason 优先级。"""

    def __init__(self, path: str | Path | None = None, doc: dict | None = None):
        if doc is None:
            if path is None:
                raise ValueError("path or doc required")
            raw = Path(path).read_text(encoding="utf-8")
            doc = yaml.safe_load(raw)
        self.path = Path(path) if path is not None else None
        self.doc = dict(doc)
        self.version = str(self.doc.get("version", ""))
        self.policy_version = str(self.doc.get("policy_version", self.version))
        self.decision_order = [
            parse_decision(item) for item in self.doc.get("decision_order", [])
        ]
        self.checks = dict(self.doc.get("checks", {}))
        self.primary_reason_priority = [
            parse_reason_code(item).value
            for item in self.doc.get("primary_reason_priority", [])
        ]
        self.notes = list(self.doc.get("notes", []))
        self.validate()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.version not in {"1.0", "1.1"}:
            errors.append(f"unsupported policy version: {self.version}")

        expected_order = [Decision.PASS, Decision.WARNING, Decision.RETAKE]
        if self.decision_order != expected_order:
            errors.append(
                f"decision_order must be {[d.value for d in expected_order]}"
            )

        for check_key, check_cfg in self.checks.items():
            try:
                check_id = parse_check_id(check_key)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not isinstance(check_cfg, dict):
                errors.append(f"check {check_key}: config must be mapping")
                continue
            if "enabled" not in check_cfg:
                errors.append(f"check {check_key}: missing enabled")
            # ontology 必须存在
            if check_id not in CHECK_DEFINITIONS:
                errors.append(f"check {check_key}: missing ontology definition")

            thresholds = check_cfg.get("thresholds")
            needs_calibration = bool(check_cfg.get("needs_calibration", False))
            implemented = bool(
                (CHECK_DEFINITIONS.get(check_id) or {}).get("implemented", False)
            )
            # policy 1.1：已实现 check 不得仍标记 needs_calibration
            if (
                implemented
                and needs_calibration
                and self.policy_version.startswith("1.1")
            ):
                errors.append(
                    f"check {check_key}: implemented but still needs_calibration "
                    f"under policy {self.policy_version}"
                )
            if thresholds is not None:
                if not isinstance(thresholds, dict):
                    errors.append(f"check {check_key}: thresholds must be mapping")
                else:
                    for threshold_name, threshold_value in thresholds.items():
                        if threshold_value is None and not needs_calibration:
                            # 允许 null 仅当显式 needs_calibration
                            errors.append(
                                f"check {check_key}: threshold {threshold_name} "
                                f"is null but needs_calibration is not true"
                            )

            # 引用的 reason 必须注册
            for reason_key in ("warning_reason", "retake_reason"):
                if reason_key in check_cfg and check_cfg[reason_key] is not None:
                    try:
                        parse_reason_code(check_cfg[reason_key])
                    except ValueError as exc:
                        errors.append(f"check {check_key}: {exc}")

        for reason in self.primary_reason_priority:
            try:
                parse_reason_code(reason)
            except ValueError as exc:
                errors.append(f"primary_reason_priority: {exc}")

        # priority 列表应覆盖 registry 中主要 reason（允许子集，但不允许未知）
        unknown_priority = set(self.primary_reason_priority) - {
            code.value for code in ReasonCode
        }
        if unknown_priority:
            errors.append(
                f"primary_reason_priority unknown: {sorted(unknown_priority)}"
            )

        if errors:
            raise ValueError(
                "input_guard policy validation failed:\n- " + "\n- ".join(errors)
            )
        return []

    def is_check_enabled(self, check: str | CheckId) -> bool:
        check_id = parse_check_id(check)
        short_name = check_id.value.split(".", 1)[1]
        cfg = self.checks.get(short_name) or self.checks.get(check_id.value) or {}
        return bool(cfg.get("enabled", False))

    def check_config(self, check: str | CheckId) -> dict[str, Any]:
        check_id = parse_check_id(check)
        short_name = check_id.value.split(".", 1)[1]
        return dict(
            self.checks.get(short_name)
            or self.checks.get(check_id.value)
            or {}
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.doc)


def load_input_guard_policy(path: str | Path) -> InputGuardPolicy:
    return InputGuardPolicy(path=path)
