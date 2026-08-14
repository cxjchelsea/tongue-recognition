"""D4-A：QC Ontology / Reason / Severity / Decision 注册表。

Input Guard 只评价图像是否适合视觉分析，不评价健康/舌象病理表型。
"""
from __future__ import annotations

from enum import Enum
from typing import Any


INPUT_GUARD_CONTRACT_VERSION = "1.1"
QC_ONTOLOGY_VERSION = "1.0"
REASON_REGISTRY_VERSION = "1.0"


class Decision(str, Enum):
    """用户采集流程动作状态（不用 FAIL）。"""

    PASS = "pass"
    WARNING = "warning"
    RETAKE = "retake"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class EvaluationState(str, Enum):
    """检查是否真正执行；缺失信息 ≠ 正常。"""

    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"
    UNAVAILABLE = "unavailable"


class EvidenceSource(str, Enum):
    SEGMENTATION_METADATA = "segmentation_metadata"
    SIGNAL_RULE = "signal_rule"
    LEARNED_MODEL = "learned_model"
    MANUAL = "manual"
    DERIVED = "derived"
    CONTRACT_SKELETON = "contract_skeleton"


class ReasonCode(str, Enum):
    NO_TONGUE_DETECTED = "NO_TONGUE_DETECTED"
    TONGUE_TOO_SMALL = "TONGUE_TOO_SMALL"
    TONGUE_SLIGHTLY_SMALL = "TONGUE_SLIGHTLY_SMALL"
    TONGUE_CROPPED = "TONGUE_CROPPED"
    TONGUE_TOUCHES_FRAME = "TONGUE_TOUCHES_FRAME"
    SEGMENTATION_FRAGMENTED = "SEGMENTATION_FRAGMENTED"
    SEGMENTATION_LOW_CONFIDENCE = "SEGMENTATION_LOW_CONFIDENCE"
    SEGMENTATION_IMPLAUSIBLE = "SEGMENTATION_IMPLAUSIBLE"
    IMAGE_BLUR = "IMAGE_BLUR"
    TONGUE_BLUR = "TONGUE_BLUR"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    HIGHLIGHT_CLIPPING = "HIGHLIGHT_CLIPPING"
    SHADOW_CLIPPING = "SHADOW_CLIPPING"
    UNEVEN_LIGHTING = "UNEVEN_LIGHTING"
    STRONG_SHADOW = "STRONG_SHADOW"
    COLOR_CAST_SUSPECTED = "COLOR_CAST_SUSPECTED"
    SEVERE_COLOR_CAST = "SEVERE_COLOR_CAST"
    TONGUE_OCCLUDED = "TONGUE_OCCLUDED"
    IMAGE_RESOLUTION_TOO_LOW = "IMAGE_RESOLUTION_TOO_LOW"
    TONGUE_RESOLUTION_TOO_LOW = "TONGUE_RESOLUTION_TOO_LOW"
    STAIN_SUSPECTED = "STAIN_SUSPECTED"
    UNKNOWN_QUALITY_ISSUE = "UNKNOWN_QUALITY_ISSUE"


# 病理/表型标签不得作为 QC fail reason（防误用）
PHENOTYPE_LABELS_NOT_QC_REASONS = frozenset(
    {
        "red_tongue",
        "pale_tongue",
        "purple_tongue",
        "crack",
        "cracked",
        "toothmark",
        "teeth_mark",
        "yellow_coating",
        "thick_coating",
        "peeled_coating",
        "白苔",
        "黄苔",
        "红舌",
        "淡舌",
        "紫舌",
        "裂纹",
        "齿痕",
        "剥苔",
        "厚苔",
    }
)


class CheckId(str, Enum):
    TONGUE_PRESENCE = "quality.tongue_presence"
    TONGUE_SCALE = "quality.tongue_scale"
    TONGUE_COMPLETENESS = "quality.tongue_completeness"
    SEGMENTATION_INTEGRITY = "quality.segmentation_integrity"
    FOCUS = "quality.focus"
    EXPOSURE = "quality.exposure"
    ILLUMINATION_UNIFORMITY = "quality.illumination_uniformity"
    COLOR_CAST = "quality.color_cast"
    OCCLUSION = "quality.occlusion"
    RESOLUTION = "quality.resolution"
    STAIN_SUSPECTED = "quality.stain_suspected"


# check_id → 允许的 finding 状态（ontology 层）
CHECK_FINDINGS: dict[CheckId, frozenset[str]] = {
    CheckId.TONGUE_PRESENCE: frozenset({"present", "uncertain", "absent"}),
    CheckId.TONGUE_SCALE: frozenset({"adequate", "small", "too_small"}),
    CheckId.TONGUE_COMPLETENESS: frozenset(
        {"complete", "possibly_cropped", "cropped"}
    ),
    CheckId.SEGMENTATION_INTEGRITY: frozenset(
        {"good", "fragmented", "uncertain", "invalid"}
    ),
    CheckId.FOCUS: frozenset({"sharp", "slightly_blurred", "blurred"}),
    CheckId.EXPOSURE: frozenset(
        {
            "normal",
            "slightly_underexposed",
            "underexposed",
            "slightly_overexposed",
            "overexposed",
        }
    ),
    CheckId.ILLUMINATION_UNIFORMITY: frozenset(
        {"uniform", "mildly_nonuniform", "nonuniform"}
    ),
    CheckId.COLOR_CAST: frozenset({"acceptable", "suspected", "severe"}),
    CheckId.OCCLUSION: frozenset({"none", "minor", "major", "possible_occlusion"}),
    CheckId.RESOLUTION: frozenset({"adequate", "low", "too_low"}),
    CheckId.STAIN_SUSPECTED: frozenset({"false", "true", "uncertain"}),
}


# defined / implemented 分离：D4-B 落地 8 项信号规则
CHECK_DEFINITIONS: dict[CheckId, dict] = {
    CheckId.TONGUE_PRESENCE: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": False,
        "description": "舌体是否存在于画面中",
    },
    CheckId.TONGUE_SCALE: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "舌体尺度是否足够支持表型分析",
    },
    CheckId.TONGUE_COMPLETENESS: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "舌体是否完整 / 是否被画框裁断",
    },
    CheckId.SEGMENTATION_INTEGRITY: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": False,
        "description": "分割结果完整性与可信度",
    },
    CheckId.FOCUS: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "整图与舌 ROI 清晰度",
    },
    CheckId.EXPOSURE: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "曝光是否适合颜色表型",
    },
    CheckId.ILLUMINATION_UNIFORMITY: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "光照均匀性",
    },
    CheckId.COLOR_CAST: {
        "defined": True,
        "implementation_stage": "D4-D",
        "implemented": True,
        "depends_on_roi": True,
        "description": "偏色嫌疑（neutral-reference；禁止舌色捷径）",
    },
    CheckId.OCCLUSION: {
        "defined": True,
        "implementation_stage": "D4-D",
        "implemented": True,
        "depends_on_roi": True,
        "description": "遮挡（多弱证据；不把裂纹/齿痕当遮挡）",
    },
    CheckId.RESOLUTION: {
        "defined": True,
        "implementation_stage": "D4-B",
        "implemented": True,
        "depends_on_roi": True,
        "description": "整图与舌 ROI 有效分辨率",
    },
    CheckId.STAIN_SUSPECTED: {
        "defined": True,
        "implementation_stage": "D4-C",
        "implemented": True,
        "production_supported": False,  # D4-E deferred
        "depends_on_roi": True,
        "description": "外源染苔嫌疑（非病理苔色）；D4-E production deferred",
    },
}


DECISION_ORDER = (Decision.PASS, Decision.WARNING, Decision.RETAKE)
DECISION_RANK = {decision: index for index, decision in enumerate(DECISION_ORDER)}


def parse_decision(value: str | Decision) -> Decision:
    if isinstance(value, Decision):
        return value
    key = str(value).strip().lower()
    try:
        return Decision(key)
    except ValueError as exc:
        raise ValueError(f"unknown decision: {value!r}") from exc


def parse_severity(value: str | Severity) -> Severity:
    if isinstance(value, Severity):
        return value
    key = str(value).strip().lower()
    try:
        return Severity(key)
    except ValueError as exc:
        raise ValueError(f"unknown severity: {value!r}") from exc


def parse_reason_code(value: str | ReasonCode) -> ReasonCode:
    if isinstance(value, ReasonCode):
        return value
    key = str(value).strip().upper()
    try:
        return ReasonCode(key)
    except ValueError as exc:
        raise ValueError(f"unknown reason code: {value!r}") from exc


def parse_check_id(value: str | CheckId) -> CheckId:
    if isinstance(value, CheckId):
        return value
    key = str(value).strip()
    # 允许配置里写短名 tongue_presence
    if not key.startswith("quality."):
        key = f"quality.{key}"
    try:
        return CheckId(key)
    except ValueError as exc:
        raise ValueError(f"unknown QC check: {value!r}") from exc


def assert_not_phenotype_as_qc_reason(reason_like: str) -> None:
    """病理表型标签不得注册为 QC reason。"""
    normalized = str(reason_like).strip().lower().replace(" ", "_")
    if normalized in PHENOTYPE_LABELS_NOT_QC_REASONS:
        raise ValueError(
            f"phenotype label cannot be QC reason: {reason_like!r}"
        )
    # 额外拦截常见表型短语
    blocked_substrings = (
        "yellow_coating",
        "red_tongue",
        "toothmark",
        "crack",
        "peeled_coating",
    )
    for substring in blocked_substrings:
        if substring in normalized:
            raise ValueError(
                f"phenotype concept cannot be QC reason: {reason_like!r}"
            )


def registered_reason_codes() -> list[str]:
    return [code.value for code in ReasonCode]


def registered_check_ids() -> list[str]:
    return [check.value for check in CheckId]


def defined_checks_count() -> int:
    return sum(1 for meta in CHECK_DEFINITIONS.values() if meta["defined"])


def implemented_checks_count() -> int:
    return sum(1 for meta in CHECK_DEFINITIONS.values() if meta["implemented"])


def production_supported_checks_count(policy: Any | None = None) -> int:
    """若给 policy：统计 enabled+production_supported；否则按 ontology。"""
    if policy is None:
        return sum(
            1
            for check_id, meta in CHECK_DEFINITIONS.items()
            if meta.get("implemented") and meta.get("production_supported", True)
        )
    count = 0
    for check_id in CheckId:
        if not policy.is_check_enabled(check_id):
            continue
        cfg = policy.check_config(check_id)
        if cfg.get("production_supported", True):
            count += 1
    return count

