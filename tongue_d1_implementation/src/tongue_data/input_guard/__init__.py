"""D4 Input Guard：图像质量门禁契约与后续规则/模型入口。"""

from .decision import (
    aggregate_decision,
    build_contract_skeleton_result,
    build_result_from_check_effects,
)
from .features import InputGuardFeatures, features_from_segmentation_result
from .ontology import (
    INPUT_GUARD_CONTRACT_VERSION,
    CheckId,
    Decision,
    EvaluationState,
    ReasonCode,
    Severity,
)
from .policy import InputGuardPolicy, load_input_guard_policy
from .schema import CheckResult, InputGuardResult
from .validators import validate_input_guard_contract

__all__ = [
    "INPUT_GUARD_CONTRACT_VERSION",
    "CheckId",
    "CheckResult",
    "Decision",
    "EvaluationState",
    "InputGuardFeatures",
    "InputGuardPolicy",
    "InputGuardResult",
    "ReasonCode",
    "Severity",
    "aggregate_decision",
    "build_contract_skeleton_result",
    "build_result_from_check_effects",
    "features_from_segmentation_result",
    "load_input_guard_policy",
    "validate_input_guard_contract",
]
