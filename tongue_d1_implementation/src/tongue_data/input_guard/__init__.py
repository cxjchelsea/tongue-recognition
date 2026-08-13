"""D4 Input Guard：图像质量门禁契约与信号规则运行时。"""

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
    defined_checks_count,
    implemented_checks_count,
)
from .policy import InputGuardPolicy, load_input_guard_policy
from .runtime import InputGuardRuntime, UnifiedInputGuard
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
    "InputGuardRuntime",
    "UnifiedInputGuard",
    "ReasonCode",
    "Severity",
    "aggregate_decision",
    "build_contract_skeleton_result",
    "build_result_from_check_effects",
    "defined_checks_count",
    "features_from_segmentation_result",
    "implemented_checks_count",
    "load_input_guard_policy",
    "validate_input_guard_contract",
]
