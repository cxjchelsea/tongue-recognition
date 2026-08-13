"""D4-C：quality.stain_suspected baseline（非 coating.color）。"""

from .calibrate import calibrate_dual_thresholds, load_frozen_thresholds
from .config import StainDataConfig, StainTrainConfig
from .detector import StainDetector
from .manifest import STAIN_CONTRACT_VERSION, build_stain_base_frame, run_d3e_roi_preflight
from .model import build_stain_model

__all__ = [
    "STAIN_CONTRACT_VERSION",
    "StainDataConfig",
    "StainDetector",
    "StainTrainConfig",
    "build_stain_base_frame",
    "build_stain_model",
    "calibrate_dual_thresholds",
    "load_frozen_thresholds",
    "run_d3e_roi_preflight",
]
