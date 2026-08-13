"""D3-B：Trainer / Loss / Checkpoint 基础设施。"""

from .trainer import SegmentationTrainer, run_smoke_training, run_tiny_overfit

__all__ = ["SegmentationTrainer", "run_smoke_training", "run_tiny_overfit"]
