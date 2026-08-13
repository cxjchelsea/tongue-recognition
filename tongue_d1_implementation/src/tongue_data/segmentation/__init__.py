"""D3：Tongue Segmentation Dataset / Training / Inference。"""

from .builder import SegmentationBuilder
from .validators import validate_segmentation

__all__ = ["SegmentationBuilder", "validate_segmentation"]
