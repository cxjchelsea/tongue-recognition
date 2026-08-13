"""D3-A：Tongue Segmentation Dataset & Training Contract。"""

from .builder import SegmentationBuilder
from .validators import validate_segmentation

__all__ = ["SegmentationBuilder", "validate_segmentation"]
