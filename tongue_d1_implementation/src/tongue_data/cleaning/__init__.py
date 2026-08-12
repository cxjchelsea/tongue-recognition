"""D2-A：清洗策略、去重合并与监督池分配。"""

from .builder import CleaningBuilder
from .validators import validate_clean

__all__ = ["CleaningBuilder", "validate_clean"]
