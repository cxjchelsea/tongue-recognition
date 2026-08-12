from .paired_mask import PairedMaskAdapter
from .folder_classification import FolderClassificationAdapter
from .tonguedx import TongueDxAdapter
from .tonguexpert import TonguExpertAdapter
from .tmc import TMCYoloAdapter

ADAPTERS = {
    "paired_mask": PairedMaskAdapter,
    "folder_classification": FolderClassificationAdapter,
    "tonguedx_csv": TongueDxAdapter,
    "tonguexpert": TonguExpertAdapter,
    "tmc_yolo": TMCYoloAdapter,
}
