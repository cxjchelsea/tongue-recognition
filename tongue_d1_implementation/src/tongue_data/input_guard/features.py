"""InputGuardFeatures：质量特征契约 + D3-E adapter。

未实现特征必须为 null，禁止用 0 填充缺失值。
QC 几何比率使用 tight bbox（非 ROI margin bbox）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class InputGuardFeatures:
    """D4 Input Guard 特征合同；缺失 = None。"""

    # 图像尺寸
    original_width: int | None = None
    original_height: int | None = None

    # 分割前景（相对整图）
    foreground_ratio: float | None = None
    tongue_pixel_count: int | None = None

    # tight bbox 比率（QC 用）
    bbox_width_ratio: float | None = None
    bbox_height_ratio: float | None = None
    bbox_area_ratio: float | None = None
    bbox_tight: tuple[int, int, int, int] | None = None

    # 边界接触（tight bbox）
    touches_left: bool | None = None
    touches_right: bool | None = None
    touches_top: bool | None = None
    touches_bottom: bool | None = None
    touches_image_border: bool | None = None
    distance_to_border_ratio: float | None = None

    # 分割完整性
    component_count: int | None = None
    largest_component_ratio: float | None = None
    mean_foreground_probability: float | None = None
    max_probability: float | None = None

    # ROI 有效分辨率（margin ROI 尺寸；与 QC bbox 比率分离）
    roi_width_px: int | None = None
    roi_height_px: int | None = None
    roi_area_px: int | None = None
    roi_mask_fill_ratio: float | None = None

    # D4-B 预留信号特征（当前未实现 → None）
    blur_score: float | None = None
    roi_blur_score: float | None = None
    mean_luminance: float | None = None
    dark_pixel_ratio: float | None = None
    bright_pixel_ratio: float | None = None
    highlight_clip_ratio: float | None = None
    shadow_clip_ratio: float | None = None
    illumination_uniformity_score: float | None = None
    color_cast_score: float | None = None

    # 元信息
    segmentation_status: str | None = None
    available_feature_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.bbox_tight is not None:
            payload["bbox_tight"] = list(self.bbox_tight)
        payload["available_feature_names"] = list(self.available_feature_names)
        return payload

    def availability_map(self) -> dict[str, bool]:
        """每个特征是否可用（非 None）。"""
        mapping: dict[str, bool] = {}
        for field_info in fields(self):
            if field_info.name == "available_feature_names":
                continue
            mapping[field_info.name] = getattr(self, field_info.name) is not None
        return mapping


def _tight_bbox_ratios(
    bbox_tight: tuple[int, int, int, int] | None,
    *,
    original_width: int,
    original_height: int,
) -> dict[str, Any]:
    if bbox_tight is None or original_width <= 0 or original_height <= 0:
        return {
            "bbox_width_ratio": None,
            "bbox_height_ratio": None,
            "bbox_area_ratio": None,
            "touches_left": None,
            "touches_right": None,
            "touches_top": None,
            "touches_bottom": None,
            "touches_image_border": None,
            "distance_to_border_ratio": None,
        }
    x1, y1, x2, y2 = (int(value) for value in bbox_tight)
    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    touches_left = x1 <= 0
    touches_top = y1 <= 0
    touches_right = x2 >= original_width
    touches_bottom = y2 >= original_height
    # 到最近边界的归一化距离（中心点）
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dist_left = center_x / original_width
    dist_right = (original_width - center_x) / original_width
    dist_top = center_y / original_height
    dist_bottom = (original_height - center_y) / original_height
    distance_to_border_ratio = float(
        min(dist_left, dist_right, dist_top, dist_bottom)
    )
    return {
        "bbox_width_ratio": float(bbox_width / original_width),
        "bbox_height_ratio": float(bbox_height / original_height),
        "bbox_area_ratio": float(
            (bbox_width * bbox_height) / (original_width * original_height)
        ),
        "touches_left": bool(touches_left),
        "touches_right": bool(touches_right),
        "touches_top": bool(touches_top),
        "touches_bottom": bool(touches_bottom),
        "touches_image_border": bool(
            touches_left or touches_right or touches_top or touches_bottom
        ),
        "distance_to_border_ratio": distance_to_border_ratio,
    }


def features_from_segmentation_result(result: Any) -> InputGuardFeatures:
    """
    将 D3-E TongueSegmentationResult 映射为 InputGuardFeatures。
    不重新跑分割；未提供的 D4-B 信号特征保持 None。
    """
    original_width = getattr(result, "original_width", None)
    original_height = getattr(result, "original_height", None)
    bbox_tight = getattr(result, "bbox_tight", None)
    if bbox_tight is not None:
        bbox_tight = tuple(int(value) for value in bbox_tight)

    ratios = _tight_bbox_ratios(
        bbox_tight,
        original_width=int(original_width or 0),
        original_height=int(original_height or 0),
    )

    roi_size = getattr(result, "roi_size", None)
    roi_width_px = None
    roi_height_px = None
    roi_area_px = None
    if roi_size is not None and len(roi_size) >= 2:
        # roi_size 约定 (width, height)
        roi_width_px = int(roi_size[0])
        roi_height_px = int(roi_size[1])
        roi_area_px = int(roi_width_px * roi_height_px)

    tongue_roi_mask = getattr(result, "tongue_roi_mask", None)
    roi_mask_fill_ratio = None
    if tongue_roi_mask is not None:
        import numpy as np

        mask_array = np.asarray(tongue_roi_mask)
        if mask_array.size > 0:
            roi_mask_fill_ratio = float((mask_array > 0).mean())

    tongue_pixel_count = getattr(result, "mask_foreground_pixels", None)
    if tongue_pixel_count is not None:
        tongue_pixel_count = int(tongue_pixel_count)

    features = InputGuardFeatures(
        original_width=int(original_width) if original_width is not None else None,
        original_height=int(original_height) if original_height is not None else None,
        foreground_ratio=getattr(result, "mask_foreground_ratio", None),
        tongue_pixel_count=tongue_pixel_count,
        bbox_tight=bbox_tight,
        component_count=getattr(result, "component_count", None),
        largest_component_ratio=getattr(result, "largest_component_ratio", None),
        mean_foreground_probability=getattr(
            result, "mean_foreground_probability", None
        ),
        max_probability=getattr(result, "max_probability", None),
        roi_width_px=roi_width_px,
        roi_height_px=roi_height_px,
        roi_area_px=roi_area_px,
        roi_mask_fill_ratio=roi_mask_fill_ratio,
        segmentation_status=getattr(result, "status", None),
        # D4-B 预留保持 None
        blur_score=None,
        roi_blur_score=None,
        mean_luminance=None,
        dark_pixel_ratio=None,
        bright_pixel_ratio=None,
        highlight_clip_ratio=None,
        shadow_clip_ratio=None,
        illumination_uniformity_score=None,
        color_cast_score=None,
        **ratios,
    )
    available = tuple(
        name for name, is_available in features.availability_map().items() if is_available
    )
    features.available_feature_names = available
    return features


FEATURE_FIELD_NAMES: tuple[str, ...] = tuple(
    field_info.name
    for field_info in fields(InputGuardFeatures)
    if field_info.name != "available_feature_names"
)
