"""D3-E 几何工具：letterbox / unletterbox / bbox / ROI。

与 D3-A `letterbox_pair` 使用同一 rounding contract，避免训练-推理漂移。
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxMetadata:
    """Letterbox 正变换元数据，足以做精确 inverse。"""

    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_size": [self.original_width, self.original_height],
            "input_size": [self.input_width, self.input_height],
            "scale": float(self.scale),
            "resized_size": [self.resized_width, self.resized_height],
            "padding": {
                "left": int(self.pad_left),
                "right": int(self.pad_right),
                "top": int(self.pad_top),
                "bottom": int(self.pad_bottom),
            },
        }

    # 兼容 D3-A GeometryMeta 字段命名
    @property
    def original_size(self) -> tuple[int, int]:
        return self.original_width, self.original_height


def compute_letterbox_metadata(
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> LetterboxMetadata:
    """
    与 D3-A 相同的 rounding / padding 分配：
    - scale = min(input_w/orig_w, input_h/orig_h)
    - resized = round(orig * scale)，至少 1
    - pad_left/top = total // 2；right/bottom = total - left/top
    """
    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"invalid original size: {original_width}x{original_height}"
        )
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"invalid input size: {input_width}x{input_height}")

    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    pad_right = input_width - resized_width - pad_left
    pad_bottom = input_height - resized_height - pad_top

    return LetterboxMetadata(
        original_width=int(original_width),
        original_height=int(original_height),
        input_width=int(input_width),
        input_height=int(input_height),
        scale=float(scale),
        resized_width=int(resized_width),
        resized_height=int(resized_height),
        pad_left=int(pad_left),
        pad_right=int(pad_right),
        pad_top=int(pad_top),
        pad_bottom=int(pad_bottom),
    )


def _pil_resample(name: str) -> Image.Resampling:
    key = str(name).lower()
    if key in {"nearest"}:
        return Image.Resampling.NEAREST
    if key in {"bilinear", "linear"}:
        return Image.Resampling.BILINEAR
    if key in {"bicubic"}:
        return Image.Resampling.BICUBIC
    raise ValueError(f"unsupported interpolation: {name}")


def letterbox_image(
    image: np.ndarray,
    input_height: int,
    input_width: int,
    image_interpolation: str = "bilinear",
    pad_value: int = 0,
) -> tuple[np.ndarray, LetterboxMetadata]:
    """RGB uint8 图像 letterbox → 模型输入尺寸。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3, got shape={image.shape}")
    original_height, original_width = image.shape[:2]
    metadata = compute_letterbox_metadata(
        original_width, original_height, input_width, input_height
    )

    image_pil = Image.fromarray(image.astype(np.uint8))
    image_resized = np.asarray(
        image_pil.resize(
            (metadata.resized_width, metadata.resized_height),
            _pil_resample(image_interpolation),
        ),
        dtype=np.uint8,
    )
    canvas = np.full(
        (input_height, input_width, 3), int(pad_value), dtype=np.uint8
    )
    canvas[
        metadata.pad_top : metadata.pad_top + metadata.resized_height,
        metadata.pad_left : metadata.pad_left + metadata.resized_width,
    ] = image_resized
    return canvas, metadata


def letterbox_mask(
    mask: np.ndarray,
    metadata: LetterboxMetadata,
    pad_value: float = 0.0,
) -> np.ndarray:
    """按已有 letterbox metadata 同步处理 mask（NEAREST）。"""
    if mask.shape[:2] != (metadata.original_height, metadata.original_width):
        raise ValueError(
            f"mask shape {mask.shape[:2]} != original "
            f"{metadata.original_height}x{metadata.original_width}"
        )
    mask_pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    mask_resized = np.asarray(
        mask_pil.resize(
            (metadata.resized_width, metadata.resized_height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    mask_resized = (mask_resized > 0).astype(np.float32)
    canvas = np.full(
        (metadata.input_height, metadata.input_width),
        float(pad_value),
        dtype=np.float32,
    )
    canvas[
        metadata.pad_top : metadata.pad_top + metadata.resized_height,
        metadata.pad_left : metadata.pad_left + metadata.resized_width,
    ] = mask_resized
    return canvas


def _crop_letterbox_content(
    model_map: np.ndarray, metadata: LetterboxMetadata
) -> np.ndarray:
    """去掉 letterbox padding，得到 resized 内容区。"""
    if model_map.ndim != 2:
        raise ValueError(f"expected 2D map, got shape={model_map.shape}")
    expected = (metadata.input_height, metadata.input_width)
    if model_map.shape != expected:
        raise ValueError(
            f"model map shape {model_map.shape} != input {expected}"
        )
    return model_map[
        metadata.pad_top : metadata.pad_top + metadata.resized_height,
        metadata.pad_left : metadata.pad_left + metadata.resized_width,
    ]


def unletterbox_probability(
    model_probability: np.ndarray,
    metadata: LetterboxMetadata,
) -> np.ndarray:
    """去 padding + bilinear 恢复到原图尺寸；float32。"""
    cropped = _crop_letterbox_content(
        np.asarray(model_probability, dtype=np.float32), metadata
    )
    if cropped.size == 0:
        raise ValueError("empty cropped probability after removing padding")
    # 将概率映射到 0-255 再 resize，避免 PIL 对 float 模式限制
    cropped_u8 = np.clip(cropped * 255.0, 0, 255).astype(np.uint8)
    restored_u8 = np.asarray(
        Image.fromarray(cropped_u8, mode="L").resize(
            (metadata.original_width, metadata.original_height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    return (restored_u8.astype(np.float32) / 255.0)


def unletterbox_binary_nearest(
    model_binary: np.ndarray,
    metadata: LetterboxMetadata,
) -> np.ndarray:
    """去 padding + NEAREST 恢复 binary mask（策略 B / 调试用）。"""
    cropped = _crop_letterbox_content(
        (np.asarray(model_binary) > 0.5).astype(np.uint8), metadata
    )
    restored = np.asarray(
        Image.fromarray(cropped * 255, mode="L").resize(
            (metadata.original_width, metadata.original_height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    return (restored > 0).astype(np.uint8)


def restore_probability_then_threshold(
    model_probability: np.ndarray,
    metadata: LetterboxMetadata,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    D3-E v1 restoration contract（策略 A）：
    probability → remove pad → bilinear → original size → threshold。
    返回 (original_probability, original_binary_uint8)。
    """
    original_probability = unletterbox_probability(model_probability, metadata)
    original_binary = (original_probability >= float(threshold)).astype(np.uint8)
    return original_probability, original_binary


def mask_to_bbox_xyxy_exclusive(
    binary_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """
    从二值 mask 计算紧包围盒。
    约定：xyxy exclusive → [x1, y1, x2, y2)，可用 image[y1:y2, x1:x2]。
    空 mask → None。
    """
    foreground = np.asarray(binary_mask) > 0
    if not np.any(foreground):
        return None
    rows = np.any(foreground, axis=1)
    cols = np.any(foreground, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y1 = int(y_indices[0])
    y2 = int(y_indices[-1]) + 1
    x1 = int(x_indices[0])
    x2 = int(x_indices[-1]) + 1
    return x1, y1, x2, y2


def expand_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float = 0.05,
) -> tuple[int, int, int, int]:
    """按比例外扩 bbox，并 clip 到图像边界（仍为 exclusive xyxy）。"""
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox: {bbox_xyxy}")
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    margin_x = bbox_width * float(margin_ratio)
    margin_y = bbox_height * float(margin_ratio)
    expanded_x1 = int(np.floor(x1 - margin_x))
    expanded_y1 = int(np.floor(y1 - margin_y))
    expanded_x2 = int(np.ceil(x2 + margin_x))
    expanded_y2 = int(np.ceil(y2 + margin_y))
    expanded_x1 = max(0, expanded_x1)
    expanded_y1 = max(0, expanded_y1)
    expanded_x2 = min(int(image_width), expanded_x2)
    expanded_y2 = min(int(image_height), expanded_y2)
    if expanded_x2 <= expanded_x1 or expanded_y2 <= expanded_y1:
        raise ValueError(
            f"expanded bbox collapsed after clip: "
            f"{(expanded_x1, expanded_y1, expanded_x2, expanded_y2)}"
        )
    return expanded_x1, expanded_y1, expanded_x2, expanded_y2


def crop_roi(array: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """按 exclusive xyxy 裁剪；支持 HxW 或 HxWxC。"""
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    return np.ascontiguousarray(array[y1:y2, x1:x2])


def apply_mask_to_rgb(
    rgb: np.ndarray, binary_mask: np.ndarray, fill_value: int = 0
) -> np.ndarray:
    """非舌区域置 fill_value；不改变舌区 RGB。"""
    if rgb.shape[:2] != binary_mask.shape[:2]:
        raise ValueError("rgb/mask shape mismatch for masked ROI")
    output = rgb.copy()
    output[np.asarray(binary_mask) == 0] = int(fill_value)
    return output


def keep_largest_connected_component(
    binary_mask: np.ndarray,
    *,
    connectivity: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    保留面积最大的前景连通域（4-连通默认）。
    返回 (filtered_mask_uint8, stats)。
    """
    foreground = (np.asarray(binary_mask) > 0).astype(np.uint8)
    height, width = foreground.shape
    total_foreground = int(foreground.sum())
    if total_foreground == 0:
        return foreground, {
            "component_count_before": 0,
            "largest_component_ratio": 0.0,
            "largest_component_area": 0,
            "total_foreground_area": 0,
        }

    visited = np.zeros_like(foreground, dtype=bool)
    if connectivity == 4:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    components: list[list[tuple[int, int]]] = []
    for row_index in range(height):
        for col_index in range(width):
            if foreground[row_index, col_index] == 0 or visited[row_index, col_index]:
                continue
            queue: deque[tuple[int, int]] = deque([(row_index, col_index)])
            visited[row_index, col_index] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and foreground[next_row, next_col] == 1
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            components.append(pixels)

    largest = max(components, key=len)
    filtered = np.zeros_like(foreground)
    for row_index, col_index in largest:
        filtered[row_index, col_index] = 1
    largest_area = int(len(largest))
    return filtered, {
        "component_count_before": int(len(components)),
        "largest_component_ratio": float(largest_area / total_foreground),
        "largest_component_area": largest_area,
        "total_foreground_area": total_foreground,
    }


def bbox_ratios(
    bbox_xyxy: tuple[int, int, int, int] | None,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """为 D4 预留的 bbox / 边界接触 metadata。"""
    if bbox_xyxy is None:
        return {
            "bbox_width_ratio": None,
            "bbox_height_ratio": None,
            "bbox_area_ratio": None,
            "touches_image_border": None,
        }
    x1, y1, x2, y2 = bbox_xyxy
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    touches = bool(x1 <= 0 or y1 <= 0 or x2 >= image_width or y2 >= image_height)
    return {
        "bbox_width_ratio": float(width / image_width) if image_width else None,
        "bbox_height_ratio": float(height / image_height) if image_height else None,
        "bbox_area_ratio": float((width * height) / (image_width * image_height))
        if image_width and image_height
        else None,
        "touches_image_border": touches,
    }


def metadata_as_plain_dict(metadata: LetterboxMetadata) -> dict[str, Any]:
    """扁平字段 + 嵌套 dict，便于 JSON。"""
    payload = asdict(metadata)
    payload.update(metadata.to_dict())
    return payload
