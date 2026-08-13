"""Retake guidance registry：reason → 采集建议（非疾病判断）。"""
from __future__ import annotations

from .ontology import ReasonCode, parse_reason_code

# 与 reason 解耦；仅采集建议，不含证候/疾病措辞
RETAKE_GUIDANCE: dict[ReasonCode, str] = {
    ReasonCode.NO_TONGUE_DETECTED: (
        "请将舌头完整伸出并置于画面中央后重新拍摄。"
    ),
    ReasonCode.TONGUE_TOO_SMALL: (
        "请将相机适当靠近，使舌头在画面中占据更大区域。"
    ),
    ReasonCode.TONGUE_SLIGHTLY_SMALL: (
        "建议稍微靠近拍摄，使舌头区域更清晰可见。"
    ),
    ReasonCode.TONGUE_CROPPED: (
        "请确保舌尖和两侧舌缘完整出现在画面内。"
    ),
    ReasonCode.TONGUE_TOUCHES_FRAME: (
        "请调整构图，避免舌体贴边或被画框裁切。"
    ),
    ReasonCode.SEGMENTATION_FRAGMENTED: (
        "请重新拍摄清晰、单主体的伸舌照片，避免复杂背景干扰。"
    ),
    ReasonCode.SEGMENTATION_LOW_CONFIDENCE: (
        "请重新对焦并保证舌体清晰完整后拍摄。"
    ),
    ReasonCode.SEGMENTATION_IMPLAUSIBLE: (
        "当前舌体区域检测不可靠，请按规范重新拍摄。"
    ),
    ReasonCode.IMAGE_BLUR: (
        "请保持手机稳定并重新对焦后拍摄。"
    ),
    ReasonCode.TONGUE_BLUR: (
        "请保持手机稳定并重新对焦后拍摄。"
    ),
    ReasonCode.UNDEREXPOSED: (
        "请在更明亮、均匀的环境中重新拍摄。"
    ),
    ReasonCode.OVEREXPOSED: (
        "请避免强光直射或闪光灯造成舌面过曝。"
    ),
    ReasonCode.HIGHLIGHT_CLIPPING: (
        "请降低过亮光源或避免闪光灯直射舌面后重拍。"
    ),
    ReasonCode.SHADOW_CLIPPING: (
        "请增加均匀照明，避免舌面大面积过暗。"
    ),
    ReasonCode.UNEVEN_LIGHTING: (
        "请使用更均匀的照明，避免一侧过亮一侧过暗。"
    ),
    ReasonCode.STRONG_SHADOW: (
        "请调整光源角度，减少舌面强阴影后重新拍摄。"
    ),
    ReasonCode.COLOR_CAST_SUSPECTED: (
        "请尽量使用自然或中性白光，避免彩色灯光环境。"
    ),
    ReasonCode.SEVERE_COLOR_CAST: (
        "请在中性照明环境下重新拍摄，避免明显彩色光源。"
    ),
    ReasonCode.TONGUE_OCCLUDED: (
        "请确保舌面无遮挡（唇、手指、食物等）后重新拍摄。"
    ),
    ReasonCode.IMAGE_RESOLUTION_TOO_LOW: (
        "请使用更高分辨率相机或靠近拍摄后重试。"
    ),
    ReasonCode.TONGUE_RESOLUTION_TOO_LOW: (
        "请靠近拍摄，使舌头在画面中有足够像素细节。"
    ),
    ReasonCode.STAIN_SUSPECTED: (
        "若刚进食、饮用有色饮品或使用可能染色的物质，"
        "请清洁口腔并间隔一段时间后重新拍摄。"
    ),
    ReasonCode.UNKNOWN_QUALITY_ISSUE: (
        "当前照片质量不适合继续分析，请按规范重新拍摄。"
    ),
}

FALLBACK_GUIDANCE = "当前照片质量不适合继续分析，请按规范重新拍摄。"


def guidance_for_reason(reason: str | ReasonCode) -> str:
    """返回用户可读重拍建议；未知 reason 使用安全 fallback。"""
    try:
        code = parse_reason_code(reason)
    except ValueError:
        return FALLBACK_GUIDANCE
    return RETAKE_GUIDANCE.get(code, FALLBACK_GUIDANCE)


def guidance_list_for_reasons(reasons: list[str | ReasonCode]) -> list[dict]:
    """去重后的 guidance 列表，保持输入顺序。"""
    seen: set[str] = set()
    items: list[dict] = []
    for reason in reasons:
        try:
            code = parse_reason_code(reason)
            key = code.value
        except ValueError:
            key = str(reason)
            text = FALLBACK_GUIDANCE
            if key in seen:
                continue
            seen.add(key)
            items.append({"reason_code": key, "guidance": text})
            continue
        if key in seen:
            continue
        seen.add(key)
        items.append({"reason_code": key, "guidance": guidance_for_reason(code)})
    return items
