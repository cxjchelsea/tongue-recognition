"""Stain 标签解析：仅 quality.stain_suspected / canonical_label。"""
from __future__ import annotations

from typing import Any


TRUE_VALUES = {"true", "1", "yes", "y", True, 1}
FALSE_VALUES = {"false", "0", "no", "n", False, 0}


def parse_stain_label(canonical_label: Any) -> int:
    """
    返回 1=stained suspected, 0=not。
    未知标签 fail-fast，禁止静默变 negative。
    """
    if canonical_label is None:
        raise ValueError("stain label is null")
    if isinstance(canonical_label, str):
        key = canonical_label.strip().lower()
        if key in {"true", "1", "yes", "y"}:
            return 1
        if key in {"false", "0", "no", "n"}:
            return 0
        raise ValueError(f"unknown stain canonical_label: {canonical_label!r}")
    if canonical_label in TRUE_VALUES:
        return 1
    if canonical_label in FALSE_VALUES:
        return 0
    raise ValueError(f"unknown stain canonical_label: {canonical_label!r}")


def assert_no_coating_color_usage(labels_frame) -> None:
    """防御：stain 流程不得依赖 coating.color。"""
    if "canonical_task" not in labels_frame.columns:
        return
    # 调用方应只传入 stain task 子集；若混入 coating.color 则报错
    tasks = set(labels_frame["canonical_task"].astype(str).unique().tolist())
    if "coating.color" in tasks:
        raise ValueError("stain pipeline must not consume coating.color labels")
