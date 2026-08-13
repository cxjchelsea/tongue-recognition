"""ResNet18 stain binary classifier：输出 raw logit。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_stain_model(model_cfg: dict[str, Any]) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "resnet18")).lower()
    if architecture != "resnet18":
        raise ValueError(f"D4-C baseline only supports resnet18, got {architecture}")
    weights_name = model_cfg.get("encoder_weights", "imagenet")
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError("torchvision required for stain ResNet18") from exc

    if weights_name in {None, "", "null", "None"}:
        weights = None
    elif str(weights_name).lower() == "imagenet":
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"unsupported encoder_weights: {weights_name}")

    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(f"failed to build resnet18(weights={weights_name}): {exc}") from exc

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(model_cfg.get("classes", 1)))
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}


def predict_probability(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """image [B,3,H,W] → probability [B]。"""
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return torch.sigmoid(logits.float())
