"""Segmentation model factory：ResNet34-UNet baseline。"""
from __future__ import annotations

from typing import Any

from .train_config import TrainConfig


ALLOWED_ARCHITECTURES = {"unet"}


def build_segmentation_model(config: TrainConfig | dict[str, Any]):
    """
    构建分割模型；输出 raw logits，不做 sigmoid。
    pretrained 下载失败时 fail-fast，禁止静默降级为 random。
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch is required for D3-B training. "
            "Install: pip install 'tongue-data-contract[train]'"
        ) from exc

    if isinstance(config, TrainConfig):
        model_cfg = config.model
    else:
        model_cfg = dict(config)

    architecture = str(model_cfg.get("architecture", "unet")).lower()
    if architecture not in ALLOWED_ARCHITECTURES:
        raise ValueError(
            f"unsupported architecture={architecture!r}; allowed={sorted(ALLOWED_ARCHITECTURES)}"
        )

    encoder = str(model_cfg.get("encoder", "resnet34"))
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    if encoder_weights in {"", "null", "None"}:
        encoder_weights = None
    in_channels = int(model_cfg.get("in_channels", 3))
    classes = int(model_cfg.get("classes", 1))

    try:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    except Exception as exc:
        # 不静默降级
        raise RuntimeError(
            f"failed to build Unet(encoder={encoder}, weights={encoder_weights}): {exc}"
        ) from exc

    return model


def count_parameters(model) -> dict[str, int]:
    """统计总参数 / 可训练参数。"""
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}


def predict_mask(model, image_tensor, threshold: float = 0.5, device: str | None = None):
    """
    基础推理 helper：preprocessed tensor → probability + binary。
    注意：不负责 unletterbox / 原图 ROI（留给 D3-E）。
    """
    import torch

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probability = torch.sigmoid(logits)
        binary = (probability >= float(threshold)).to(dtype=probability.dtype)
    return {"logits": logits, "probability": probability, "binary": binary}
