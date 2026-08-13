"""D4-C.1-A：frozen ResNet18 embedding / Grad-CAM / logit（只读）。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def forward_logit_prob(
    model: torch.nn.Module, batch: torch.Tensor
) -> tuple[float, float]:
    """返回 (logit, probability)。"""
    model.eval()
    logits = model(batch)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logit = float(logits[0, 0].detach().cpu())
    else:
        logit = float(logits.reshape(-1)[0].detach().cpu())
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return logit, prob


@torch.inference_mode()
def extract_embedding(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    """在 classifier(fc) 前提取 global pooled embedding [512]。"""
    model.eval()
    # ResNet: 逐层到 avgpool
    x = batch
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    feat = torch.flatten(x, 1)
    return feat[0].detach().float().cpu().numpy()


def grad_cam_resnet18(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> np.ndarray:
    """
    Grad-CAM on layer4；返回 [H,W] 归一化 heatmap（与输入空间对齐）。
    允许梯度，但不更新参数。
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)
    try:
        batch = batch.detach().requires_grad_(True)
        logits = model(batch)
        score = logits.reshape(-1)[0]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        acts = activations["value"]  # [1,C,h,w]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(batch.shape[2], batch.shape[3]), mode="bilinear", align_corners=False
        )
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        denom = cam_np.max() + 1e-8
        cam_np = cam_np / denom
        return cam_np.astype(np.float32)
    finally:
        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)


def cam_region_ratios(
    cam: np.ndarray,
    roi_mask: np.ndarray,
    input_size: int,
    *,
    pad_top: int,
    pad_left: int,
    new_height: int,
    new_width: int,
) -> dict[str, float]:
    """量化 CAM 能量在 tongue / boundary / fill / padding。"""
    from .transforms import letterbox_rgb

    mask = (np.asarray(roi_mask) > 0).astype(np.uint8) * 255
    mask_lb = letterbox_rgb(
        np.stack([mask, mask, mask], axis=-1), input_size, fill_value=0
    )[..., 0]
    fore = mask_lb > 0
    # 边界环
    try:
        from scipy import ndimage

        eroded = ndimage.binary_erosion(fore, iterations=2)
        boundary = fore & (~eroded)
    except Exception:
        boundary = np.zeros_like(fore)
    padding = np.ones((input_size, input_size), dtype=bool)
    padding[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = False
    content = ~padding
    fill = content & (~fore)
    energy = cam.astype(np.float64)
    total = float(energy.sum()) + 1e-8
    inside = fore & (~boundary)
    return {
        "inside_ratio": float(energy[inside].sum() / total) if inside.any() else 0.0,
        "boundary_ratio": float(energy[boundary].sum() / total) if boundary.any() else 0.0,
        "background_ratio": float(energy[fill].sum() / total) if fill.any() else 0.0,
        "padding_ratio": float(energy[padding].sum() / total) if padding.any() else 0.0,
        "energy_sum": float(energy.sum()),
    }


def centroid_distances(embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """组间 centroid L2 距离。"""
    centers = {
        key: values.mean(axis=0) for key, values in embeddings.items() if len(values)
    }
    out: dict[str, float] = {}
    keys = list(centers.keys())
    for index_i, key_i in enumerate(keys):
        for key_j in keys[index_i + 1 :]:
            dist = float(np.linalg.norm(centers[key_i] - centers[key_j]))
            out[f"{key_i}__{key_j}"] = dist
    return out
