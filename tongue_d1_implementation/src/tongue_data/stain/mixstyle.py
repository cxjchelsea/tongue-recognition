"""MixStyle：feature-level channel-wise mean/std randomization（非 RGB ColorJitter）。"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """
    对 feature map [B,C,H,W] 做 Beta 混合的 mean/std。
    train only；eval 恒等。
    支持 cross-domain：若提供 domain_ids，则优先与不同 domain 配对。
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self._activated = True

    def set_activated(self, activated: bool) -> None:
        self._activated = bool(activated)

    def forward(
        self,
        feature: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (not self.training) or (not self._activated) or self.p <= 0:
            return feature
        if torch.rand(1).item() > self.p:
            return feature
        batch_size = feature.size(0)
        if batch_size < 2:
            return feature

        feature_float = feature.float()
        mu = feature_float.mean(dim=[2, 3], keepdim=True)
        var = feature_float.var(dim=[2, 3], keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-6)
        normalized = (feature_float - mu) / sigma

        perm = self._permutation(batch_size, domain_ids, feature.device)
        mu2 = mu[perm]
        sigma2 = sigma[perm]
        # Beta(alpha, alpha) → [0,1]
        lam = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample((batch_size, 1, 1, 1))
            .to(feature.device)
        )
        mu_mix = lam * mu + (1.0 - lam) * mu2
        sigma_mix = lam * sigma + (1.0 - lam) * sigma2
        out = normalized * sigma_mix + mu_mix
        return out.type_as(feature)

    def _permutation(
        self,
        batch_size: int,
        domain_ids: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if domain_ids is None:
            return torch.randperm(batch_size, device=device)
        domain_ids = domain_ids.view(-1)
        perm = torch.arange(batch_size, device=device)
        # 尽量与不同 domain 配对：随机打乱后若不跨域则再打乱一次
        shuffled = torch.randperm(batch_size, device=device)
        same = domain_ids == domain_ids[shuffled]
        if bool(same.any()) and batch_size >= 3:
            shuffled2 = torch.randperm(batch_size, device=device)
            # 对仍同域位置尝试用 shuffled2 替换
            for index in torch.where(same)[0].tolist():
                candidate = int(shuffled2[index].item())
                if int(domain_ids[index].item()) != int(domain_ids[candidate].item()):
                    shuffled[index] = candidate
        return shuffled


class MixStyleHookManager:
    """在 ResNet layer 输出注册 MixStyle。"""

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[str],
        *,
        p: float,
        alpha: float,
    ):
        self.mix = MixStyle(p=p, alpha=alpha)
        self.handles = []
        self._domain_ids: torch.Tensor | None = None
        layer_map = {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }
        for name in layers:
            if name not in layer_map:
                raise ValueError(f"unsupported MixStyle layer: {name}")
            handle = layer_map[name].register_forward_hook(self._make_hook())
            self.handles.append(handle)

    def _make_hook(self):
        def _hook(_module, _inputs, output):
            return self.mix(output, self._domain_ids)

        return _hook

    def set_domain_ids(self, domain_ids: torch.Tensor | None) -> None:
        self._domain_ids = domain_ids

    def train_mode(self, enabled: bool = True) -> None:
        self.mix.train(enabled)
        self.mix.set_activated(enabled)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
