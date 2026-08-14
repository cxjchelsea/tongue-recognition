"""D4-C.1-C：ResNet18 + MixStyle hooks + GRL domain head。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .grl import GradientReversal
from .mixstyle import MixStyleHookManager
from .model import build_stain_model


DOMAIN_TO_ID = {"stained": 0, "biohit": 1, "tongueset3": 2}
ID_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_ID.items()}


class DomainInvariantStainModel(nn.Module):
    """
    推理只暴露 stain_logit。
    训练可选：MixStyle + GRL domain head。
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        *,
        mixstyle_cfg: dict[str, Any] | None = None,
        domain_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = build_stain_model(model_cfg)
        self.mixstyle_cfg = dict(mixstyle_cfg or {})
        self.domain_cfg = dict(domain_cfg or {})
        self.mix_hooks: MixStyleHookManager | None = None
        if self.mixstyle_cfg.get("enabled", False):
            # MixStyle 挂到 hook manager，并注册为子模块以便 train/eval 同步
            self.mix_hooks = MixStyleHookManager(
                self.backbone,
                layers=list(self.mixstyle_cfg.get("layers", ["layer1"])),
                p=float(self.mixstyle_cfg.get("p", 0.5)),
                alpha=float(self.mixstyle_cfg.get("alpha", 0.1)),
            )
            self.add_module("mixstyle_module", self.mix_hooks.mix)
        hidden = int(self.domain_cfg.get("hidden", 256))
        dropout = float(self.domain_cfg.get("dropout", 0.1))
        self.grl = GradientReversal(lambda_domain=0.0)
        self.domain_head = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.domain_enabled = bool(self.domain_cfg.get("enabled", False))

    def train(self, mode: bool = True):
        super().train(mode)
        # eval/val/test/inference：强制关闭 MixStyle
        if self.mix_hooks is not None:
            self.mix_hooks.train_mode(bool(mode))
        return self

    def set_grl_lambda(self, lambda_domain: float) -> None:
        self.grl.set_lambda(lambda_domain)

    def set_mixstyle_enabled(self, enabled: bool) -> None:
        if self.mix_hooks is not None:
            self.mix_hooks.train_mode(enabled and self.training)

    def extract_embedding(
        self,
        images: torch.Tensor,
        *,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mix_hooks is not None:
            self.mix_hooks.set_domain_ids(domain_ids if self.training else None)
        x = images
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward_stain(
        self,
        images: torch.Tensor,
        *,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.extract_embedding(images, domain_ids=domain_ids)
        logits = self.backbone.fc(embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return logits

    def forward_domain(
        self,
        images: torch.Tensor,
        *,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.extract_embedding(images, domain_ids=domain_ids)
        reversed_emb = self.grl(embedding)
        return self.domain_head(reversed_emb)

    def forward(
        self,
        images: torch.Tensor,
        *,
        domain_ids: torch.Tensor | None = None,
        return_domain: bool = False,
    ):
        embedding = self.extract_embedding(images, domain_ids=domain_ids)
        stain_logits = self.backbone.fc(embedding)
        if stain_logits.ndim == 2 and stain_logits.shape[1] == 1:
            stain_logits = stain_logits[:, 0]
        if not return_domain:
            return stain_logits
        domain_logits = self.domain_head(self.grl(embedding))
        return stain_logits, domain_logits, embedding


def build_domain_invariant_model(
    train_doc: dict[str, Any],
    *,
    candidate: str,
) -> DomainInvariantStainModel:
    """按预注册 candidate 打开 MixStyle / GRL。"""
    model_cfg = dict(train_doc["model"])
    mix = dict(train_doc.get("mixstyle", {}))
    domain = dict(train_doc.get("domain_adversarial", {}))
    if candidate == "c1":
        mix["enabled"] = True
        domain["enabled"] = False
    elif candidate == "c2":
        mix["enabled"] = False
        domain["enabled"] = True
    elif candidate == "c3":
        mix["enabled"] = True
        domain["enabled"] = True
    else:
        raise ValueError(f"unknown candidate: {candidate}")
    return DomainInvariantStainModel(model_cfg, mixstyle_cfg=mix, domain_cfg=domain)
