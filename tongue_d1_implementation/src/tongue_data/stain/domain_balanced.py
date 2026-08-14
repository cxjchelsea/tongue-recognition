"""D4-C.1-C：三域均衡采样（stained / biohit / tongueset3 等量）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import StainDataConfig, StainTrainConfig
from .domain_invariant_model import DOMAIN_TO_ID
from .domain_loader import ExternalConsistencyDataset, SourceDualViewDataset, _stable_seed
from .style_augment import load_style_contract
from .transforms import preprocess_masked_roi


class ThreeDomainBalancedSampler(Sampler[list[int]]):
    """
    每个 batch：stained=N, biohit=N, tongueset3=N。
    防止 TongueSet3 数量主导 domain classifier。
    """

    def __init__(
        self,
        domain_names: list[str],
        *,
        per_domain: int,
        seed: int = 20260814,
    ):
        self.domain_names = [str(name) for name in domain_names]
        self.per_domain = int(per_domain)
        self.seed = int(seed)
        self.indices_by_domain: dict[str, list[int]] = {
            "stained": [],
            "biohit": [],
            "tongueset3": [],
        }
        for index, name in enumerate(self.domain_names):
            if name not in self.indices_by_domain:
                raise ValueError(f"unknown domain in sampler: {name}")
            self.indices_by_domain[name].append(index)
        for domain_name, indices in self.indices_by_domain.items():
            if len(indices) < self.per_domain:
                raise ValueError(
                    f"domain {domain_name} has {len(indices)} < per_domain={self.per_domain}"
                )
        self.num_batches = min(
            len(self.indices_by_domain["stained"]) // self.per_domain,
            len(self.indices_by_domain["biohit"]) // self.per_domain,
            len(self.indices_by_domain["tongueset3"]) // self.per_domain,
        )
        if self.num_batches <= 0:
            raise ValueError("not enough samples for three-domain balanced batches")

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pools = {
            key: np.array(value, dtype=np.int64)
            for key, value in self.indices_by_domain.items()
        }
        for key in pools:
            rng.shuffle(pools[key])
        for batch_index in range(self.num_batches):
            start = batch_index * self.per_domain
            end = start + self.per_domain
            batch = np.concatenate(
                [
                    pools["stained"][start:end],
                    pools["biohit"][start:end],
                    pools["tongueset3"][start:end],
                ]
            )
            rng.shuffle(batch)
            # 断言等量
            selected = [self.domain_names[int(index)] for index in batch.tolist()]
            counts = {name: selected.count(name) for name in ("stained", "biohit", "tongueset3")}
            if counts != {
                "stained": self.per_domain,
                "biohit": self.per_domain,
                "tongueset3": self.per_domain,
            }:
                raise RuntimeError(f"unbalanced domain batch: {counts}")
            yield batch.tolist()


class DomainIdentityDataset(Dataset):
    """
    三域单 view（weak）图像 + domain_id。
    stained 可带 gold stain label；external 禁止 stain label。
    """

    def __init__(
        self,
        stain_manifest: pd.DataFrame | str | Path,
        roi_index: pd.DataFrame | str | Path,
        data_config: StainDataConfig | str | Path,
        train_config: StainTrainConfig | str | Path,
        style_contract: dict[str, Any] | str | Path,
        *,
        split: str = "train",
    ):
        if isinstance(data_config, (str, Path)):
            data_config = StainDataConfig(data_config)
        if isinstance(train_config, (str, Path)):
            train_config = StainTrainConfig(train_config)
        if isinstance(style_contract, (str, Path)):
            style_contract = load_style_contract(style_contract)
        self.data_config = data_config
        self.train_config = train_config
        self.style_contract = style_contract
        self.split = split
        self.seed = train_config.seed

        stain = (
            pd.read_parquet(stain_manifest)
            if isinstance(stain_manifest, (str, Path))
            else stain_manifest.copy()
        )
        stain = stain[(stain["split"] == split) & (stain["eligible"] == True)].copy()
        stain["dataset"] = "stained"
        stain["domain_name"] = "stained"
        if stain["label"].isna().any():
            raise ValueError("stained samples require gold labels")

        external = (
            pd.read_parquet(roi_index)
            if isinstance(roi_index, (str, Path))
            else roi_index.copy()
        )
        external = external[
            (external["split"] == split)
            & (external["dataset"].isin(["biohit", "tongueset3"]))
            & external["roi_rgb_path"].notna()
        ].copy()
        if "label" in external.columns and external["label"].notna().any():
            raise ValueError("external must not carry stain labels")
        external["domain_name"] = external["dataset"].astype(str)

        stained_part = stain[
            ["sample_id", "roi_rgb_path", "roi_mask_path", "label", "domain_name", "dataset"]
        ]
        external_part = external[
            ["sample_id", "roi_rgb_path", "roi_mask_path", "domain_name", "dataset"]
        ].copy()
        external_part["label"] = np.nan
        self.frame = (
            pd.concat([stained_part, external_part], ignore_index=True)
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict:
        from .domain_loader import _load_roi

        row = self.frame.iloc[int(index)]
        sample_id = str(row["sample_id"])
        domain_name = str(row["domain_name"])
        roi_rgb, roi_mask = _load_roi(str(row["roi_rgb_path"]), str(row["roi_mask_path"]))
        rng = np.random.default_rng(_stable_seed(sample_id, self.seed, 31))
        geom_cfg = self.train_config.augmentation.get("train", {}) if self.split == "train" else None
        tensor = preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            self.data_config,
            split="train" if self.split == "train" else "val",
            rng=rng if self.split == "train" else None,
            # domain identity batch：不做 random style（style 走 consistency path）
            augment_cfg=geom_cfg if self.split == "train" else None,
        )
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(tensor)),
            "domain_id": torch.tensor(DOMAIN_TO_ID[domain_name], dtype=torch.long),
            "domain_name": domain_name,
            "sample_id": sample_id,
            "dataset": str(row["dataset"]),
            "has_gold_label": domain_name == "stained",
        }
        if domain_name == "stained":
            item["label"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        return item


def collate_domain_identity(batch: list[dict]) -> dict:
    for item in batch:
        if not item["has_gold_label"] and "label" in item:
            raise RuntimeError("external domain sample must not carry stain label")
    out = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in batch], dim=0),
        "domain_name": [item["domain_name"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "has_gold_label": [item["has_gold_label"] for item in batch],
    }
    if any(item["has_gold_label"] for item in batch):
        # 仅 stained 有 label；external 位置填 -1 哨兵且不得进 BCE
        labels = []
        for item in batch:
            if item["has_gold_label"]:
                labels.append(item["label"])
            else:
                labels.append(torch.tensor(-1.0, dtype=torch.float32))
        out["label"] = torch.stack(labels, dim=0)
    return out


def create_three_domain_loader(
    stain_manifest: str | Path,
    roi_index: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path | dict,
    *,
    split: str = "train",
    per_domain: int = 8,
) -> DataLoader:
    dataset = DomainIdentityDataset(
        stain_manifest,
        roi_index,
        data_config,
        train_config,
        style_contract,
        split=split,
    )
    train_cfg = (
        train_config
        if isinstance(train_config, StainTrainConfig)
        else StainTrainConfig(train_config)
    )
    sampler = ThreeDomainBalancedSampler(
        domain_names=dataset.frame["domain_name"].astype(str).tolist(),
        per_domain=per_domain,
        seed=train_cfg.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_domain_identity,
    )


def dataset_names_to_domain_ids(dataset_names: list[str]) -> torch.Tensor:
    """biohit/tongueset3/stained_coating → domain id。"""
    ids = []
    for name in dataset_names:
        key = str(name).lower()
        if key in {"stained", "stained_coating"}:
            ids.append(DOMAIN_TO_ID["stained"])
        elif key == "biohit":
            ids.append(DOMAIN_TO_ID["biohit"])
        elif key == "tongueset3":
            ids.append(DOMAIN_TO_ID["tongueset3"])
        else:
            raise ValueError(f"cannot map dataset to domain: {name}")
    return torch.tensor(ids, dtype=torch.long)


# 复用 v2 dual-view 构造，供类型检查引用
__all__ = [
    "ThreeDomainBalancedSampler",
    "DomainIdentityDataset",
    "create_three_domain_loader",
    "dataset_names_to_domain_ids",
    "SourceDualViewDataset",
    "ExternalConsistencyDataset",
]
