"""D4-C.1-B：source dual-view loader + external unlabeled consistency loader。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import StainDataConfig, StainTrainConfig
from .style_augment import apply_style_transform, load_style_contract
from .transforms import preprocess_masked_roi


def _stable_seed(sample_id: str, base_seed: int, salt: int = 0) -> int:
    digest = hashlib.md5(f"{sample_id}:{salt}".encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


def _load_roi(rgb_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    roi_rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    roi_mask = (np.asarray(Image.open(mask_path)) > 0).astype(np.uint8)
    return roi_rgb, roi_mask


class SourceDualViewDataset(Dataset):
    """Stained gold：weak view + style-strong view；共享同一 label。"""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        data_config: StainDataConfig | str | Path,
        train_config: StainTrainConfig | str | Path,
        style_contract: dict[str, Any] | str | Path,
        split: str = "train",
        *,
        disable_style: bool = False,
        subset_ids: list[str] | None = None,
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
        self.disable_style = bool(disable_style)
        self.split = split
        self.seed = train_config.seed

        frame = (
            pd.read_parquet(manifest)
            if isinstance(manifest, (str, Path))
            else manifest.copy()
        )
        frame = frame[(frame["split"] == split) & (frame["eligible"] == True)].copy()
        if subset_ids is not None:
            frame = frame[frame["sample_id"].isin(subset_ids)]
        if frame.empty:
            raise ValueError("empty source dual-view dataset")
        # 必须有 gold stain label
        if frame["label"].isna().any():
            raise ValueError("source samples require gold stain labels")
        self.frame = frame.sort_values("sample_id").reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[int(index)]
        sample_id = str(row["sample_id"])
        roi_rgb, roi_mask = _load_roi(str(row["roi_rgb_path"]), str(row["roi_mask_path"]))
        label = float(row["label"])
        geom_cfg = self.train_config.augmentation.get("train", {})
        rng_weak = np.random.default_rng(_stable_seed(sample_id, self.seed, 1))
        rng_style = np.random.default_rng(_stable_seed(sample_id, self.seed, 2))

        # weak：标准 black-mask + 几何（无 acquisition-style）
        tensor_weak = preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            self.data_config,
            split="train" if self.split == "train" else "val",
            rng=rng_weak if self.split == "train" else None,
            augment_cfg=geom_cfg if self.split == "train" else None,
        )

        # style：先做 acquisition-style，再标准 preprocess + 几何
        style_rgb = roi_rgb
        if not self.disable_style:
            style_rgb, _params = apply_style_transform(
                roi_rgb, self.style_contract, rng_style, strength="moderate"
            )
        tensor_style = preprocess_masked_roi(
            style_rgb,
            roi_mask,
            self.data_config,
            split="train" if self.split == "train" else "val",
            rng=rng_style if self.split == "train" else None,
            augment_cfg=geom_cfg if self.split == "train" else None,
        )
        return {
            "image_weak": torch.from_numpy(np.ascontiguousarray(tensor_weak)),
            "image_style": torch.from_numpy(np.ascontiguousarray(tensor_style)),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": sample_id,
            "dataset": "stained_coating",
            "split": self.split,
            "has_gold_label": True,
        }


def apply_mask_black(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(roi_rgb, dtype=np.uint8).copy()
    out[np.asarray(roi_mask) <= 0] = 0
    return out


class ExternalConsistencyDataset(Dataset):
    """BioHit/TongueSet3 unlabeled：weak + style views；无 label。"""

    def __init__(
        self,
        roi_index: pd.DataFrame | str | Path,
        data_config: StainDataConfig | str | Path,
        train_config: StainTrainConfig | str | Path,
        style_contract: dict[str, Any] | str | Path,
        split: str = "train",
        *,
        datasets: tuple[str, ...] = ("biohit", "tongueset3"),
        subset_ids: list[str] | None = None,
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

        frame = (
            pd.read_parquet(roi_index)
            if isinstance(roi_index, (str, Path))
            else roi_index.copy()
        )
        frame = frame[
            (frame["split"] == split) & (frame["dataset"].isin(list(datasets)))
        ].copy()
        frame = frame[frame["roi_rgb_path"].notna()]
        if subset_ids is not None:
            frame = frame[frame["sample_id"].isin(subset_ids)]
        if frame.empty:
            raise ValueError("empty external consistency dataset")
        # 硬禁：不得携带 stain gold
        if "label" in frame.columns and frame["label"].notna().any():
            raise ValueError("external dataset must not carry stain labels")
        if "true_stain_label" in frame.columns and frame["true_stain_label"].notna().any():
            raise ValueError("external true_stain_label must be null")
        self.frame = frame.sort_values("sample_id").reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[int(index)]
        sample_id = str(row["sample_id"])
        roi_rgb, roi_mask = _load_roi(str(row["roi_rgb_path"]), str(row["roi_mask_path"]))
        rng_a = np.random.default_rng(_stable_seed(sample_id, self.seed, 11))
        rng_b = np.random.default_rng(_stable_seed(sample_id, self.seed, 12))
        geom_cfg = {
            "horizontal_flip": True,
            "rotation_degrees": 5,
            "scale_min": 0.95,
            "scale_max": 1.05,
            "translate_frac": 0.02,
        }
        # view A：minimal geometry
        tensor_a = preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            self.data_config,
            split="train",
            rng=rng_a,
            augment_cfg=geom_cfg,
        )
        # view B：style + mild geometry
        style_rgb, _params = apply_style_transform(
            roi_rgb, self.style_contract, rng_b, strength="moderate"
        )
        tensor_b = preprocess_masked_roi(
            style_rgb,
            roi_mask,
            self.data_config,
            split="train",
            rng=rng_b,
            augment_cfg=geom_cfg,
        )
        return {
            "image_weak": torch.from_numpy(np.ascontiguousarray(tensor_a)),
            "image_style": torch.from_numpy(np.ascontiguousarray(tensor_b)),
            "sample_id": sample_id,
            "dataset": str(row["dataset"]),
            "split": self.split,
            "has_gold_label": False,
            # 明确不提供 label 字段给训练
        }


class BalancedDomainBatchSampler(Sampler[list[int]]):
    """external batch 内 BioHit/TongueSet3 各半。"""

    def __init__(
        self,
        datasets: list[str],
        batch_size: int,
        *,
        biohit_fraction: float = 0.5,
        seed: int = 20260813,
        drop_last: bool = True,
    ):
        self.datasets = list(datasets)
        self.batch_size = int(batch_size)
        self.biohit_fraction = float(biohit_fraction)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.bio_indices = [i for i, name in enumerate(self.datasets) if name == "biohit"]
        self.ts_indices = [
            i for i, name in enumerate(self.datasets) if name == "tongueset3"
        ]
        if not self.bio_indices or not self.ts_indices:
            raise ValueError("need both biohit and tongueset3 for balanced sampler")
        self.n_bio = max(1, int(round(self.batch_size * self.biohit_fraction)))
        self.n_ts = self.batch_size - self.n_bio
        self.num_batches = min(
            len(self.bio_indices) // self.n_bio,
            len(self.ts_indices) // self.n_ts,
        )
        if self.num_batches <= 0:
            raise ValueError("not enough samples for balanced batches")

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        bio = np.array(self.bio_indices)
        ts3 = np.array(self.ts_indices)
        rng.shuffle(bio)
        rng.shuffle(ts3)
        for batch_index in range(self.num_batches):
            bio_batch = bio[batch_index * self.n_bio : (batch_index + 1) * self.n_bio]
            ts_batch = ts3[batch_index * self.n_ts : (batch_index + 1) * self.n_ts]
            batch = np.concatenate([bio_batch, ts_batch])
            rng.shuffle(batch)
            yield batch.tolist()


def collate_source(batch: list[dict]) -> dict:
    return {
        "image_weak": torch.stack([item["image_weak"] for item in batch], dim=0),
        "image_style": torch.stack([item["image_style"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "has_gold_label": True,
    }


def collate_external(batch: list[dict]) -> dict:
    # 确保无 label 键
    for item in batch:
        if "label" in item:
            raise RuntimeError("external batch must not contain label")
    return {
        "image_weak": torch.stack([item["image_weak"] for item in batch], dim=0),
        "image_style": torch.stack([item["image_style"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "has_gold_label": False,
    }


def create_source_loader(
    manifest: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path | dict,
    *,
    split: str = "train",
    batch_size: int = 32,
    disable_style: bool = False,
    subset_ids: list[str] | None = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SourceDualViewDataset(
        manifest,
        data_config,
        train_config,
        style_contract,
        split=split,
        disable_style=disable_style,
        subset_ids=subset_ids,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_source,
    )


def create_external_loader(
    roi_index: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    style_contract: str | Path | dict,
    *,
    split: str = "train",
    batch_size: int = 32,
    biohit_fraction: float = 0.5,
    subset_ids: list[str] | None = None,
) -> DataLoader:
    dataset = ExternalConsistencyDataset(
        roi_index,
        data_config,
        train_config,
        style_contract,
        split=split,
        subset_ids=subset_ids,
    )
    sampler = BalancedDomainBatchSampler(
        datasets=dataset.frame["dataset"].astype(str).tolist(),
        batch_size=batch_size,
        biohit_fraction=biohit_fraction,
        seed=StainTrainConfig(train_config).seed
        if not isinstance(train_config, StainTrainConfig)
        else train_config.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_external,
    )
