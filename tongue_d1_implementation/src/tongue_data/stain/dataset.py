"""Stain Dataset：从缓存 original-RGB ROI + mask 读取（预 flight 生成）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import StainDataConfig, StainTrainConfig
from .transforms import preprocess_masked_roi


def _stable_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


class StainRoiDataset(Dataset):
    """eligible stain samples；train 可几何增广，val/test 禁止。"""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        data_config: StainDataConfig | str | Path,
        train_config: StainTrainConfig | str | Path | None,
        split: str,
        *,
        disable_augmentation: bool = False,
        seed: int | None = None,
    ):
        if isinstance(data_config, (str, Path)):
            data_config = StainDataConfig(data_config)
        if isinstance(train_config, (str, Path)):
            train_config = StainTrainConfig(train_config)
        self.data_config = data_config
        self.train_config = train_config
        self.split = str(split)
        self.disable_augmentation = bool(disable_augmentation)
        self.seed = int(seed if seed is not None else (train_config.seed if train_config else 0))

        if isinstance(manifest, (str, Path)):
            frame = pd.read_parquet(manifest)
        else:
            frame = manifest.copy()
        frame = frame[
            (frame["split"].astype(str) == self.split) & (frame["eligible"] == True)
        ].copy()
        if frame.empty:
            raise ValueError(f"no eligible stain samples for split={self.split}")
        self.frame = frame.sort_values("sample_id").reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.frame))

    def _load_roi(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        rgb_path = row.get("roi_rgb_path")
        mask_path = row.get("roi_mask_path")
        if not rgb_path or not mask_path or not Path(str(rgb_path)).exists():
            raise FileNotFoundError(
                f"ROI cache missing for {row['sample_id']}; run stain-preflight first"
            )
        roi_rgb = np.asarray(Image.open(str(rgb_path)).convert("RGB"), dtype=np.uint8)
        roi_mask = (np.asarray(Image.open(str(mask_path))) > 0).astype(np.uint8)
        return roi_rgb, roi_mask

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[int(index)]
        sample_id = str(row["sample_id"])
        roi_rgb, roi_mask = self._load_roi(row)
        if roi_rgb.dtype != np.uint8 or roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
            raise ValueError("roi_rgb must be uint8 HxWx3 from original RGB")

        use_aug = (
            self.split == "train"
            and not self.disable_augmentation
            and self.train_config is not None
        )
        augment_cfg = (
            self.train_config.augmentation.get("train", {}) if use_aug else None
        )
        if self.split in {"val", "test"} and self.train_config is not None:
            if self.train_config.augmentation.get(self.split, {}).get("enabled", False):
                raise ValueError(f"{self.split} augmentation must be disabled")

        rng = np.random.default_rng(_stable_seed(sample_id, self.seed))
        tensor = preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            self.data_config,
            split="train" if use_aug else "val",
            rng=rng if use_aug else None,
            augment_cfg=augment_cfg,
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(tensor)),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": sample_id,
            "split": self.split,
            "md5": str(row["md5"]),
        }


def create_stain_dataloader(
    dataset: StainRoiDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def select_overfit_subset(
    manifest: pd.DataFrame, *, positives: int = 8, negatives: int = 8
) -> pd.DataFrame:
    train = manifest[(manifest["split"] == "train") & (manifest["eligible"] == True)].copy()
    train = train.sort_values("sample_id")
    pos = train[train["label"] == 1].head(int(positives))
    neg = train[train["label"] == 0].head(int(negatives))
    selected = pd.concat([pos, neg], ignore_index=True)
    if len(selected) < positives + negatives:
        raise ValueError("not enough samples for tiny overfit subset")
    return selected.sort_values("sample_id").reset_index(drop=True)
