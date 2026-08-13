"""TongueSegmentationDataset / DataLoader 工厂。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SegmentationConfig
from .mask_ops import load_image_rgb, load_mask_raw, normalize_binary_mask
from .reproducibility import dataloader_worker_init_fn, seed_everything
from .transforms import preprocess_pair


def _stable_sample_seed(sample_id: str, base_seed: int) -> int:
    """跨进程稳定的样本种子（避免 Python hash 随机化）。"""
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


class TongueSegmentationDataset:
    """基于 segmentation_manifest 的分割 Dataset；禁止重新划分。"""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        config: SegmentationConfig | str | Path,
        split: str,
        datasets: list[str] | None = None,
        seed: int | None = None,
    ):
        if isinstance(config, (str, Path)):
            config = SegmentationConfig(config)
        self.config = config
        self.split = str(split)
        self.seed = int(seed if seed is not None else config.seed)

        if isinstance(manifest, (str, Path)):
            frame = pd.read_parquet(manifest)
        else:
            frame = manifest.copy()

        frame = frame[frame["split"].astype(str) == self.split].copy()
        if datasets is not None:
            frame = frame[frame["dataset"].astype(str).isin(datasets)].copy()
        # 稳定顺序，保证可复现
        frame = frame.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"no segmentation samples for split={self.split}")
        self.manifest = frame

    def __len__(self) -> int:
        return int(len(self.manifest))

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[int(index)]
        sample_id = str(row["sample_id"])
        dataset_name = str(row["dataset"])
        image = load_image_rgb(str(row["image_path"]))
        mask = normalize_binary_mask(load_mask_raw(str(row["mask_path"])))

        # 每样本确定性 RNG（train 增广可复现）
        rng = np.random.default_rng(_stable_sample_seed(sample_id, self.seed))
        image_tensor, mask_tensor, geometry = preprocess_pair(
            image, mask, self.config, self.split, rng=rng
        )

        try:
            import torch

            image_out = torch.from_numpy(np.ascontiguousarray(image_tensor))
            mask_out = torch.from_numpy(np.ascontiguousarray(mask_tensor))
        except ImportError as exc:
            raise ImportError("torch is required for TongueSegmentationDataset") from exc

        return {
            "image": image_out,
            "mask": mask_out,
            "sample_id": sample_id,
            "dataset": dataset_name,
            "split": self.split,
            "original_size": (geometry.original_height, geometry.original_width),
            "geometry": {
                "scale": geometry.scale,
                "pad_left": geometry.pad_left,
                "pad_top": geometry.pad_top,
                "pad_right": geometry.pad_right,
                "pad_bottom": geometry.pad_bottom,
                "input_height": geometry.input_height,
                "input_width": geometry.input_width,
            },
            "foreground_ratio": float(row["foreground_ratio"]),
            "md5": str(row["md5"]),
        }


def create_dataloader(
    dataset: TongueSegmentationDataset,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
):
    """创建 DataLoader；val/test 默认不 shuffle。"""
    import torch
    from torch.utils.data import DataLoader

    config = dataset.config
    if batch_size is None:
        batch_size = int(config.training_contract.get("batch_size", 8))
    if num_workers is None:
        num_workers = int(config.training_contract.get("num_workers", 0))
    if shuffle is None:
        shuffle = dataset.split == "train"

    generator = torch.Generator()
    generator.manual_seed(dataset.seed)

    def _worker_init(worker_id: int):
        dataloader_worker_init_fn(worker_id, base_seed=dataset.seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        generator=generator,
    )


def smoke_test_dataset(
    manifest_path: str | Path,
    config_path: str | Path,
    max_batches: int = 1,
) -> dict:
    """对 BioHit/TongueSet3 × train/val/test 做加载冒烟测试。"""
    config = SegmentationConfig(config_path)
    seed_everything(config.seed)
    manifest = pd.read_parquet(manifest_path)
    results = {"ok": True, "checks": []}

    for dataset_name in config.datasets:
        for split_name in ["train", "val", "test"]:
            subset = manifest[
                (manifest["dataset"].astype(str) == dataset_name)
                & (manifest["split"].astype(str) == split_name)
            ]
            if subset.empty:
                results["ok"] = False
                results["checks"].append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "status": "FAIL",
                        "reason": "empty subset",
                    }
                )
                continue
            dataset = TongueSegmentationDataset(
                subset, config, split=split_name, seed=config.seed
            )
            loader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            image = batch["image"]
            mask = batch["mask"]
            expected_h = config.input_height
            expected_w = config.input_width
            status = "PASS"
            reason = "ok"
            # image: [B,C,H,W]  mask: [B,1,H,W]
            if image.ndim != 4 or tuple(image.shape[1:]) != (3, expected_h, expected_w):
                status = "FAIL"
                reason = f"image shape={tuple(image.shape)}"
            if mask.ndim != 4 or tuple(mask.shape[1:]) != (1, expected_h, expected_w):
                status = "FAIL"
                reason = f"mask shape={tuple(mask.shape)}"
            unique = set(mask.unique().detach().cpu().tolist())
            if not unique.issubset({0.0, 1.0}):
                status = "FAIL"
                reason = f"mask unique={unique}"
            if "sample_id" not in batch or "dataset" not in batch:
                status = "FAIL"
                reason = "metadata missing"
            if status != "PASS":
                results["ok"] = False
            results["checks"].append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "status": status,
                    "reason": reason,
                    "batch_image_shape": list(image.shape),
                    "batch_mask_shape": list(mask.shape),
                    "n_samples": int(len(dataset)),
                }
            )
            if max_batches <= 0:
                break
    return results
