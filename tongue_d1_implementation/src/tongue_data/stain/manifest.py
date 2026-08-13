"""构建 Stain Manifest：继承 D2 split + D3-E ROI audit。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import StainDataConfig
from .labels import assert_no_coating_color_usage, parse_stain_label

STAIN_CONTRACT_VERSION = "1.0"


def _load_tables(processed_dir: Path, split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(processed_dir / "samples_clean.parquet")
    labels = pd.read_parquet(processed_dir / "labels_clean.parquet")
    splits = pd.read_parquet(split_dir / "split_assignments.parquet")
    return samples, labels, splits


def build_stain_base_frame(
    processed_dir: str | Path,
    split_dir: str | Path,
    data_config: StainDataConfig | str | Path,
) -> pd.DataFrame:
    """仅组装 sample/label/split，不跑 D3。"""
    if isinstance(data_config, (str, Path)):
        data_config = StainDataConfig(data_config)
    processed_dir = Path(processed_dir)
    split_dir = Path(split_dir)
    samples, labels, splits = _load_tables(processed_dir, split_dir)

    stain_samples = samples[samples["dataset"].astype(str) == data_config.dataset].copy()
    if stain_samples.empty:
        raise ValueError(f"no samples for dataset={data_config.dataset}")

    stain_labels = labels[
        (labels["canonical_task"].astype(str) == data_config.task)
        & (labels["sample_id"].isin(stain_samples["sample_id"]))
    ].copy()
    assert_no_coating_color_usage(stain_labels)
    # 防御：不得把全部 labels 传入后误读 coating.color
    other = labels[
        (labels["sample_id"].isin(stain_samples["sample_id"]))
        & (labels["canonical_task"].astype(str) == "coating.color")
    ]
    if len(other):
        raise ValueError("stained_coating must not carry coating.color labels")

    if stain_labels["sample_id"].duplicated().any():
        raise ValueError("duplicate stain labels for sample_id")

    frame = stain_samples.merge(
        stain_labels[["sample_id", "canonical_label", "value", "label_available"]],
        on="sample_id",
        how="inner",
    )
    frame = frame.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")
    if frame["split"].isna().any():
        raise ValueError("missing D2 split for some stain samples")

    # 标签：只用 canonical_label
    frame["label"] = frame["canonical_label"].map(parse_stain_label).astype(int)
    # 校验 value 列不可作为标签源（本数据恒为 1）
    if set(frame["value"].dropna().unique().tolist()) == {1} and set(frame["label"].unique()) != {1}:
        pass  # expected: value unusable

    # leakage checks
    for split_name in ("train", "val", "test"):
        if split_name not in set(frame["split"].astype(str)):
            raise ValueError(f"missing split={split_name} in stain frame")
    train_ids = set(frame.loc[frame.split == "train", "sample_id"])
    val_ids = set(frame.loc[frame.split == "val", "sample_id"])
    test_ids = set(frame.loc[frame.split == "test", "sample_id"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("sample_id leakage across splits")
    train_md5 = set(frame.loc[frame.split == "train", "md5"])
    val_md5 = set(frame.loc[frame.split == "val", "md5"])
    test_md5 = set(frame.loc[frame.split == "test", "md5"])
    if train_md5 & val_md5 or train_md5 & test_md5 or val_md5 & test_md5:
        raise ValueError("md5 leakage across splits")

    frame["contract_version"] = STAIN_CONTRACT_VERSION
    frame["eligible"] = False
    frame["exclusion_reason"] = "roi_pending"
    return frame.sort_values(["split", "sample_id"]).reset_index(drop=True)


def run_d3e_roi_preflight(
    base_frame: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    data_config_path: str | Path,
    train_config_path: str | Path,
    stain_data_config: StainDataConfig | str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """对全部 stain samples 跑 D3-E；生成 manifest + audit。"""
    from tongue_data.segmentation.inference import TongueSegmentationInference, load_rgb_image
    from tongue_data.segmentation.train_config import TrainConfig
    from tongue_data.segmentation.training.checkpoint import load_checkpoint

    if isinstance(stain_data_config, (str, Path)):
        stain_data_config = StainDataConfig(stain_data_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_cache_dir = output_dir / "roi_cache"
    roi_cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/d4")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    d3_hash = ckpt.get("config_hash")
    train_cfg = TrainConfig(train_config_path)
    if d3_hash != train_cfg.config_hash:
        raise ValueError(
            f"D3 config_hash mismatch ckpt={d3_hash} train={train_cfg.config_hash}"
        )

    engine = TongueSegmentationInference(
        checkpoint_path=checkpoint_path,
        data_config=data_config_path,
        train_config=train_config_path,
        device=device,
        return_model_space=False,
        return_probability=True,
        return_masked_roi=False,
    )

    rows: list[dict[str, Any]] = []
    success = 0
    failures: list[dict[str, Any]] = []
    cache_skips = 0
    # 断点续跑：已有完整 ROI cache 则跳过推理
    existing_manifest_path = output_dir / "stain_manifest.parquet"
    cached_rows: dict[str, dict[str, Any]] = {}
    if existing_manifest_path.exists():
        prior = pd.read_parquet(existing_manifest_path)
        for _index, prior_row in prior.iterrows():
            if bool(prior_row.get("eligible")) and prior_row.get("roi_rgb_path") and Path(
                str(prior_row["roi_rgb_path"])
            ).exists() and Path(str(prior_row.get("roi_mask_path") or "")).exists():
                cached_rows[str(prior_row["sample_id"])] = prior_row.to_dict()

    total_planned = int(len(base_frame))
    print(
        f"[stain-preflight] start total={total_planned} "
        f"manifest_cache={len(cached_rows)} roi_dir={roi_cache_dir.resolve()}",
        flush=True,
    )
    for row_index, (_index, row) in enumerate(base_frame.iterrows(), start=1):
        sample_id = str(row["sample_id"])
        image_path = str(row["source_image_path"])
        if row_index == 1 or row_index % 50 == 0 or row_index == total_planned:
            print(
                f"[stain-preflight] {row_index}/{total_planned} "
                f"success={success} cache_skips={cache_skips}",
                flush=True,
            )
        if sample_id in cached_rows:
            record = dict(cached_rows[sample_id])
            # 用当前 base 标签/split 覆盖，防止旧缓存语义漂移
            record["label"] = int(row["label"])
            record["split"] = row["split"]
            record["canonical_label"] = row["canonical_label"]
            record["md5"] = row["md5"]
            rows.append(record)
            success += 1
            cache_skips += 1
            continue
        # 仅 cache 文件存在但无 manifest 时也可复用
        safe_name = (
            sample_id.replace("::", "__")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("／", "_")
        )
        roi_rgb_path = roi_cache_dir / f"{safe_name}_roi.png"
        roi_mask_path = roi_cache_dir / f"{safe_name}_mask.png"
        if roi_rgb_path.exists() and roi_mask_path.exists():
            from PIL import Image

            roi_rgb = np.asarray(Image.open(roi_rgb_path).convert("RGB"))
            roi_mask = np.asarray(Image.open(roi_mask_path))
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label": int(row["label"]),
                    "canonical_label": row["canonical_label"],
                    "source_image_path": image_path,
                    "md5": row["md5"],
                    "d3_status": "success",
                    "d3_checkpoint_hash": d3_hash,
                    "bbox_tight": None,
                    "bbox_roi": None,
                    "roi_width": int(roi_rgb.shape[1]),
                    "roi_height": int(roi_rgb.shape[0]),
                    "roi_rgb_path": str(roi_rgb_path.resolve()),
                    "roi_mask_path": str(roi_mask_path.resolve()),
                    "foreground_ratio": float((roi_mask > 0).mean()),
                    "component_count": None,
                    "largest_component_ratio": None,
                    "eligible": True,
                    "exclusion_reason": None,
                    "contract_version": STAIN_CONTRACT_VERSION,
                    "width": int(row["width"]) if pd.notna(row.get("width")) else None,
                    "height": int(row["height"]) if pd.notna(row.get("height")) else None,
                }
            )
            success += 1
            cache_skips += 1
            continue
        try:
            rgb, _mode = load_rgb_image(image_path)
            result = engine.predict(rgb, sample_id=sample_id)
        except Exception as exc:
            record = {
                **row.to_dict(),
                "d3_status": "error",
                "d3_checkpoint_hash": d3_hash,
                "bbox_tight": None,
                "bbox_roi": None,
                "roi_width": None,
                "roi_height": None,
                "foreground_ratio": None,
                "component_count": None,
                "largest_component_ratio": None,
                "eligible": False,
                "exclusion_reason": f"d3_error:{exc}",
            }
            rows.append(record)
            failures.append({"sample_id": sample_id, "reason": record["exclusion_reason"]})
            continue

        eligible = (
            result.status == "success"
            and result.bbox_roi is not None
            and result.tongue_roi_rgb is not None
            and result.tongue_roi_mask is not None
            and result.tongue_roi_rgb.size > 0
        )
        roi_rgb_path = None
        roi_mask_path = None
        if eligible:
            success += 1
            exclusion_reason = None
            # 本地缓存 ROI（gitignore）；训练时读取，避免每 step 重跑 D3-E
            safe_name = sample_id.replace("::", "__").replace("/", "_").replace("\\", "_")
            roi_rgb_path = str(roi_cache_dir / f"{safe_name}_roi.png")
            roi_mask_path = str(roi_cache_dir / f"{safe_name}_mask.png")
            from PIL import Image

            Image.fromarray(result.tongue_roi_rgb).save(roi_rgb_path)
            Image.fromarray((result.tongue_roi_mask > 0).astype(np.uint8) * 255).save(
                roi_mask_path
            )
        else:
            exclusion_reason = f"invalid_roi:status={result.status}"
            failures.append({"sample_id": sample_id, "reason": exclusion_reason})

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": row["dataset"],
                "split": row["split"],
                "label": int(row["label"]),
                "canonical_label": row["canonical_label"],
                "source_image_path": image_path,
                "md5": row["md5"],
                "d3_status": result.status,
                "d3_checkpoint_hash": d3_hash,
                "bbox_tight": list(result.bbox_tight) if result.bbox_tight else None,
                "bbox_roi": list(result.bbox_roi) if result.bbox_roi else None,
                "roi_width": int(result.roi_size[0]) if result.roi_size else None,
                "roi_height": int(result.roi_size[1]) if result.roi_size else None,
                "roi_rgb_path": roi_rgb_path,
                "roi_mask_path": roi_mask_path,
                "foreground_ratio": result.mask_foreground_ratio,
                "component_count": result.component_count,
                "largest_component_ratio": result.largest_component_ratio,
                "eligible": bool(eligible),
                "exclusion_reason": exclusion_reason,
                "contract_version": STAIN_CONTRACT_VERSION,
                "width": int(row["width"]) if pd.notna(row.get("width")) else None,
                "height": int(row["height"]) if pd.notna(row.get("height")) else None,
            }
        )

    manifest = pd.DataFrame(rows)
    total = len(manifest)
    success_rate = float(success / total) if total else 0.0
    min_rate = stain_data_config.min_roi_success_rate
    gate_pass = success_rate >= min_rate

    audit = {
        "stage": "D4-C",
        "dataset": stain_data_config.dataset,
        "task": stain_data_config.task,
        "total_canonical_samples": int(total),
        "train_samples": int((manifest["split"] == "train").sum()),
        "val_samples": int((manifest["split"] == "val").sum()),
        "test_samples": int((manifest["split"] == "test").sum()),
        "positive_total": int((manifest["label"] == 1).sum()),
        "negative_total": int((manifest["label"] == 0).sum()),
        "per_split_balance": {
            split_name: {
                "positive": int(
                    (
                        (manifest["split"] == split_name) & (manifest["label"] == 1)
                    ).sum()
                ),
                "negative": int(
                    (
                        (manifest["split"] == split_name) & (manifest["label"] == 0)
                    ).sum()
                ),
                "positive_rate": float(
                    manifest.loc[manifest["split"] == split_name, "label"].mean()
                )
                if (manifest["split"] == split_name).any()
                else None,
            }
            for split_name in ("train", "val", "test")
        },
        "d3_checkpoint_hash": d3_hash,
        "roi_success_count": int(success),
        "roi_success_rate": success_rate,
        "min_roi_success_rate": min_rate,
        "roi_success_gate": "PASS" if gate_pass else "FAIL",
        "no_tongue_detected": int((manifest["d3_status"] == "no_tongue_detected").sum()),
        # 仅统计失败样本中的 invalid bbox；cache resume 可能缺 bbox 但仍 eligible
        "invalid_bbox": int(
            ((manifest["eligible"] == False) & (manifest["bbox_roi"].isna())).sum()
        ),
        "empty_roi": int((manifest["eligible"] == False).sum()),
        "bbox_metadata_missing_but_eligible": int(
            ((manifest["eligible"] == True) & (manifest["bbox_roi"].isna())).sum()
        ),
        "excluded_samples": failures,
        "foreground_ratio": {
            "mean": float(manifest["foreground_ratio"].dropna().mean())
            if manifest["foreground_ratio"].notna().any()
            else None,
            "median": float(manifest["foreground_ratio"].dropna().median())
            if manifest["foreground_ratio"].notna().any()
            else None,
        },
        "note": (
            "Stain model is image quality/contamination detector; "
            "NOT TCM coating.color phenotype model."
        ),
    }

    manifest_path = output_dir / "stain_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    audit_path = reports_dir / "d4c_stain_dataset_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "d4c_stain_dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not gate_pass:
        raise RuntimeError(
            f"D3-E ROI success_rate={success_rate:.4f} < {min_rate}; "
            f"STOP before stain training. failures={len(failures)}"
        )
    return {"manifest_path": str(manifest_path), "audit": audit, "manifest": manifest}


def class_balance_report(manifest: pd.DataFrame) -> dict[str, Any]:
    report = {"overall": {}, "splits": {}}
    report["overall"] = {
        "stained": int((manifest["label"] == 1).sum()),
        "non_stained": int((manifest["label"] == 0).sum()),
        "positive_rate": float((manifest["label"] == 1).mean()),
    }
    for split_name in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split_name]
        report["splits"][split_name] = {
            "n": int(len(subset)),
            "positive": int((subset["label"] == 1).sum()),
            "negative": int((subset["label"] == 0).sum()),
            "positive_rate": float((subset["label"] == 1).mean()) if len(subset) else None,
        }
    return report
