"""统计 D1.1 TongueDx 正/负监督（不写入隐私字段）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "v1"
TONGUEDX = Path(r"D:/project/tongue-recognition/TongueDx")


def attr_stats(frame: pd.DataFrame, field: str, label: str):
    subset = frame[
        (frame["source_field"] == field)
        & (frame["canonical_label"].astype(str) == label)
    ]
    positive = int((subset["value"].astype(int) == 1).sum())
    negative = int((subset["value"].astype(int) == 0).sum())
    return positive, negative, len(subset)


def main():
    samples = pd.read_parquet(MANIFEST / "samples.parquet")
    labels = pd.read_parquet(MANIFEST / "labels.parquet")
    spatial = pd.read_parquet(MANIFEST / "spatial_annotations.parquet")
    metadata = json.loads((MANIFEST / "build_metadata.json").read_text(encoding="utf-8"))
    dataset_stats = json.loads((MANIFEST / "dataset_statistics.json").read_text(encoding="utf-8"))

    frames = []
    for csv_name in ["train_fold1.csv", "val_fold1.csv", "test.csv"]:
        frames.append(pd.read_csv(TONGUEDX / "TongueDx2" / "release" / "list" / csv_name))
    source = pd.concat(frames, ignore_index=True)

    fields = [
        "TonguePale", "FurYellow", "Crack", "Toothmark",
        "Spot", "Ecchymosis", "FurThick", "TipSideRed",
    ]
    source_counts = {}
    for field in fields:
        source_counts[field] = {
            str(key): int(value)
            for key, value in source[field].value_counts(dropna=False).to_dict().items()
        }

    tonguedx_labels = labels[labels["source_dataset"] == "tonguedx"]
    pale_pos, pale_neg, _ = attr_stats(tonguedx_labels, "TonguePale", "pale")
    yellow_pos, yellow_neg, _ = attr_stats(tonguedx_labels, "FurYellow", "yellow")

    binary_out = {}
    for field in ["Crack", "Toothmark", "Spot", "Ecchymosis", "FurThick", "TipSideRed"]:
        subset = tonguedx_labels[tonguedx_labels["source_field"] == field]
        label_text = subset["canonical_label"].astype(str).str.lower()
        binary_out[field] = {
            "positive": int(((label_text == "true") & (subset["value"].astype(int) == 1)).sum()),
            "explicit_negative": int(((label_text == "false") & (subset["value"].astype(int) == 1)).sum()),
        }

    fur_yellow_to_white = int(
        (
            (tonguedx_labels["source_field"] == "FurYellow")
            & (tonguedx_labels["canonical_label"].astype(str) == "white")
        ).sum()
    )
    pale_to_normal = int(
        (
            (tonguedx_labels["source_field"] == "TonguePale")
            & (tonguedx_labels["canonical_label"].astype(str) == "normal")
        ).sum()
    )

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "contract_version": "1.1",
        "ontology_version": metadata.get("ontology_version"),
        "tonguedx_mapping_version": "1.1",
        "manifest_version": metadata.get("manifest_version"),
        "ingest_version": "1.1",
        "git_commit": git_commit,
        "build_timestamp": metadata.get("build_timestamp"),
        "samples_total": int(len(samples)),
        "labels_total": int(len(labels)),
        "spatial_total": int(len(spatial)),
        "warnings_count": metadata.get("warnings_count"),
        "dataset_statistics": dataset_stats,
        "d1_v1_0_labels_total": 82929,
        "d1_1_labels_total": int(len(labels)),
        "labels_delta_vs_d1_v1_0": int(len(labels) - 82929),
        "tonguedx_source_csv_counts": source_counts,
        "tonguedx_manifest_attr": {
            "TonguePale": {
                "positive": pale_pos,
                "explicit_negative": pale_neg,
                "unavailable": 0,
            },
            "FurYellow": {
                "positive": yellow_pos,
                "explicit_negative": yellow_neg,
                "unavailable": 0,
            },
        },
        "tonguedx_manifest_binary": binary_out,
        "guards": {
            "FurYellow_mapped_to_white": fur_yellow_to_white,
            "TonguePale_mapped_to_normal": pale_to_normal,
        },
    }
    out_path = ROOT / "docs" / "D1_1_FREEZE_STATS.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["tonguedx_manifest_attr"], ensure_ascii=False, indent=2))
    print(json.dumps(report["tonguedx_manifest_binary"], ensure_ascii=False, indent=2))
    print("delta", report["labels_delta_vs_d1_v1_0"], "warnings", report["warnings_count"])
    print("guards", report["guards"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
