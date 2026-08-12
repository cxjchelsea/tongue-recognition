#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit local tongue diagnosis datasets (fact layer only).

Scans dataset roots under 舌象更新/ (including Tongueset3), produces per-dataset
JSON, summary CSV/MD, cross-dataset MD5 duplicates, and matrix_facts.json.
Does NOT decide commercial use or trust grades.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install Pillow") from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {"__macosx", ".git", ".svn", "__pycache__", ".ipynb_checkpoints"}
MAC_RESOURCE_PREFIX = "._"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_skipped_path(path: Path) -> bool:
    parts_lower = {p.lower() for p in path.parts}
    if parts_lower & SKIP_DIR_NAMES:
        return True
    if path.name.startswith(MAC_RESOURCE_PREFIX):
        return True
    if path.name in {".DS_Store", "Thumbs.db"}:
        return True
    return False


def iter_images(root: Path, *, only_under: Optional[Sequence[Path]] = None) -> Iterable[Path]:
    roots = list(only_under) if only_under else [root]
    for base in roots:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # prune in-place
            dirnames[:] = [
                d
                for d in dirnames
                if d.lower() not in SKIP_DIR_NAMES and not d.startswith(MAC_RESOURCE_PREFIX)
            ]
            dp = Path(dirpath)
            for name in filenames:
                p = dp / name
                if is_skipped_path(p):
                    continue
                if p.suffix.lower() in IMAGE_EXTS:
                    yield p


def md5_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def inspect_image(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "rel": None,
        "bytes": path.stat().st_size,
        "md5": None,
        "ok": False,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }
    try:
        info["md5"] = md5_file(path)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            info["mode"] = im.mode
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 — audit must catch all decode failures
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def tree_summary(root: Path, max_depth: int = 3) -> Dict[str, Any]:
    """Folder counts at depth <= max_depth (relative to root)."""
    folders: Dict[str, int] = {}
    if not root.exists():
        return {"depth": max_depth, "folders": {}}
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        try:
            rel = dp.relative_to(root)
        except ValueError:
            continue
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        # prune junk
        dirnames[:] = [
            d
            for d in dirnames
            if d.lower() not in SKIP_DIR_NAMES and not d.startswith(MAC_RESOURCE_PREFIX)
        ]
        key = "." if str(rel) == "." else str(rel).replace("\\", "/")
        file_count = sum(
            1
            for n in filenames
            if not n.startswith(MAC_RESOURCE_PREFIX) and n not in {".DS_Store", "Thumbs.db"}
        )
        folders[key] = file_count
    return {"depth": max_depth, "folders": dict(sorted(folders.items())[:200])}


def find_license_hits(root: Path) -> Dict[str, Any]:
    names = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "CITATION", "CITATION.cff")
    found: List[str] = []
    keyword_hits: List[Dict[str, str]] = []
    keywords = (
        "license",
        "copyright",
        "commercial",
        "non-commercial",
        "noncommercial",
        "cc-by",
        "cc by",
        "all rights reserved",
        "research only",
        "for research",
    )
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIR_NAMES]
        for name in filenames:
            lower = name.lower()
            p = Path(dirpath) / name
            if name.upper() in {n.upper() for n in names} or lower.startswith("license"):
                found.append(str(p))
            if lower in {"readme", "readme.txt", "readme.md"} or lower.startswith("readme"):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
                except OSError:
                    continue
                for kw in keywords:
                    if kw in text:
                        keyword_hits.append({"file": str(p), "keyword": kw})
    # dedupe keyword hits
    seen = set()
    uniq = []
    for h in keyword_hits:
        key = (h["file"], h["keyword"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return {"license_files_found": found, "readme_keyword_hits": uniq[:50]}


@dataclass
class ImageScanResult:
    scanned: int = 0
    readable: int = 0
    corrupt: int = 0
    unique_md5: int = 0
    duplicate_md5_groups: int = 0
    duplicate_extra_files: int = 0
    size_counter: Counter = field(default_factory=Counter)
    mode_counter: Counter = field(default_factory=Counter)
    corrupt_samples: List[Dict[str, Any]] = field(default_factory=list)
    md5_to_paths: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    stem_to_paths: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, info: Dict[str, Any], rel_root: Path) -> None:
        try:
            info["rel"] = str(Path(info["path"]).resolve().relative_to(rel_root.resolve())).replace(
                "\\", "/"
            )
        except Exception:  # noqa: BLE001
            info["rel"] = info["path"]
        self.scanned += 1
        self.records.append(info)
        if info["ok"]:
            self.readable += 1
            self.size_counter[f"{info['width']}x{info['height']}"] += 1
            self.mode_counter[str(info["mode"])] += 1
            md5 = info["md5"]
            if md5:
                self.md5_to_paths[md5].append(info["rel"])
            stem = Path(info["path"]).stem
            self.stem_to_paths[stem].append(info["rel"])
        else:
            self.corrupt += 1
            if len(self.corrupt_samples) < 30:
                self.corrupt_samples.append(
                    {"rel": info["rel"], "error": info["error"], "bytes": info["bytes"]}
                )

    def finalize(self) -> Dict[str, Any]:
        uniq = {m for m, paths in self.md5_to_paths.items() if m}
        dup_groups = {m: paths for m, paths in self.md5_to_paths.items() if len(paths) > 1}
        extra = sum(len(v) - 1 for v in dup_groups.values())
        multi_stem = {s: paths for s, paths in self.stem_to_paths.items() if len(paths) > 1}
        self.unique_md5 = len(uniq)
        self.duplicate_md5_groups = len(dup_groups)
        self.duplicate_extra_files = extra
        top_sizes = self.size_counter.most_common(15)
        return {
            "scanned_images": self.scanned,
            "readable_images": self.readable,
            "corrupt_images": self.corrupt,
            "unique_md5": self.unique_md5,
            "intra_duplicate_md5_groups": self.duplicate_md5_groups,
            "intra_duplicate_extra_files": self.duplicate_extra_files,
            "size_distribution_top": [{"size": k, "count": v} for k, v in top_sizes],
            "mode_distribution": dict(self.mode_counter),
            "corrupt_samples": self.corrupt_samples,
            "multi_stem_path_groups": len(multi_stem),
            "multi_stem_examples": [
                {"stem": s, "paths": paths[:5], "count": len(paths)}
                for s, paths in list(sorted(multi_stem.items(), key=lambda x: -len(x[1])))[:10]
            ],
            "duplicate_md5_examples": [
                {"md5": m, "count": len(paths), "paths": paths[:5]}
                for m, paths in list(sorted(dup_groups.items(), key=lambda x: -len(x[1])))[:10]
            ],
        }


def scan_images(paths: Iterable[Path], rel_root: Path) -> ImageScanResult:
    result = ImageScanResult()
    for p in paths:
        info = inspect_image(p)
        result.add(info, rel_root)
    return result


def read_text_rows(path: Path, delim: Optional[str] = None) -> List[List[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    if delim is None:
        if "\t" in lines[0]:
            delim = "\t"
        elif "," in lines[0]:
            delim = ","
        else:
            delim = "\t"
    rows = []
    for ln in lines:
        rows.append([c.strip() for c in ln.split(delim)])
    return rows


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


def adapt_biohit(root: Path) -> Dict[str, Any]:
    ds = root / "TongeImageDataset" / "dataset"
    mask = root / "TongeImageDataset" / "groundtruth" / "mask"
    gt_images = root / "TongeImageDataset" / "groundtruth" / "images"
    scan = scan_images(iter_images(root, only_under=[ds]), root)
    summary = scan.finalize()

    ds_stems = {p.stem for p in iter_images(ds)} if ds.exists() else set()
    mask_stems = {p.stem for p in iter_images(mask)} if mask.exists() else set()
    gt_stems = {p.stem for p in iter_images(gt_images)} if gt_images.exists() else set()
    paired = ds_stems & mask_stems
    missing_mask = sorted(ds_stems - mask_stems)[:20]
    missing_ds = sorted(mask_stems - ds_stems)[:20]

    fg_ratios: List[float] = []
    for p in list(iter_images(mask))[:300]:
        try:
            with Image.open(p) as im:
                arr = list(im.getdata())
            if not arr:
                continue
            # binary-ish: count non-zero
            fg = sum(1 for v in arr if (v if isinstance(v, int) else v[0]) > 0)
            fg_ratios.append(fg / len(arr))
        except Exception:  # noqa: BLE001
            continue
    fg_ratios.sort()

    def pct(q: float) -> Optional[float]:
        if not fg_ratios:
            return None
        idx = min(len(fg_ratios) - 1, max(0, int(round((len(fg_ratios) - 1) * q))))
        return round(fg_ratios[idx], 4)

    labels = {
        "task": "segmentation",
        "dataset_stems": len(ds_stems),
        "mask_stems": len(mask_stems),
        "gt_image_stems": len(gt_stems),
        "paired_dataset_mask": len(paired),
        "pair_rate": round(len(paired) / len(ds_stems), 4) if ds_stems else None,
        "missing_mask_examples": missing_mask,
        "missing_dataset_examples": missing_ds,
        "mask_fg_ratio_p50": pct(0.5),
        "mask_fg_ratio_p10": pct(0.1),
        "mask_fg_ratio_p90": pct(0.9),
        "patient_id_evidence": "none (numeric filenames 1..N)",
        "human_label_evidence": "manual segmentation masks in groundtruth/mask",
    }
    return {
        "dataset_id": "BioHit",
        "display_name": "BioHit Tongue Image Dataset",
        "root": str(root),
        "canonical_image_roots": [str(ds)],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_tongueset3(root: Path) -> Dict[str, Any]:
    """TongueSAM TongueSet3: wild/mobile tongue images + Labelme masks."""
    img_dir = root / "img"
    gt_dir = root / "gt"
    scan = scan_images(iter_images(root, only_under=[img_dir] if img_dir.exists() else [root]), root)
    summary = scan.finalize()

    img_stems = {p.stem for p in iter_images(img_dir)} if img_dir.exists() else set()
    gt_stems = {p.stem for p in iter_images(gt_dir)} if gt_dir.exists() else set()
    paired = img_stems & gt_stems

    fg_ratios: List[float] = []
    unique_fg_values: Counter = Counter()
    inverted_masks = 0
    for p in list(iter_images(gt_dir))[:1000] if gt_dir.exists() else []:
        try:
            with Image.open(p) as im:
                arr = list(im.getdata())
            if not arr:
                continue
            vals = []
            fg = 0
            for v in arr:
                pix = v[0] if isinstance(v, tuple) else v
                vals.append(pix)
                if pix > 0:
                    fg += 1
            ratio = fg / len(arr)
            fg_ratios.append(ratio)
            # TongueSet3 convention: (1,1,1)=tongue, (0,0,0)=bg; some may be inverted
            uniq = tuple(sorted(set(vals)))
            unique_fg_values[str(uniq)] += 1
            if ratio > 0.5:
                inverted_masks += 1
        except Exception:  # noqa: BLE001
            continue
    fg_ratios.sort()

    def pct(q: float) -> Optional[float]:
        if not fg_ratios:
            return None
        idx = min(len(fg_ratios) - 1, max(0, int(round((len(fg_ratios) - 1) * q))))
        return round(fg_ratios[idx], 4)

    labels = {
        "task": "segmentation",
        "source_note": (
            "TongueSAM TongueSet3: 1000 images from Baidu AI Studio, "
            "manual Labelme masks; resized 400x400; in-the-wild/mobile domain"
        ),
        "img_stems": len(img_stems),
        "gt_stems": len(gt_stems),
        "paired_img_gt": len(paired),
        "pair_rate": round(len(paired) / len(img_stems), 4) if img_stems else None,
        "mask_encoding": "RGB; tongue≈(1,1,1), background≈(0,0,0) — appears black in viewers",
        "mask_value_sets_top": [
            {"values": k, "count": v} for k, v in unique_fg_values.most_common(5)
        ],
        "mask_fg_ratio_p10": pct(0.1),
        "mask_fg_ratio_p50": pct(0.5),
        "mask_fg_ratio_p90": pct(0.9),
        "masks_with_fg_ratio_gt_0_5": inverted_masks,
        "patient_id_evidence": "none (numeric filenames 1..N)",
        "human_label_evidence": "manual Labelme segmentation (TongueSAM authors)",
        "warning": "No phenotype/disease class labels; mask pixel value is 1 not 255",
    }
    return {
        "dataset_id": "Tongueset3",
        "display_name": "Tongueset3",
        "root": str(root),
        "canonical_image_roots": [str(img_dir)],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_dsct(root: Path) -> Dict[str, Any]:
    expert = root / "data" / "data" / "expert data"
    class0 = expert / "0"
    class1 = expert / "1"
    canonical = [p for p in (class0, class1) if p.exists()]
    scan = scan_images(iter_images(root, only_under=canonical), root)
    summary = scan.finalize()

    # all jpegs for redundancy ratio
    all_imgs = list(iter_images(root))
    all_scan_count = len(all_imgs)

    dist = {
        "0": len(list(iter_images(class0))) if class0.exists() else 0,
        "1": len(list(iter_images(class1))) if class1.exists() else 0,
    }

    csv_labels: Dict[str, int] = Counter()
    csv_files = list(root.rglob("*.csv"))
    for cf in csv_files[:30]:
        if is_skipped_path(cf):
            continue
        try:
            rows = read_text_rows(cf, delim=",")
        except OSError:
            continue
        if not rows:
            continue
        header = [h.lower() for h in rows[0]]
        label_idx = None
        for cand in ("expert label", "label", "expert_label"):
            if cand in header:
                label_idx = header.index(cand)
                break
        if label_idx is None:
            continue
        for row in rows[1:]:
            if len(row) > label_idx:
                csv_labels[row[label_idx]] += 1

    labels = {
        "task": "binary_crack",
        "canonical_root": str(expert),
        "folder_label_distribution": dist,
        "csv_label_counts_sample": dict(csv_labels),
        "total_images_in_tree": all_scan_count,
        "canonical_images": summary["scanned_images"],
        "redundancy_factor": round(all_scan_count / summary["scanned_images"], 2)
        if summary["scanned_images"]
        else None,
        "patient_id_evidence": "none (numeric jpeg names)",
        "human_label_evidence": "expert data/{0,1} folders; CSV expert label columns present",
    }
    return {
        "dataset_id": "DSCT",
        "display_name": "DSCT裂纹舌数据集",
        "root": str(root),
        "canonical_image_roots": [str(p) for p in canonical],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_tmc(root: Path) -> Dict[str, Any]:
    txt_root = root / "shezhen datasets" / "shezhenv3-txt"
    # handle nested or flat
    if not txt_root.exists():
        # fallback search
        candidates = list(root.rglob("shezhenv3-txt"))
        txt_root = candidates[0] if candidates else txt_root

    split_counts = {}
    class_counter: Counter = Counter()
    empty_labels = 0
    bad_labels = 0
    image_label_pair_ok = 0
    image_missing_label = 0
    label_missing_image = 0
    yaml_names: Dict[str, Any] = {}
    classes_txt_names: List[str] = []

    yaml_path = txt_root / "shezhenv3-txt.yaml"
    if yaml_path.exists():
        try:
            import yaml  # optional

            yaml_names = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
        except Exception as exc:  # noqa: BLE001
            yaml_names = {"_parse_error": str(exc)}

    image_dirs = []
    for split in ("train", "val", "test"):
        img_dir = txt_root / split / "images"
        lbl_dir = txt_root / split / "labels"
        if not img_dir.exists():
            # sometimes nested shezhenv3-txt/shezhenv3-txt
            alt = txt_root / "shezhenv3-txt" / split / "images"
            if alt.exists():
                img_dir = alt
                lbl_dir = txt_root / "shezhenv3-txt" / split / "labels"
                txt_root = txt_root / "shezhenv3-txt"
        image_dirs.append(img_dir)
        imgs = list(iter_images(img_dir)) if img_dir.exists() else []
        split_counts[split] = len(imgs)
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        img_stems = {p.stem for p in imgs}
        image_label_pair_ok += len(img_stems & lbl_stems)
        image_missing_label += len(img_stems - lbl_stems)
        label_missing_image += len(lbl_stems - img_stems)
        for lp in lbl_dir.glob("*.txt") if lbl_dir.exists() else []:
            try:
                lines = [
                    ln.strip()
                    for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if ln.strip()
                ]
            except OSError:
                bad_labels += 1
                continue
            if not lines:
                empty_labels += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    bad_labels += 1
                    continue
                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    bad_labels += 1
                    continue
                class_counter[cls] += 1

    # classes.txt from any split
    for split in ("train", "val", "test"):
        ct = txt_root / split / "classes.txt"
        if not ct.exists():
            # coco sibling often has it
            ct = (
                root
                / "shezhen datasets"
                / "shezhenv3-coco"
                / "shezhenv3-coco"
                / split
                / "classes.txt"
            )
        if ct.exists():
            classes_txt_names = [
                ln.strip()
                for ln in ct.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ln.strip()
            ]
            break

    scan = scan_images(iter_images(root, only_under=[d for d in image_dirs if d.exists()]), root)
    summary = scan.finalize()

    expected = {"train": 5594, "val": 572, "test": 553}
    split_match = {k: split_counts.get(k) == expected[k] for k in expected}

    yaml_name_map = {}
    if isinstance(yaml_names, dict):
        names = yaml_names.get("names")
        if isinstance(names, dict):
            yaml_name_map = {str(k): v for k, v in names.items()}
        elif isinstance(names, list):
            yaml_name_map = {str(i): n for i, n in enumerate(names)}

    drift = []
    for i, cname in enumerate(classes_txt_names):
        yname = yaml_name_map.get(str(i))
        if yname and yname != cname:
            drift.append({"id": i, "classes_txt": cname, "yaml": yname})

    labels = {
        "task": "object_detection",
        "canonical_format": "shezhenv3-txt",
        "split_counts": split_counts,
        "expected_split_counts": expected,
        "split_matches_dryad_readme": split_match,
        "class_instance_distribution": {str(k): v for k, v in sorted(class_counter.items())},
        "num_classes_observed": len(class_counter),
        "empty_label_files": empty_labels,
        "bad_label_lines_or_files": bad_labels,
        "image_label_pair_ok": image_label_pair_ok,
        "images_missing_label": image_missing_label,
        "labels_missing_image": label_missing_image,
        "classes_txt": classes_txt_names,
        "yaml_names": yaml_name_map,
        "name_drift_classes_txt_vs_yaml": drift,
        "patient_id_evidence": "none (mixed filename patterns, no subject id field)",
        "human_label_evidence": "Dryad README claims pathological annotations; local LICENSE absent",
        "note": "coco/txt/xml are format mirrors; audit uses txt only to avoid x3 inflation",
    }
    return {
        "dataset_id": "TMC-Tongue",
        "display_name": "TMC-Tongue",
        "root": str(root),
        "canonical_image_roots": [str(d) for d in image_dirs if d.exists()],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_tonguedx(root: Path) -> Dict[str, Any]:
    release = root / "TongueDx2" / "release"
    origin = release / "origin"
    seg = release / "seg"
    list_dir = release / "list"
    scan = scan_images(iter_images(root, only_under=[origin] if origin.exists() else [release]), root)
    summary = scan.finalize()

    label_cols = [
        "TonguePale",
        "TipSideRed",
        "Spot",
        "Ecchymosis",
        "Crack",
        "Toothmark",
        "FurThick",
        "FurYellow",
        "Heart",
        "Lung",
        "Spleen",
        "Liver",
        "Kidney",
    ]
    col_counts: Dict[str, Counter] = {c: Counter() for c in label_cols}
    ids: Set[str] = set()
    rows_total = 0
    path_missing = 0
    csv_files = sorted(list_dir.glob("*.csv")) if list_dir.exists() else []

    # Prefer union of all list csvs but count unique ids/paths
    seen_paths: Set[str] = set()
    for cf in csv_files:
        try:
            with cf.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_total += 1
                    pid = (row.get("id") or "").strip()
                    if pid:
                        ids.add(pid)
                    ip = (row.get("image_path") or "").strip().replace("\\", "/")
                    if ip and ip not in seen_paths:
                        seen_paths.add(ip)
                        for c in label_cols:
                            if c in row and row[c] != "":
                                col_counts[c][row[c]] += 1
                        # resolve path
                        cand = release / ip
                        if not cand.exists():
                            # sometimes image_path already includes origin/
                            alt = root / "TongueDx2" / "release" / ip
                            if not alt.exists() and not (origin / Path(ip).name).exists():
                                path_missing += 1
        except OSError:
            continue

    origin_files = list(iter_images(origin)) if origin.exists() else []
    seg_files = list(iter_images(seg)) if seg.exists() else []
    origin_rel = {}
    for p in origin_files:
        try:
            origin_rel[str(p.relative_to(origin)).replace("\\", "/")] = p
        except ValueError:
            pass
    seg_rel = {}
    for p in seg_files:
        try:
            seg_rel[str(p.relative_to(seg)).replace("\\", "/")] = p
        except ValueError:
            pass
    paired = set(origin_rel) & set(seg_rel)

    # basename collisions across buckets
    basename_map: Dict[str, List[str]] = defaultdict(list)
    for p in origin_files:
        basename_map[p.name].append(str(p.relative_to(root)).replace("\\", "/"))
    collisions = {k: v for k, v in basename_map.items() if len(v) > 1}

    # patient views: id_view.jpg
    view_counter: Counter = Counter()
    for name in basename_map:
        m = re.match(r"^(\d+)_(\d+)\.", name)
        if m:
            view_counter[m.group(1)] += 1

    multi_view_patients = sum(1 for _, n in view_counter.items() if n > 1)

    labels = {
        "task": "multi_label_classification",
        "csv_files": [str(p.relative_to(root)).replace('\\', '/') for p in csv_files],
        "csv_rows_total_all_lists": rows_total,
        "unique_patient_ids": len(ids),
        "unique_image_paths_in_csv": len(seen_paths),
        "csv_image_path_missing_on_disk_unique": path_missing,
        "label_value_distributions": {c: dict(col_counts[c]) for c in label_cols},
        "origin_images": len(origin_files),
        "seg_images": len(seg_files),
        "origin_seg_paired_relpaths": len(paired),
        "origin_seg_pair_rate": round(len(paired) / len(origin_rel), 4) if origin_rel else None,
        "basename_collisions": len(collisions),
        "basename_collision_examples": [
            {"name": k, "paths": v[:4]} for k, v in list(collisions.items())[:8]
        ],
        "patients_with_multi_view_filename": multi_view_patients,
        "patient_id_evidence": "csv id column + filename {id}_{view}.jpg",
        "human_label_evidence": "structured binary phenotype columns in release/list CSV",
    }
    return {
        "dataset_id": "TongueDx",
        "display_name": "TongueDx",
        "root": str(root),
        "canonical_image_roots": [str(origin)],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_tonguexpert(root: Path) -> Dict[str, Any]:
    raw = root / "TongueImage" / "Raw"
    mask = root / "TongueImage" / "Mask"
    pheno = root / "Phenotypes"
    scan = scan_images(iter_images(root, only_under=[raw] if raw.exists() else [root / "TongueImage"]), root)
    summary = scan.finalize()

    raw_stems = {p.stem for p in iter_images(raw)} if raw.exists() else set()
    mask_stems = {p.stem for p in iter_images(mask)} if mask.exists() else set()
    paired = raw_stems & mask_stems

    l1 = pheno / "L1_Labels_Manual.txt"
    l2 = pheno / "L2_Labels_Predict.txt"

    def parse_label_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"exists": False}
        rows = read_text_rows(path)
        if not rows:
            return {"exists": True, "rows": 0}
        header = rows[0]
        # skip description-like headers if first col not SID
        if header[0].upper() != "SID" and len(rows) > 1 and rows[0][0] != "SID":
            # find header line
            for i, r in enumerate(rows[:5]):
                if r and r[0] == "SID":
                    header = r
                    rows = rows[i:]
                    break
        data = rows[1:]
        col_idx = {h: i for i, h in enumerate(header)}
        dists: Dict[str, Counter] = {}
        sids = []
        for col in header[1:]:
            if "label" in col.lower() or col.startswith("labels_"):
                dists[col] = Counter()
        for row in data:
            if not row:
                continue
            sids.append(row[0])
            for col, ctr in dists.items():
                i = col_idx[col]
                if i < len(row) and row[i] != "":
                    ctr[row[i]] += 1
        return {
            "exists": True,
            "header": header,
            "rows": len(data),
            "unique_sid": len(set(sids)),
            "label_distributions": {k: dict(v) for k, v in dists.items()},
        }

    l1_info = parse_label_file(l1)
    l2_info = parse_label_file(l2)
    l1_sids = set()
    l2_sids = set()
    if l1.exists():
        rows = read_text_rows(l1)
        for r in rows[1:]:
            if r:
                l1_sids.add(r[0])
    if l2.exists():
        rows = read_text_rows(l2)
        # header may be first
        start = 1 if rows and rows[0] and rows[0][0] == "SID" else 0
        for r in rows[start:]:
            if r:
                l2_sids.add(r[0])

    labels = {
        "task": "phenotype_multi_label",
        "raw_images": len(raw_stems),
        "mask_images": len(mask_stems),
        "raw_mask_paired": len(paired),
        "pair_rate": round(len(paired) / len(raw_stems), 4) if raw_stems else None,
        "L1_manual": l1_info,
        "L2_predict": l2_info,
        "L1_sid_in_raw": len(l1_sids & raw_stems),
        "L2_sid_in_raw": len(l2_sids & raw_stems),
        "L1_coverage_of_raw": round(len(l1_sids & raw_stems) / len(raw_stems), 4)
        if raw_stems
        else None,
        "L2_coverage_of_raw": round(len(l2_sids & raw_stems) / len(raw_stems), 4)
        if raw_stems
        else None,
        "patient_id_evidence": "SID sample id (TE...); not guaranteed person-level",
        "human_label_evidence": "L1_Labels_Manual.txt is manual; L2_Labels_Predict.txt is model-predicted",
        "warning": "Do not use L2 predicted labels as gold standard",
    }
    return {
        "dataset_id": "TonguExpert",
        "display_name": "TonguExpertDatabase",
        "root": str(root),
        "canonical_image_roots": [str(raw)],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_toothmark(root: Path) -> Dict[str, Any]:
    marked = root / "marked" / "marked"
    unmarked = root / "unmarked" / "unmarked"
    if not marked.exists():
        marked = root / "marked"
    if not unmarked.exists():
        unmarked = root / "unmarked"
    canonical = [p for p in (marked, unmarked) if p.exists()]
    scan = scan_images(iter_images(root, only_under=canonical), root)
    summary = scan.finalize()
    dist = {
        "marked": len(list(iter_images(marked))) if marked.exists() else 0,
        "unmarked": len(list(iter_images(unmarked))) if unmarked.exists() else 0,
    }
    labels = {
        "task": "binary_toothmark",
        "folder_label_distribution": dist,
        "positive_rate": round(dist["marked"] / sum(dist.values()), 4) if sum(dist.values()) else None,
        "patient_id_evidence": "weak (timestamp-like filenames, no explicit subject id)",
        "human_label_evidence": "folder names marked/unmarked only",
    }
    return {
        "dataset_id": "Tooth-Marked",
        "display_name": "Tooth-Marked Tongue",
        "root": str(root),
        "canonical_image_roots": [str(p) for p in canonical],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


def adapt_stain(root: Path) -> Dict[str, Any]:
    # nested 中医舌诊染苔数据/染苔
    inner = root / "中医舌诊染苔数据"
    base = inner if inner.exists() else root
    stained = base / "染苔"
    clean = base / "非染苔"
    canonical = [p for p in (stained, clean) if p.exists()]
    scan = scan_images(iter_images(root, only_under=canonical), root)
    summary = scan.finalize()
    dist = {
        "染苔": len(list(iter_images(stained))) if stained.exists() else 0,
        "非染苔": len(list(iter_images(clean))) if clean.exists() else 0,
    }
    food_counter: Counter = Counter()
    if stained.exists():
        for p in iter_images(stained):
            # e.g. 1000_台式梅肉.JPG or 100_好丽友派.JPG
            stem = p.stem
            m = re.match(r"^\d+_(.+)$", stem)
            food = m.group(1) if m else stem
            food_counter[food] += 1
    labels = {
        "task": "binary_stain_vs_clean",
        "folder_label_distribution": dist,
        "stain_food_source_top": [
            {"source": k, "count": v} for k, v in food_counter.most_common(20)
        ],
        "unique_food_sources": len(food_counter),
        "patient_id_evidence": "none (serial + food name)",
        "human_label_evidence": "folder 染苔/非染苔; stain filenames encode food source",
        "warning": "Stain positives are food-dye artifacts, not pathological coating",
    }
    return {
        "dataset_id": "Stain",
        "display_name": "中医舌诊染苔数据",
        "root": str(root),
        "canonical_image_roots": [str(p) for p in canonical],
        "images": summary,
        "labels": labels,
        "tree": tree_summary(root),
        "license_probe": find_license_hits(root),
        "md5_index": dict(scan.md5_to_paths),
    }


DATASETS = [
    ("BioHit Tongue Image Dataset", adapt_biohit),
    ("DSCT裂纹舌数据集", adapt_dsct),
    ("TMC-Tongue", adapt_tmc),
    ("TongueDx", adapt_tonguedx),
    ("Tongueset3", adapt_tongueset3),
    ("TonguExpertDatabase", adapt_tonguexpert),
    ("Tooth-Marked Tongue", adapt_toothmark),
    ("中医舌诊染苔数据", adapt_stain),
]


def strip_md5_for_disk(report: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(report)
    out.pop("md5_index", None)
    return out


def build_matrix_facts(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    items = []
    for r in reports:
        img = r["images"]
        lab = r["labels"]
        items.append(
            {
                "dataset_id": r["dataset_id"],
                "display_name": r["display_name"],
                "unique_images_md5": img.get("unique_md5"),
                "scanned_canonical_images": img.get("scanned_images"),
                "corrupt_images": img.get("corrupt_images"),
                "corrupt_rate": round(img["corrupt_images"] / img["scanned_images"], 6)
                if img.get("scanned_images")
                else None,
                "intra_dup_extra_files": img.get("intra_duplicate_extra_files"),
                "top_size": (img.get("size_distribution_top") or [{}])[0],
                "patient_id_evidence": lab.get("patient_id_evidence"),
                "human_label_evidence": lab.get("human_label_evidence"),
                "task": lab.get("task"),
                "license_files_found": r.get("license_probe", {}).get("license_files_found", []),
                "readme_keyword_hits": r.get("license_probe", {}).get("readme_keyword_hits", []),
                "label_snapshot": {
                    k: lab[k]
                    for k in lab
                    if k
                    in {
                        "folder_label_distribution",
                        "split_counts",
                        "class_instance_distribution",
                        "unique_patient_ids",
                        "L1_manual",
                        "L2_predict",
                        "L1_coverage_of_raw",
                        "paired_dataset_mask",
                        "paired_img_gt",
                        "pair_rate",
                        "mask_encoding",
                        "mask_fg_ratio_p10",
                        "mask_fg_ratio_p50",
                        "mask_fg_ratio_p90",
                        "masks_with_fg_ratio_gt_0_5",
                        "canonical_images",
                        "redundancy_factor",
                        "stain_food_source_top",
                        "label_value_distributions",
                        "name_drift_classes_txt_vs_yaml",
                        "source_note",
                        "warning",
                    }
                },
            }
        )
    return {"generated_at": utc_now(), "datasets": items}


def write_summary_csv(path: Path, reports: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset_id",
        "display_name",
        "scanned_canonical_images",
        "readable_images",
        "corrupt_images",
        "unique_md5",
        "intra_duplicate_extra_files",
        "top_image_size",
        "task",
        "patient_id_evidence",
        "license_files_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in reports:
            top = (r["images"].get("size_distribution_top") or [{"size": ""}])[0]
            w.writerow(
                {
                    "dataset_id": r["dataset_id"],
                    "display_name": r["display_name"],
                    "scanned_canonical_images": r["images"]["scanned_images"],
                    "readable_images": r["images"]["readable_images"],
                    "corrupt_images": r["images"]["corrupt_images"],
                    "unique_md5": r["images"]["unique_md5"],
                    "intra_duplicate_extra_files": r["images"]["intra_duplicate_extra_files"],
                    "top_image_size": top.get("size", ""),
                    "task": r["labels"].get("task", ""),
                    "patient_id_evidence": r["labels"].get("patient_id_evidence", ""),
                    "license_files_count": len(
                        r.get("license_probe", {}).get("license_files_found", [])
                    ),
                }
            )


def write_summary_md(path: Path, reports: List[Dict[str, Any]], cross: Dict[str, Any]) -> None:
    lines = [
        "# Tongue dataset audit summary (fact layer)",
        "",
        f"Generated: {utc_now()}",
        "",
        "| dataset | unique_md5 | scanned | corrupt | intra_dup_extra | top_size | task |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in reports:
        top = (r["images"].get("size_distribution_top") or [{"size": "-"}])[0]
        lines.append(
            f"| {r['dataset_id']} | {r['images']['unique_md5']} | {r['images']['scanned_images']} | "
            f"{r['images']['corrupt_images']} | {r['images']['intra_duplicate_extra_files']} | "
            f"{top.get('size', '-')} | {r['labels'].get('task', '')} |"
        )
    lines += [
        "",
        "## Cross-dataset MD5 collisions",
        "",
        f"- colliding_md5_count: **{cross.get('colliding_md5_count', 0)}**",
        f"- colliding_file_pairs_approx: **{cross.get('pair_count_approx', 0)}**",
        "",
    ]
    for ex in (cross.get("examples") or [])[:15]:
        lines.append(f"- `{ex['md5'][:12]}…` datasets={ex['datasets']} files={ex['count']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cross_dataset_duplicates(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    md5_owners: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in reports:
        did = r["dataset_id"]
        for md5, paths in r.get("md5_index", {}).items():
            if not md5:
                continue
            for p in paths:
                md5_owners[md5].append((did, p))
    collisions = []
    pair_count = 0
    for md5, owners in md5_owners.items():
        datasets = sorted({d for d, _ in owners})
        if len(datasets) < 2:
            continue
        pair_count += 1
        collisions.append(
            {
                "md5": md5,
                "datasets": datasets,
                "count": len(owners),
                "examples": [{"dataset": d, "path": p} for d, p in owners[:6]],
            }
        )
    collisions.sort(key=lambda x: (-len(x["datasets"]), -x["count"]))
    return {
        "generated_at": utc_now(),
        "colliding_md5_count": len(collisions),
        "pair_count_approx": pair_count,
        "examples": collisions[:100],
        "all_collisions_truncated": len(collisions) > 100,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit tongue diagnosis datasets")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "舌象更新",
        help="Directory containing the 7 dataset folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output audit directory (default: <base>/_audit)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional dataset folder names to audit",
    )
    args = parser.parse_args(argv)
    base: Path = args.base
    out: Path = args.out or (base / "_audit")
    per_dir = out / "per_dataset"
    per_dir.mkdir(parents=True, exist_ok=True)

    selected = DATASETS
    if args.only:
        only_set = set(args.only)
        selected = [(n, fn) for n, fn in DATASETS if n in only_set]
        if not selected:
            print(f"No datasets matched --only {args.only}", file=sys.stderr)
            return 2

    reports: List[Dict[str, Any]] = []
    print(f"[audit] base={base}")
    print(f"[audit] out={out}")
    for name, adapter in selected:
        root = base / name
        print(f"[audit] >>> {name}")
        if not root.exists():
            report = {
                "dataset_id": name,
                "display_name": name,
                "root": str(root),
                "error": "root_missing",
                "images": {
                    "scanned_images": 0,
                    "readable_images": 0,
                    "corrupt_images": 0,
                    "unique_md5": 0,
                    "intra_duplicate_extra_files": 0,
                    "size_distribution_top": [],
                },
                "labels": {},
                "license_probe": {"license_files_found": [], "readme_keyword_hits": []},
                "md5_index": {},
            }
        else:
            report = adapter(root)
            report["audited_at"] = utc_now()
        disk = strip_md5_for_disk(report)
        out_path = per_dir / f"{report.get('dataset_id', name)}.json"
        out_path.write_text(json.dumps(disk, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[audit] <<< {report.get('dataset_id')} unique_md5="
            f"{report.get('images', {}).get('unique_md5')} scanned="
            f"{report.get('images', {}).get('scanned_images')}"
        )
        reports.append(report)

    cross = cross_dataset_duplicates(reports)
    (out / "cross_dataset_duplicates.json").write_text(
        json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    facts = build_matrix_facts(reports)
    (out / "matrix_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(out / "summary.csv", reports)
    write_summary_md(out / "summary.md", reports, cross)

    print(f"[audit] done. summary -> {out / 'summary.md'}")
    print(f"[audit] cross collisions: {cross['colliding_md5_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
