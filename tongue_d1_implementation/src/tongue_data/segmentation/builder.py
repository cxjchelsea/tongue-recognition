"""D3-A Segmentation Builder：manifest → audit → validate → smoke。"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import SegmentationConfig
from .dataset import smoke_test_dataset
from .manifest import build_segmentation_manifest
from .reproducibility import environment_record, seed_everything
from .validators import validate_segmentation


def _git_commit(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


class SegmentationBuilder:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = SegmentationConfig(self.config_path)

    def build(
        self,
        processed_dir: str | Path,
        split_dir: str | Path,
        output_dir: str | Path,
        report_dir: str | Path,
        run_smoke: bool = True,
    ) -> dict:
        seed_everything(self.config.seed)
        output_dir = Path(output_dir)
        report_dir = Path(report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        manifest, audit = build_segmentation_manifest(
            processed_dir, split_dir, self.config
        )
        if audit["errors_count"] > 0:
            # 仍写出 audit，便于排障
            (report_dir / "segmentation_dataset_audit.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise ValueError(
                f"segmentation manifest build failed: {audit['errors'][:10]}"
            )

        package_root = Path(__file__).resolve().parents[3]
        env = environment_record(self.config_path, self.config.seed, "auto")
        metadata = {
            "stage": "D3-A",
            "contract_version": self.config.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(package_root),
            "seed": self.config.seed,
            "input_resolution": {
                "height": self.config.input_height,
                "width": self.config.input_width,
            },
            "resize_policy": self.config.resize.get("policy", "letterbox"),
            "foreground_rule": self.config.foreground_rule,
            "datasets": self.config.datasets,
            "total_samples": audit["total_samples"],
            "per_split": audit["per_split"],
            "per_dataset": audit["per_dataset"],
            "missing_images": audit["missing_images"],
            "missing_masks": audit["missing_masks"],
            "shape_mismatches": audit["shape_mismatches"],
            "empty_masks": audit["empty_masks"],
            "full_masks": audit["full_masks"],
            "sample_leakage": audit["sample_leakage"],
            "md5_leakage": audit["md5_leakage"],
            "errors_count": audit["errors_count"],
            "warnings_count": audit["warnings_count"],
            "environment": env,
            "loss_contract": self.config.loss,
            "baseline_architecture": self.config.doc.get("baseline_architecture", {}),
            "training_contract": self.config.training_contract,
            "acceptance_targets_d3bc": self.config.acceptance_targets,
        }

        manifest.to_parquet(output_dir / "segmentation_manifest.parquet", index=False)
        (output_dir / "segmentation_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "segmentation_dataset_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        errors, warnings = validate_segmentation(
            output_dir, self.config_path, split_dir
        )
        metadata["validate_errors"] = errors
        metadata["validate_warnings"] = warnings
        smoke = None
        if run_smoke:
            smoke = smoke_test_dataset(
                output_dir / "segmentation_manifest.parquet",
                self.config_path,
            )
            metadata["smoke_test"] = smoke
            (report_dir / "segmentation_smoke_test.json").write_text(
                json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not smoke.get("ok", False):
                errors = list(errors) + ["smoke_test_failed"]

        (output_dir / "segmentation_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if errors:
            raise ValueError(f"validate-segmentation failed: {errors}")

        return {
            "metadata": metadata,
            "audit": audit,
            "smoke": smoke,
            "validate_errors": errors,
            "validate_warnings": warnings,
        }
