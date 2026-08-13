"""StainDetector：segmentation ROI → QCCheckResult。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tongue_data.input_guard.ontology import (
    CheckId,
    Decision,
    EvaluationState,
    EvidenceSource,
    ReasonCode,
    Severity,
)
from tongue_data.input_guard.schema import CheckResult

from .calibrate import load_frozen_thresholds
from .config import StainDataConfig, StainTrainConfig
from .metrics import map_probability_to_finding
from .model import build_stain_model
from .train import load_stain_checkpoint, resolve_device
from .transforms import preprocess_masked_roi


class StainDetector:
    """learned_model 实现 quality.stain_suspected。"""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        data_config_path: str | Path,
        train_config_path: str | Path,
        thresholds_path: str | Path,
        device: str = "auto",
    ):
        self.data_config = StainDataConfig(data_config_path)
        self.train_config = StainTrainConfig(train_config_path)
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_stain_checkpoint(
            checkpoint_path,
            train_config=self.train_config,
            data_config=self.data_config,
            map_location=self.device,
            strict=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.thresholds = load_frozen_thresholds(thresholds_path)
        self.t_clear = float(self.thresholds["t_clear"])
        self.t_retake = float(self.thresholds["t_retake"])
        if self.checkpoint.get("input_size") != self.data_config.input_size:
            raise ValueError("input_size mismatch between checkpoint and data contract")

    def _prepare_tensor(
        self, roi_rgb: np.ndarray, roi_mask: np.ndarray
    ) -> torch.Tensor:
        tensor = preprocess_masked_roi(
            roi_rgb,
            roi_mask,
            self.data_config,
            split="val",
            rng=None,
            augment_cfg=None,
        )
        return torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)

    @torch.inference_mode()
    def predict_probability(
        self, roi_rgb: np.ndarray, roi_mask: np.ndarray
    ) -> float:
        batch = self._prepare_tensor(roi_rgb, roi_mask).to(self.device)
        logits = self.model(batch)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return float(torch.sigmoid(logits.float())[0].cpu())

    def predict(
        self,
        image: np.ndarray | None,
        segmentation_result: Any,
    ) -> CheckResult:
        """
        返回 QC CheckResult。
        image 参数保留接口兼容；实际使用 segmentation_result 中的 original-space ROI。
        """
        del image  # 不直接用整图，避免背景 shortcut
        check_id = CheckId.STAIN_SUSPECTED.value
        status = getattr(segmentation_result, "status", None)
        roi_rgb = getattr(segmentation_result, "tongue_roi_rgb", None)
        roi_mask = getattr(segmentation_result, "tongue_roi_mask", None)
        if (
            status != "success"
            or roi_rgb is None
            or roi_mask is None
            or getattr(roi_rgb, "size", 0) == 0
        ):
            return CheckResult(
                check_id=check_id,
                evaluation_state=EvaluationState.UNAVAILABLE.value,
                finding=None,
                severity=Severity.NONE.value,
                decision_effect=None,
                score=None,
                thresholds={
                    "clear": self.t_clear,
                    "retake": self.t_retake,
                },
                evidence={"reason": "invalid_roi", "segmentation_status": status},
                reason_code=None,
                source=EvidenceSource.LEARNED_MODEL.value,
            )

        probability = self.predict_probability(roi_rgb, roi_mask)
        finding = map_probability_to_finding(
            probability, self.t_clear, self.t_retake
        )
        if finding == "false":
            return CheckResult(
                check_id=check_id,
                evaluation_state=EvaluationState.EVALUATED.value,
                finding="false",
                severity=Severity.NONE.value,
                decision_effect=Decision.PASS.value,
                score=probability,
                thresholds={"clear": self.t_clear, "retake": self.t_retake},
                evidence={
                    "p_stain": probability,
                    "model_contract": self.data_config.version,
                    "train_config_hash": self.train_config.config_hash,
                },
                reason_code=None,
                source=EvidenceSource.LEARNED_MODEL.value,
            )
        if finding == "uncertain":
            return CheckResult(
                check_id=check_id,
                evaluation_state=EvaluationState.EVALUATED.value,
                finding="uncertain",
                severity=Severity.MODERATE.value,
                decision_effect=Decision.WARNING.value,
                score=probability,
                thresholds={"clear": self.t_clear, "retake": self.t_retake},
                evidence={
                    "p_stain": probability,
                    "model_contract": self.data_config.version,
                    "train_config_hash": self.train_config.config_hash,
                },
                reason_code=ReasonCode.STAIN_SUSPECTED.value,
                source=EvidenceSource.LEARNED_MODEL.value,
            )
        return CheckResult(
            check_id=check_id,
            evaluation_state=EvaluationState.EVALUATED.value,
            finding="true",
            severity=Severity.SEVERE.value,
            decision_effect=Decision.RETAKE.value,
            score=probability,
            thresholds={"clear": self.t_clear, "retake": self.t_retake},
            evidence={
                "p_stain": probability,
                "model_contract": self.data_config.version,
                "train_config_hash": self.train_config.config_hash,
            },
            reason_code=ReasonCode.STAIN_SUSPECTED.value,
            source=EvidenceSource.LEARNED_MODEL.value,
        )


def update_policy_with_stain_thresholds(
    policy_path: str | Path,
    *,
    t_clear: float,
    t_retake: float,
    output_path: str | Path | None = None,
    policy_version: str = "1.2",
) -> Path:
    """将 val frozen thresholds 写入 policy；不写绝对 checkpoint 路径。"""
    import yaml

    path = Path(policy_path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["version"] = policy_version
    doc["policy_version"] = policy_version
    stain = doc.setdefault("checks", {}).setdefault("stain_suspected", {})
    stain["enabled"] = True
    stain["implementation_stage"] = "D4-C"
    stain["implementation"] = "learned_model"
    stain["model_contract"] = "1.0"
    stain["needs_calibration"] = False
    stain["thresholds"] = {
        "clear": float(t_clear),
        "retake": float(t_retake),
    }
    stain["warning_reason"] = "STAIN_SUSPECTED"
    stain["retake_reason"] = "STAIN_SUSPECTED"
    notes = list(doc.get("notes") or [])
    note = (
        "D4-C stain_suspected thresholds calibrated on stained_coating val only; "
        "test excluded from threshold selection."
    )
    if note not in notes:
        notes.append(note)
    doc["notes"] = notes
    out = Path(output_path) if output_path else path
    out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return out
