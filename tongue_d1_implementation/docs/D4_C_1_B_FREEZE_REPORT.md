# D4-C.1-B Domain-Robust Stain Freeze Report

## Status

- **baseline_status**: `NEEDS_IMPROVEMENT`
- **recommendation**: `NEEDS_IMPROVEMENT_STOP`
- **policy_activated**: `false`（仍为 policy 1.3 / stain v1）
- **stain_contract_version**: `1.1`（v2 合同存在，未切换 active detector）
- **v1 preserved**: `true`（`runs/input_guard/d4c/stain/` 未覆盖；`t_clear/t_retake=0.95/0.96` 保留）

## Early STOP

按协议：external VAL 上 TongueSet3 仍接近灾难性高分饱和时，

**不得继续消耗 source TEST / known external 130。**

因此下列评估 **SKIPPED**：

- source TEST one-shot evaluation
- known external 130 audit
- Unified Guard recovery audit
- full embedding / Grad-CAM v1↔v2 comparison（完整版）

## What was trained

| item | value |
|---|---|
| strategy | source supervised + source consistency + external unlabeled consistency |
| pseudo labels | **false** |
| architecture | ResNet18 ImageNet（不从 v1 fine-tune） |
| representation | black_masked_roi + letterbox 224 |
| source train/val/test | 1548 / 194 / 193 |
| external train | BioHit 240 + TongueSet3 799 |
| external val | BioHit 30 + TongueSet3 100 |
| seed | 20260813 |
| planned/actual epochs | 30 / 8（early stop） |
| best epoch | 8 |
| best source val AUROC / PR-AUC | 0.9978 / 0.9978 |
| t_clear_v2 / t_retake_v2 | 0.95 / 0.96（source val only） |

Style augmentation（train-only calibrated）：

- RGB channel gains ∈ [0.90, 1.35]
- gamma ∈ [0.80, 1.25]
- exposure ∈ [0.80, 1.25]
- contrast ∈ [0.85, 1.20]
- JPEG disabled in final train contract（throughput；range 保留）

## External VAL robustness（gate failure）

| metric | v1 | v2 |
|---|---:|---:|
| TongueSet3 median p | 0.9985 | 0.9895 |
| TongueSet3 highscore rate | 0.84 | **0.74** |
| TongueSet3 median logit | 6.50 | 4.54 |
| BioHit median p | 0.0069 | 0.0022 |
| domain median-logit gap | 11.52 | 10.65 |
| logit gap reduction | — | **7.5%**（目标 ≥50%） |
| catastrophic_saturation_resolved | — | **false** |

解读：

- source discrimination 仍然很强
- BioHit 更干净
- TongueSet3 饱和仅轻微缓解，**未达标**
- 当前 consistency + acquisition-style aug **不足以**消除 COLOR_ACQUISITION_STYLE shortcut

## Gates

| gate | result |
|---|---|
| v1 preserved | PASS |
| no external pseudo labels | PASS |
| style contract freeze | PASS |
| tiny overfit | PASS |
| external consistency smoke | PASS |
| best by source val AUROC | PASS |
| source val strong | PASS |
| external VAL saturation resolved | **FAIL** |
| policy activation | **NOT DONE** |

## Next

STOP。不要自动 Final Freeze D4，不要进 phenotype，不要换更大模型碰运气。

候选后续（需确认）：**D4-C.1-C**（更强 domain-robust 方案，仍禁止 external 伪标）。
