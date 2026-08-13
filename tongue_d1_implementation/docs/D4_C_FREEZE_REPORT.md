# D4-C Freeze Report

## Status

**TARGET_PASS**

Stage：D4-C Stain Detection Baseline  
Date：2026-08-13

## What was frozen

1. Stain data / ROI contract (`configs/stain_detection_v1.yaml`, version `1.0`)
2. Stain train config (`configs/stain_train_v1.yaml`)
3. ResNet18 binary stain detector checkpoint（本地 `runs/input_guard/d4c/stain/best.pt`，不入 Git）
4. Validation-only thresholds：`t_clear=0.95`, `t_retake=0.96`
5. Input Guard policy → `1.2`（`stain_suspected` enabled / learned_model）
6. Ontology：`quality.stain_suspected` marked `implemented=true`

## Dataset audit

| item | value |
|---|---|
| dataset | stained_coating |
| total | 1935 |
| train / val / test | 1548 / 194 / 193 |
| positive / negative | 991 / 944 |
| train pos/neg | 793 / 755 |
| val pos/neg | 99 / 95 |
| test pos/neg | 99 / 94 |
| D3-E ROI success | **1.0** |
| excluded | 0 |

D2 split 原样继承；sample/MD5 leakage = 0。  
未按 D4-B RETAKE 过滤训练样本。

## Model

| item | value |
|---|---|
| architecture | ResNet18 ImageNet |
| parameters | 11,177,025 |
| input | masked tongue ROI 224×224 letterbox |
| mask outside fill | 0 |
| augmentation | train geometry only；no hue/sat/color |
| loss | BCEWithLogits |
| seed | 20260813 |
| device | CUDA RTX 4090 |

## Training

| item | value |
|---|---|
| tiny overfit | PASS（acc=1.0） |
| planned epochs | 30 |
| actual epochs | 20（early stop） |
| best epoch | 13 |
| best val AUROC | 0.9988 |
| best val PR-AUC | 0.9989 |
| checkpoint monitor | val_auroc only |
| test used in train/selection | false |

## Threshold calibration（val only）

| item | value |
|---|---|
| t_clear | 0.95 |
| t_retake | 0.96 |
| constraint_not_met | false |
| val clean purity | 0.9895 |
| val stain precision | 0.9899 |
| val stain recall | 0.9899 |
| val uncertain rate | 0.0 |
| val confident coverage | 1.0 |

## Frozen test（once）

| metric | value |
|---|---|
| AUROC | 0.9918 |
| PR-AUC | 0.9956 |
| accuracy@0.5 | 0.9948 |
| balanced_accuracy@0.5 | 0.9949 |
| precision@0.5 | 1.0 |
| recall@0.5 | 0.9899 |
| specificity@0.5 | 1.0 |
| F1@0.5 | 0.9949 |
| confident clean purity | 0.9495 |
| confident stain precision | 1.0 |
| stain recall（3-state） | 0.9495 |
| uncertain rate | 0.0 |
| confident coverage | 1.0 |
| false negatives | 5 |
| false positives | 0 |
| baseline_status | **TARGET_PASS** |

注：3-state 下 5 个 FN 被映射为 `false`（分数 < t_clear），无 FP。  
test 未参与 threshold 重算。

## Input Guard state after D4-C

| item | value |
|---|---|
| defined checks | 11 |
| implemented checks | **9** |
| new implemented | `stain_suspected` |
| still deferred | `color_cast`, `occlusion` |
| evaluation_complete | **false** |
| guard_ready | **false** |
| policy version | **1.2** |

## Acceptance checklist

- [x] D3 frozen model 未修改
- [x] D4-A/B contract 未破坏（policy 升至 1.2，信号阈值未改）
- [x] Stain 语义 quality-only
- [x] 不与 coating.color 混淆
- [x] D2 split 原样继承
- [x] test 未参与训练 / threshold
- [x] Stained dataset real audit 完成
- [x] D3-E ROI success ≥ 0.99
- [x] background shortcut 抑制（mask fill 0）
- [x] original RGB 输入
- [x] 无强 color augmentation
- [x] binary stain model + tiny overfit PASS
- [x] best by val AUROC
- [x] t_clear / t_retake 仅 val
- [x] false / uncertain / true runtime 映射
- [x] frozen test once
- [x] Input Guard policy 1.2
- [x] evaluation_complete=false / guard_ready=false
- [x] Freeze docs / stats
- [x] pytest PASS（见 FREEZE_STATS）

## Stop line

D4-C 完成。**不要自动进入 D4-D / color_cast / occlusion / phenotype / API / UI。**
