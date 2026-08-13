# D4-B Freeze Report — Rule / Signal Quality Baseline

## Status

**D4-B = PASS**

机器可读：`docs/D4_B_FREEZE_STATS.json`  
契约：`docs/D4_B_SIGNAL_QC_CONTRACT.md`

## Versions

| Field | Value |
|---|---|
| Stage | **D4-B** |
| Input Guard contract | **1.0**（未改语义） |
| Policy | **1.1**（`configs/input_guard_v1.yaml`） |
| Threshold status | **engineering_heuristic** |
| Upstream | D4-A PASS / D3-E PASS |
| D3 checkpoint | **未修改** |
| Base commit before D4-B | `47d8298` |

## Discipline

- Calibration splits：**train + val only**（n=**1169**）
- **test 未用于** threshold 选择 / 调整
- Test audit 一次；audit 后 **policy 未修改**
- 未训练 QC / stain 模型
- 未伪实现 color_cast / occlusion / stain

## Implemented Checks（8）

presence / scale / completeness / segmentation_integrity / focus / exposure / illumination_uniformity / resolution

## Still not_evaluated（3）

color_cast / occlusion / stain_suspected

## Runtime Flags

| Flag | Value |
|---|---|
| evaluation_complete | **false** |
| guard_ready | **false** |
| quality_confidence | null |

## Calibration Rationale（摘要）

- **Scale**：foreground / tight-bbox width/height 的 overall p05→warning、p01→retake；多特征 coincidence
- **Completeness**：top-only 不自动 RETAKE；左右/底部高 touch ratio
- **Segmentation**：train+val 的 `largest_component_ratio` 退化到 1.0 → 使用 engineering floors **0.95 / 0.85**；mean probability 仅 proxy
- **Focus**：ROI Laplacian @ long_side=**256**；RETAKE=min(domain p01)，WARNING=overall p05（减少 false RETAKE）
- **Exposure**：dark/bright/clip ratios 的 p95/p99；不用 mean RGB 当舌色
- **Illumination**：mask-aware relative range p95/p99
- **Resolution**：tongue_pixel_count / effective_short_side p05/p01

完整数值见 `reports/d4/d4b_threshold_calibration.json` 与 Freeze Stats。

## Frozen Test Engineering Audit（130）

| Decision | Count |
|---|---:|
| pass | **74** |
| warning | **43** |
| retake | **13** |
| retake_rate | **0.10** |
| calibration_review_required | **false** |

Per dataset：

| Dataset | pass | warning | retake |
|---|---:|---:|---:|
| BioHit | 29 | 1 | 0 |
| TongueSet3 | 45 | 42 | 13 |

主要 reason（非互斥）：`TONGUE_SLIGHTLY_SMALL`、`IMAGE_BLUR`、`TONGUE_TOUCHES_FRAME`、`UNDEREXPOSED`、`IMAGE_RESOLUTION_TOO_LOW`、`UNEVEN_LIGHTING` 等。

> Test RETAKE ≠ 自动错误（无 QC gold）。未据此改 threshold。

## Known Failure：`biohit::278.bmp`

D4-B decision = **pass**（signal QC 未识别该 D3 严重错分割）。  
说明：当前规则抓采集质量信号，不替代分割失败诊断。未为 278 调阈值。

## Pytest / Validator

```text
176 passed
validate-input-guard → OK (warn: 8/11 implemented)
```

## Acceptance Gate

| Gate | Result |
|---|---|
| D4-A contract 未破坏 | PASS |
| D3 weights 未改 | PASS |
| test 未用于 calibration | PASS |
| train+val audit + thresholds Freeze | PASS |
| 8 checks implemented | PASS |
| color/occlusion/stain 未伪实现 | PASS |
| blur=ROI + fixed scale | PASS |
| exposure=original RGB ROI | PASS |
| illumination mask-aware | PASS |
| scale/resolution 分离 | PASS |
| evaluation_complete/guard_ready=false | PASS |
| test audit + policy unchanged | PASS |
| pytest PASS | PASS |

## 阶段判断

具备进入 **D4-C Stain Detection Baseline** 的条件。

本阶段 **STOP**；未经确认不进入 D4-C。
