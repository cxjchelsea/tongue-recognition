# D4-B Signal QC Contract — Rule / Signal Quality Baseline

## Purpose

D4-A 定义“表达什么”；D4-B 实现：

> 哪些质量问题可由图像信号 + D3 metadata **稳定计算**？

本阶段实现 8 个 signal checks，并基于 **train+val** 分布冻结 engineering heuristic thresholds。

## Scope

**Implemented (8)**

1. `quality.tongue_presence`
2. `quality.tongue_scale`
3. `quality.tongue_completeness`
4. `quality.segmentation_integrity`
5. `quality.focus`
6. `quality.exposure`
7. `quality.illumination_uniformity`
8. `quality.resolution`

**Not implemented (still not_evaluated)**

9. `quality.color_cast` → D4-D（禁止 RGB 比例伪实现）
10. `quality.occlusion` → D4-D
11. `quality.stain_suspected` → D4-C

## Data Discipline

| Use | Splits |
|---|---|
| Feature audit / threshold calibration | train + val only |
| Forbidden for calibration | **test** |
| After freeze | test engineering audit once |

Thresholds = **engineering heuristics**  
≠ clinical thresholds / 医学标准。

## Runtime Semantics

- Known severe failure（如 `NO_TONGUE_DETECTED`）→ `decision=retake`, `usable=false`
- Implemented checks 全 PASS → `decision=pass` **但仍**：
  - `evaluation_complete=false`
  - `guard_ready=false`
- 不得当作正式完整采集清关

## Signal Notes

| Check | Key rules |
|---|---|
| presence | D3-E `no_tongue_detected` → RETAKE；ROI checks not_evaluated |
| scale | **tight bbox** + foreground；多特征 coincidence |
| completeness | top-only 不自动 RETAKE；左右/底部高接触才严重 |
| segmentation | largest_component_ratio；mean probability = **proxy only** |
| focus | **tongue ROI** Laplacian @ long_side=256 + Tenengrad |
| exposure | original RGB ROI luminance percentiles / clipping |
| illumination | mask-aware 3×3 grid；空 cell 不参与 |
| resolution | tongue pixels / effective short side（与 scale 分离） |

## Boundary Convention

数值比较统一使用严格不等式：

- lower-is-worse：`score < threshold` 触发
- higher-is-worse：`score >= threshold` 触发（clipping / uneven）

## Policy

- Contract version：**1.0**（不变）
- Policy version：**1.1**
- File：`configs/input_guard_v1.yaml`
- `source = signal_rule`；evidence 含 feature + thresholds

## Reports

```text
reports/d4/d4b_feature_distribution.json
reports/d4/d4b_threshold_calibration.json
reports/d4/d4b_test_audit.json
reports/d4/d4b_known_failure_analysis.json
```
