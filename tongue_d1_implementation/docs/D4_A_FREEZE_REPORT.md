# D4-A Freeze Report — Input Guard Contract / QC Ontology

## Status

**D4-A = PASS**

机器可读：`docs/D4_A_FREEZE_STATS.json`  
契约文档：`docs/D4_A_INPUT_GUARD_CONTRACT.md`

## Versions

| Field | Value |
|---|---|
| Stage | **D4-A** |
| Input Guard contract | **1.0** |
| QC ontology | **1.0** |
| Reason registry | **1.0** |
| Policy | `configs/input_guard_v1.yaml` |
| Upstream D3-E | PASS（`587259a`） |
| Segmentation threshold | **0.5**（未改） |
| D3 checkpoint | **未修改** |

## Discipline

本阶段 **未**：

- 训练 QC / blur / exposure 模型
- 训练染苔模型
- 修改 D3 weights / threshold
- 进入 phenotype classification
- 开始 D4-B 规则阈值标定

## Decision / Usable / Completeness

| Decision | usable |
|---|---|
| pass | true |
| warning | true |
| retake | false |

D4-A runtime skeleton：

- `evaluation_complete = false`
- `guard_ready = false`
- `quality_confidence = null`

禁止把 contract smoke 的 skeleton `pass` 解读为“照片质量已完整通过”。

## Ontology

- **Defined checks**: **11**
- **Implemented checks**: **0**（信号/模型在 D4-B/C）
- **Reason codes**: **23**
- **Severity**: none / mild / moderate / severe
- **Evaluation state**: evaluated / not_evaluated / unavailable

## Quality vs Phenotype

病理表型（红舌、黄苔、裂纹、齿痕等）**不得**作为 QC RETAKE reason。  
`STAIN_SUSPECTED` 属于外源污染质量问题，与病理苔色分离。

## Features / Adapter

`InputGuardFeatures`：**32** 字段（含 D4-B 预留 null 信号）。  
`features_from_segmentation_result()` 映射 D3-E：

- foreground_ratio / tight bbox ratios
- border touches
- component / probability
- ROI 像素尺寸

缺失特征保持 **null**，不用 0。

## Real Smoke

10 张（BioHit 5 + TongueSet3 5）：

| Check | Result |
|---|---|
| contract_status | **PASS** |
| schema_ok | all true |
| evaluation_complete | all **false** |
| guard_ready | all **false** |
| blur_score null | all true |

输出：`runs/input_guard/d4a/smoke/`、`reports/d4/d4a_contract_smoke.json`（gitignore）

## Pytest

```text
156 passed
```

## Acceptance Gate

| Gate | Result |
|---|---|
| D3 frozen model 未修改 | PASS |
| D3-E contract 未修改 | PASS |
| D3/D4 boundary 文档化 | PASS |
| PASS/WARNING/RETAKE + usable | PASS |
| evaluation_complete semantics | PASS |
| QC ontology v1 | PASS |
| quality vs phenotype | PASS |
| stain vs pathological coating | PASS |
| reason / severity / eval state | PASS |
| schemas + feature adapter | PASS |
| missing ≠ 0；not_evaluated ≠ pass | PASS |
| policy/reason fail-fast | PASS |
| unimplemented 不影响 decision | PASS |
| real smoke contract PASS | PASS |
| pytest PASS | PASS |
| Freeze Report / Stats | PASS |

## 阶段判断

D4-A 契约层完成，**具备进入 D4-B Rule/Signal Quality Baseline 的条件**。

本阶段 **STOP**；未经确认不进入 D4-B。
