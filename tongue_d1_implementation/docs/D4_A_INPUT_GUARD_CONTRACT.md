# D4-A Input Guard Contract — QC Ontology / Features / Decision Schema

## Purpose

D4 回答：

> 这张照片是否足够可靠，可以继续进入舌象视觉表型分析？

输出：

- `PASS` / `WARNING` / `RETAKE`
- reason codes
- quality features
- retake guidance

**不**输出舌色/苔色/裂纹/齿痕/证候/疾病诊断。

## Scope（本阶段）

D4-A **只建立契约**：

1. QC ontology  
2. Feature contract  
3. Decision semantics  
4. Reason codes  
5. Severity / evaluation state  
6. Evidence schema  
7. Retake guidance schema  
8. Policy config  
9. Validator  
10. Tests + Freeze  

**不训练** QC / 染苔模型；**不修改** D3 frozen segmentation。

## D3 / D4 Boundary

| Stage | Question | Output |
|---|---|---|
| D3 | 舌头在哪里？ | mask / bbox / ROI / seg metadata |
| D4 | 照片是否可继续表型分析？ | PASS/WARNING/RETAKE + reasons |

流程：

```text
原始图片 → D3 segmentation → D4 Input Guard
  ├─ RETAKE → 用户重拍建议
  ├─ WARNING → 可继续，但记录 warning / 可降权
  └─ PASS → phenotype pipeline
```

## Decision Semantics

| Decision | usable | 含义 |
|---|---|---|
| `pass` | true | 满足当前 V1 表型分析采集要求 |
| `warning` | true | 轻度问题，可继续，必须保存 reason |
| `retake` | false | 显著影响表型判断，不应继续正式 inference |

不用 `FAIL`；面向采集流程统一 `RETAKE`。

聚合：

```text
PASS < WARNING < RETAKE
```

仅 **evaluated** 且带 `decision_effect` 的 check 参与聚合。  
`not_evaluated` / disabled / unimplemented **不得**当作 PASS，也**不得**因 score=null 自动 RETAKE。

## evaluation_complete / guard_ready

| 字段 | D4-A | 含义 |
|---|---|---|
| `evaluation_complete` | **false** | 尚未完成全部质量检查 |
| `guard_ready` | **false** | 不可宣称已完整 QC 清关 |
| `quality_confidence` | **null** | 无 calibration，禁止编造分数 |

D4-D 完整实现后才可为 true。

## QC Ontology v1（11 checks）

| check_id | findings（摘要） | implementation |
|---|---|---|
| `quality.tongue_presence` | present / uncertain / absent | D4-B |
| `quality.tongue_scale` | adequate / small / too_small | D4-B |
| `quality.tongue_completeness` | complete / possibly_cropped / cropped | D4-B |
| `quality.segmentation_integrity` | good / fragmented / uncertain / invalid | D4-B |
| `quality.focus` | sharp / slightly_blurred / blurred | D4-B |
| `quality.exposure` | normal / under/over (+slightly) | D4-B |
| `quality.illumination_uniformity` | uniform / mildly_nonuniform / nonuniform | D4-B |
| `quality.color_cast` | acceptable / suspected / severe | D4-B |
| `quality.occlusion` | none / minor / major / possible_occlusion | D4-B |
| `quality.resolution` | adequate / low / too_low | D4-B |
| `quality.stain_suspected` | false / true / uncertain | **D4-C** |

D4-A：`defined=11`，`implemented=0`。

## Severity

统一：

`none` | `mild` | `moderate` | `severe`

Severity ≠ Decision；由 policy 映射。

## Evaluation State

`evaluated` | `not_evaluated` | `unavailable`

规则：

- 无舌 / 无 ROI 时，ROI 依赖 check → `not_evaluated`
- `not_evaluated` 不得带 `finding=sharp` 或 `decision_effect=pass`

## Reason Code Registry

集中 Enum（节选）：

- `NO_TONGUE_DETECTED`
- `TONGUE_TOO_SMALL` / `TONGUE_SLIGHTLY_SMALL`
- `TONGUE_CROPPED` / `TONGUE_TOUCHES_FRAME`
- `SEGMENTATION_*`
- `IMAGE_BLUR` / `TONGUE_BLUR`
- exposure / lighting / color cast / occlusion / resolution
- `STAIN_SUSPECTED`
- `UNKNOWN_QUALITY_ISSUE`

未知 reason → fail-fast。

`primary_reason`：RETAKE 档优先，同档按 `configs/input_guard_v1.yaml` 的 `primary_reason_priority`。

## Quality vs Phenotype（硬边界）

**不得**作为不合格照片原因：

红舌 / 淡舌 / 紫舌 / 裂纹 / 齿痕 / 厚苔 / 黄苔 / 剥苔 等病理表型。

这些是 phenotype 目标，不是 quality problem。

## Stain Semantics

`STAIN_SUSPECTED` = **外源污染 / contamination**，  
与病理 `coating.color`（如黄苔）严格分离。  
D1/D2 stained coating 数据集仅作 quality-only auxiliary；D4-C 再训 stain baseline。

## Feature Contract

`InputGuardFeatures`：

- 来自 D3-E：foreground / tight-bbox ratios / border touches / components / probabilities / ROI px
- D4-B 预留信号：blur / luminance / clip / illumination / color_cast → **null until implemented**
- **禁止**用 `0` 填充缺失

QC 几何比率使用 **tight bbox**（非 ROI margin bbox）。

Adapter：`features_from_segmentation_result(TongueSegmentationResult)`。

## Check / Result Schema

每个 check：

```json
{
  "check_id": "quality.focus",
  "evaluation_state": "not_evaluated",
  "finding": null,
  "severity": "none",
  "decision_effect": null,
  "score": null,
  "evidence": {},
  "reason_code": null,
  "source": "contract_skeleton"
}
```

`InputGuardResult`：decision / usable / evaluation_complete / checks / reason_codes /
primary_reason / warnings / retake_guidance / features / segmentation_reference /
quality_confidence / contract_version / guard_ready。

## Policy

`configs/input_guard_v1.yaml`

- check enabled / implementation_stage
- thresholds 可为 null，但必须 `needs_calibration: true`
- unknown check / reason → fail-fast

## Retake Guidance

reason → 用户采集建议（中文），不含疾病判断。  
未知 reason → safe fallback。

## Future Stages

| Stage | Role |
|---|---|
| D4-A | Contract / ontology（本阶段） |
| D4-B | Rule/signal quality baseline |
| D4-C | Stain detection baseline |
| D4-D | Unified calibrated Input Guard |
| D4-E | Retake guidance integration + Freeze |

## CLI

```bash
tongue-data validate-input-guard --policy configs/input_guard_v1.yaml

tongue-data input-guard-smoke \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --segmentation-dir data/segmentation/v1 \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --policy configs/input_guard_v1.yaml \
  --output runs/input_guard/d4a/smoke
```

Smoke 的 `contract_status=PASS` **只**表示契约执行通过，**不**表示图像质量全部 PASS。
