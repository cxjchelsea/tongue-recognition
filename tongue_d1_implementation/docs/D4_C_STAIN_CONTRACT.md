# D4-C Stain Detection Contract

## Scope

实现且仅实现：

`quality.stain_suspected`

回答：

> 这张舌头照片是否可能受到食物、饮料、药物、色素等外源性染色影响，从而使舌苔颜色分析不可靠？

**不回答**：

- 舌苔是什么颜色（`coating.color`）
- 病理表型 / 疾病判断

## Semantics

| label | meaning |
|---|---|
| false | 无明显外源染色嫌疑 |
| uncertain | 证据不足，进入安全缓冲 |
| true | 外源染色嫌疑成立 |

Reason code（仅 quality）：

- `STAIN_SUSPECTED`

禁止：

- `YELLOW_COATING_STAIN`
- `BLACK_TONGUE`
- 任何 `coating.color` / phenotype reason

## Dataset

- source：D1/D2 frozen `stained_coating`
- role：`quality-only auxiliary supervision`
- split：100% 继承 D2 frozen split（禁止重分）
- label field：`canonical_label` → `true` / `false`
- 禁止从 `coating.color` 或 yellow/white/black 推导

## Input Contract

```
original RGB
→ D3-E bbox_roi / tongue ROI
→ tongue mask 外填充固定值 0
→ letterbox 224×224（preserve aspect ratio）
→ ImageNet normalize
→ ResNet18 → single raw logit
```

约束：

- RGB 必须来自 original image，不是 D3 normalized tensor
- 无 Hue / Saturation / ColorJitter / grayscale
- train 仅几何增强；val/test 无随机增强

## Model

- architecture：ResNet18 ImageNet pretrained
- loss：BCEWithLogitsLoss
- output：raw logit（forward 内不做 sigmoid）
- checkpoint selection：`val_auroc` only
- test 不参与训练 / checkpoint / threshold

## Threshold Contract

Validation-only dual thresholds：

- `p <= t_clear` → false
- `t_clear < p < t_retake` → uncertain
- `p >= t_retake` → true

校准目标：

- clear 区 clean purity ≥ `target_confident_precision`（默认 0.90）
- stain 区 stain precision ≥ 0.90

选择规则：在满足约束且 `t_clear < t_retake` 的合法阈值对中，按

1. 最大 confident coverage
2. 最大 `t_clear`
3. 最小 `t_retake`

确定性选择。禁止默认拍脑袋使用 0.3/0.7。

## Runtime Mapping

| finding | severity | decision_effect | reason |
|---|---|---|---|
| false | none | pass / no_effect | null |
| uncertain | moderate | warning | STAIN_SUSPECTED |
| true | severe | retake | STAIN_SUSPECTED |

source：`learned_model`

## Versions

- stain contract：`1.0`
- Input Guard contract：`1.0`
- Input Guard policy after D4-C：`1.2`

## Non-goals

本阶段不实现：

- `color_cast`
- `occlusion`
- Unified Input Guard（D4-D）
- phenotype model

因此：

- `evaluation_complete = false`
- `guard_ready = false`
