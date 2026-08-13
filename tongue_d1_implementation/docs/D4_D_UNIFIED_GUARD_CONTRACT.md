# D4-D Unified Input Guard Contract

## Goal

将 11 项 QC 组成可运行的统一 Input Guard，并明确：

- system-level `guard_ready`
- sample-level `evaluation_complete`
- `unavailable != pass`

Contract schema 保持 **1.0**（不新增破坏字段）。

## Checks

| # | check | implementation |
|---|---|---|
| 1-8 | D4-B signal rules | frozen |
| 9 | stain_suspected | D4-C learned_model frozen |
| 10 | color_cast | D4-D signal_rule（neutral-reference） |
| 11 | occlusion | D4-D signal_rule（multi-evidence） |

## Color Cast

- 只看 tongue mask **外** neutral / luminance 参考
- 禁止 tongue mean RGB / 舌色 phenotype 捷径
- insufficient support → `evaluation_state=unavailable`, finding=null
- findings: acceptable / suspected / severe

## Occlusion

- 使用 original-resolution probability map + mask
- interior low-probability holes + bright neutral intrusion
- 单弱证据不得单独 severe RETAKE
- 细裂纹 / 齿痕状边缘 / 小红点不得 major
- missing probability → unavailable（≠ none）
- findings（兼容 D4-A）: none / possible_occlusion / major

## Aggregation

- PASS < WARNING < RETAKE
- 除 no-tongue 外不强制 short-circuit；可收集多 reason
- `quality_confidence` 保持 null（不伪造综合分）

## Readiness

- `guard_ready`：系统 11/11 实现且 color_cast/occlusion engineering status=PASS
- `evaluation_complete`：本样本所有 check 均为 evaluated
- no-tongue RETAKE：决策可行动，但 evaluation_complete 仍可为 false

## Frozen references

- D3 checkpoint hash：不变
- D4-B thresholds：不变
- D4-C stain t_clear/t_retake：不变
