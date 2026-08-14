# D4-C.1-C Experiment Report

**final_status:** `REPRESENTATION_DOMAIN_INVARIANCE_FAILED`  
**annotation:** `SOURCE_CONFOUNDING_SUSPECTED`  
**policy_activated:** `false`  
**Freeze PASS:** 否（本阶段不写 Freeze PASS）

## Summary

C1 / C2 / C3 均完成训练与 external VAL dual-gate。  
**无一 candidate 通过预注册 acceptance**（source + domain robustness 全门）。  
按协议：**未访问** source TEST / known 130 / Unified recovery；**未切换** policy 1.3→1.4。

C2 最接近：domain logit gap 相对 v2 下降约 **77%**，TongueSet3 highscore **0.47&lt;0.50**，但 embedding domain-probe drop **0.078 &lt; 0.10**，故 FAIL。

Source confounding audit：`source_confounding_suspected=true`  
（Stained pos vs neg：luminance / red / resolution 差异极大）→ 建议下一步 **DATASET_CONFOUNDING_AUDIT**，而非继续加大模型/GRL。

## Baselines preserved

| Artifact | Status |
|---|---|
| D4-C v1 checkpoint / thresholds 0.95/0.96 | 保留 |
| D4-C.1-B v2 checkpoint / thresholds | 保留 |
| Input Guard Policy 1.3 | 未切换 |
| D4-B / color_cast / occlusion | 未改 |

## Contract / setup

- stain contract v3：`configs/stain_detection_v3.yaml`（1.2）
- train：`configs/stain_train_v3.yaml`
- backbone：ResNet18 ImageNet（未换更大架构）
- MixStyle：`layer1`, p=0.5, alpha=0.1, cross-domain
- GRL：3-way domain head + linear warmup → λ_max=0.3
- domain-balanced batch：per_domain=8
- external：**无** stain pseudo-label / entropy min
- pytest：`tests/data_contract/test_input_guard_d4c1c.py` → **50 passed**

## Candidate results (external VAL)

| Candidate | best_epoch | source val AUROC | BioHit med logit | TS3 med logit | gap | gap↓ vs v2 | TS3 highscore | domain probe | probeΔ vs v2 | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 (v2 ref) | — | ~0.9978 | -6.11 | +4.54 | 10.65 | — | 0.74 | 0.878 | — | frozen |
| C1 MixStyle | 13 | 0.9994 | -8.27 | +2.84 | 11.11 | **-4.3%** | **0.45** | 0.808 | 0.070 | FAIL |
| C2 GRL | 12 | 0.9985 | +0.43 | +2.85 | **2.43** | **+77.2%** | **0.47** | 0.800 | 0.078 | FAIL |
| C3 Mix+GRL | 9 | 0.9982 | -4.53 | +3.81 | 8.34 | +21.7% | 0.60 | 0.784 | 0.095 | FAIL |

## Gates not passed → blocked

- source TEST：未触发  
- threshold v3 calibrate（正式 final）：未触发  
- known external audit / Unified recovery：未触发  
- D4 Final Freeze / Phenotype：仍 BLOCKED  

## Recommendation

1. 停止 representation 堆叠（禁止 C4+/更大 backbone）。  
2. 进入 **DATASET_CONFOUNDING_AUDIT**：核查 Stained pos/neg 是否与采集条件（分辨率、亮度、色温）强耦合。  
3. 在确认前，**不要**切换 active stain detector 或 policy。
