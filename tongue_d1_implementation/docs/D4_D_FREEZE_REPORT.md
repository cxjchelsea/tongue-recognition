# D4-D Freeze Report

## Status

**TARGET_PASS**

Stage：D4-D Color Cast + Occlusion + Unified Input Guard  
Date：2026-08-13

## Versions

| item | value |
|---|---|
| Input Guard Contract | **1.0**（未破坏字段语义） |
| Input Guard Policy | **1.3** |
| defined / implemented | **11 / 11** |
| guard_ready（system） | **true** |

## Frozen references preserved

| reference | status |
|---|---|
| D3 checkpoint hash `a26934531e6643f6` | unchanged |
| D4-B signal thresholds（e.g. focus） | unchanged |
| D4-C stain `t_clear=0.95` / `t_retake=0.96` | unchanged |
| D4-C stain checkpoint | not modified / not recalibrated |

## Color Cast

Implementation：`signal_rule` + neutral-reference（tongue mask 外）

| metric | value |
|---|---|
| status | **PASS** |
| calibration samples | 1169（BioHit+TongueSet3 train+val） |
| neutral support rate | 0.9487 |
| clean false-retake | **0.0188** |
| synthetic severe detection | **0.9271** |
| synthetic moderate detection | ~0.40 |
| warning_cast_magnitude | 20.0888 |
| retake_cast_magnitude | 28.6007 |

Rules：

- insufficient neutral support → `unavailable`（≠ PASS）
- 禁止 tongue mean RGB / phenotype 色捷径
- test 未参与 calibration

## Occlusion

Implementation：multi-evidence signal_rule

Evidence：

1. interior low-probability holes（distance-transform interior）
2. bright neutral intrusion
3. combined score；单弱证据不得单独 severe RETAKE

| metric | value |
|---|---|
| status | **PASS** |
| clean false-retake | **0.0** |
| synthetic severe detection | **1.0** |
| synthetic small retake | **0.0** |
| warning_combined_score | 0.2359 |
| retake_combined_score | 0.2862 |

## Unified test audit（frozen once）

BioHit+TongueSet3 test n=130：

| decision | count |
|---|---|
| pass | 36 |
| warning | 14 |
| retake | 80 |
| evaluation_complete=true | 120 |
| evaluation_complete=false | 10 |

注：test audit 仅工程观察，未用于调参。

## Runtime smoke

- 10 BioHit + 10 TongueSet3：OK，`guard_ready=true`
- 10 Stained coating integration smoke：OK（未重调 stain）
- `biohit::278.bmp`：final **pass**；color_cast=acceptable；occlusion=none  
  （known D3 failure，未针对其调参）

## Acceptance

- [x] 11-check unified runtime
- [x] color_cast / occlusion engineering gates PASS
- [x] unavailable != pass
- [x] evaluation_complete / guard_ready 语义分离
- [x] quality_confidence 保持 null
- [x] D4-B/C frozen artifacts 未改
- [x] pytest / validator PASS
- [x] Freeze docs

## Stop line

D4-D **TARGET_PASS** 完成。  
**STOP**：不自动进入 D4-E UI/API、phenotype、诊断模型。
