# D2-B/C Freeze Report — Leakage-Safe Groups + Train/Val/Test Split

## Versions

| Field | Value |
|---|---|
| Stage | **D2-B/C** |
| Input Contract | D1.1 Manifest + **D2-A.1 clean**（`data/processed/v1`） |
| Cleaning Policy | v1.1（只读消费，未修改） |
| Split Policy | **v1.0**（`configs/split_policy_v1.yaml`） |
| Seed | **20260813** |
| Base code commit | `fe02d79`（D2-A.1 Freeze；本阶段代码待单独 commit） |

机器可读：`docs/D2_BC_FREEZE_STATS.json`

## Scope

本阶段只构建：

- leakage-safe `split_group_id`
- train / val / test / external_holdout
- distribution audit
- leakage validation
- split-level effective supervision

**未做**：模型训练、segmentation training、增广、重采样、class weight、阈值搜索。

## Inputs（未改写）

```text
data/processed/v1/
  samples_clean.parquet
  labels_clean.parquet
  spatial_clean.parquet
  dedup_decisions.parquet
  supervision_assignments.parquet
  cleaning_metadata.json

data/manifests/v1/samples.parquet   # 只读：alias patient identity
```

## Outputs

```text
data/splits/v1/
  split_groups.parquet
  sample_group_assignments.parquet
  split_assignments.parquet
  split_supervision_assignments.parquet
  split_metadata.json

reports/d2/
  group_audit.json
  split_report.json
  leakage_report.json
  task_distribution.json
```

## Leakage group 构造

使用 union-find / connected components：

| 关系 | 行为 |
|---|---|
| 同 TongueDx `patient_id` | union |
| 同 MD5 | union |
| dedup alias → canonical + 原始 patient | union（避免 canonical 选择丢失患者关系） |
| DSCT | `forced_split=external_holdout`，不进 80/10/10 |
| missing patient | **fallback 到 sample identity**，绝不合并为 `patient::unknown` |

最终节点连通分量 → 稳定命名 `grp::<min_member_sample_id>`。

## Real-data numbers

### Samples / groups

| Metric | Value |
|---:|---:|
| samples_total | **22235** |
| leakage_groups | **21776** |
| train | **17715** |
| val | **2214** |
| test | **2214** |
| external_holdout | **92** |
| ratios_actual | 0.8000 / 0.1000 / 0.1000 |

### TongueDx

| Metric | Value |
|---:|---:|
| patient_groups | **4633** |
| patients_with_multiple_images | 450 |
| max_images_per_patient | 4 |
| missing_patient_ids | **0** |
| canonical_with_multiple_origin_patient_ids | **17**（已 union） |
| patients_train / val / test | **3896 / 352 / 385** |

### TonguExpert

| Metric | Value |
|---:|---:|
| samples_with_L1 | 1586 |
| samples_with_L2 | 5992 |
| samples_with_both | 1586 |
| L2 pseudo total | 27124 |
| pseudo effective_for_train（仅 train sample） | 21690 |
| pseudo effective_for_val / test | **0 / 0** |

### DSCT

| Metric | Value |
|---:|---:|
| external_holdout samples | **92** |
| train / val / regular test | **0 / 0 / 0** |

### Per-dataset split

| Dataset | train | val | test | external_holdout |
|---|---:|---:|---:|---:|
| tonguedx | 4074 | 509 | 509 | 0 |
| tmc_tongue | 5260 | 657 | 658 | 0 |
| tonguexpert | 4794 | 599 | 599 | 0 |
| stained_coating | 1548 | 194 | 193 | 0 |
| tooth_marked | 1000 | 125 | 125 | 0 |
| tongueset3 | 799 | 100 | 100 | 0 |
| biohit | 240 | 30 | 30 | 0 |
| dsct | 0 | 0 | 0 | **92** |

### Unstratifiable labels

| Fact | group_count |
|---|---:|
| `coating.color:light_yellow:positive` | 2 |
| `tongue_body.color:normal:positive` | 1 |

未拆 leakage group。

### Distribution warning（非 leakage failure）

| Fact | max_absolute_prevalence_deviation |
|---|---:|
| `tongue_body.color::pale` | 0.1607（train≈0.319 vs val/test≈0.158） |

安全优先：未为凑 prevalence 拆 patient group。

## Leakage gates

| Check | Count |
|---|---:|
| patient_leakage | **0** |
| md5_leakage | **0** |
| sample_leakage | **0** |
| group_leakage | **0** |
| pseudo_leakage | **0** |
| external_holdout_leakage | **0** |

## Validation

| Check | Result |
|---|---|
| validate-split | **PASS** |
| pytest | **49 passed** |
| Stained coating.color effective | **0**（仍 quality-only） |

## Freeze checklist

- [x] D2-A.1 Freeze 输入未修改
- [x] split_policy_v1 已建立
- [x] leakage component / split group 已实现
- [x] TongueDx patient-level grouping 正确
- [x] duplicate alias patient identity 未丢失（17 collisions union）
- [x] missing patient fallback 正确（真实数据 missing=0）
- [x] TonguExpert L1/L2 同 sample 同 split
- [x] pseudo 只可用于 train sample
- [x] pseudo 不进入正式 val/test supervision
- [x] DSCT 100% external_holdout
- [x] split deterministic（seed=20260813）
- [x] train/val/test 在 group 粒度生成
- [x] explicit negatives 参与 distribution
- [x] task distribution audit 已生成
- [x] rare/unstratifiable label 已报告
- [x] 六类 leakage = 0
- [x] validate-split PASS
- [x] pytest PASS
- [x] D2-B/C Freeze Report 已生成

## Stop

D2-B/C 完成。**不自动进入 D3**。

从数据契约角度：BioHit / TongueSet3 已具备独立 train/val/test，可在确认后进入 D3 Tongue Segmentation Baseline。
