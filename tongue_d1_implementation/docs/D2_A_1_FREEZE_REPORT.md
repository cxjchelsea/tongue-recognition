# D2-A.1 Freeze Report — Cleaning Semantics Patch

## Versions

| Field | Value |
|---|---|
| Stage | **D2-A.1** |
| Input Contract | D1.1 / Manifest v1.1 |
| Cleaning Policy | **v1.1**（`configs/cleaning_policy_v1.yaml`） |
| Prior stage | D2-A（`docs/D2_A_FREEZE_REPORT.md` 保留为历史） |

机器可读：`docs/D2_A_1_FREEZE_STATS.json`

## What changed

### 1. Spatial semantics

旧语义错误：

```text
同 task + 同 label + 不同 geometry → spatial conflict
```

新语义：

| Case | 处理 |
|---|---|
| identical geometry | dedup，合并 `origin_sample_id` |
| 同 task/label、不同 geometry | **合法 multi-instance**，全部保留，**不是 conflict** |
| review | 第一版无可靠自动判定 → `review_groups=0` |

明确：**multi-instance geometry ≠ conflict**

### 2. Label conflict policy 统一

| Item | Value |
|---|---|
| Policy key | `conflict_policy` |
| Value | `drop_conflicted_facts_from_clean` |
| Behavior | 冲突 fact → conflict_report；**不**写入 `labels_clean`；同 sample 非冲突互补监督保留 |
| Unknown policy | fail-fast `ValueError` |

旧值 `exclude_from_train_eligible` 已移除（与真实代码不一致）。

## Real-data numbers

| Metric | D2-A | D2-A.1 | Note |
|---|---:|---:|---|
| samples_after | 22235 | **22235** | 不变 |
| labels_after | 87736 | **87736** | 不变 |
| spatial_after | 24363 | **24363** | 不变（旧实现本就保留多框） |
| label_conflicts | 15 | **15** | 仍完整上报并 drop 冲突 fact |
| spatial_conflicts（旧名） | 1605 | **废除** | 不再作为 conflict |

### Spatial reclassification

| 旧 | 新 |
|---|---|
| spatial_conflicts = 1605 | — |
| — | spatial_identical_deduped = **2** |
| — | spatial_multi_instance_groups = **1606** |
| — | spatial_multi_instance_annotations = **5329** |
| — | spatial_review_groups = **0** |

说明：旧报告在统计 conflict 时排除了 `segmentation.tongue`；D2-A.1 将所有同 task/label 多几何组计为 multi-instance，故 groups=1606（≈旧 1605 + 1）。

### Supervision pools（未变）

silver 71160 / pseudo 27124 / auxiliary 9177 / gold_candidate 4473 / external_holdout 184

Guards 仍成立：L2=pseudo，DSCT=external_holdout，Stained≠coating.color，TMC 无缺框阴性。

## Validation

| Check | Result |
|---|---|
| pytest | **31 passed** |
| validate-clean | **PASS** |
| samples/labels/spatial 数量变化 | 相对 D2-A **无变化**（可解释：仅语义/统计修正） |

## Completion checklist

- [x] 多 bbox 同 task/label 不再自动定义成 conflict
- [x] legitimate multi-instance 全部保留
- [x] identical bbox 正确 dedup
- [x] spatial statistics 语义正确；1605 已重分类
- [x] conflict_policy 与代码一致并被消费
- [x] label conflict 只删除冲突 fact
- [x] 非冲突互补 supervision 不丢失
- [x] 回归：L2 / DSCT / Stained / TMC
- [x] validate-clean + pytest PASS
- [x] D2-A.1 Freeze Report / Stats 已生成

## Stop

D2-A.1 完成。**未进入 D2-B/C split / 训练。**
