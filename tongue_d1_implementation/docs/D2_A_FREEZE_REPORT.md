# D2-A Freeze Report — Cleaning Policy + Duplicate Resolution + Supervision Pool

## Versions

| Field | Value |
|---|---|
| Input Contract | D1.1 / Manifest v1.1 |
| Cleaning Policy | **v1.0**（`configs/cleaning_policy_v1.yaml`） |
| Stage | **D2-A**（未做 train/val/test split） |

机器可读统计：`docs/D2_A_FREEZE_STATS.json`  
运行报告（本地构建产物，默认不入库）：`reports/d2/dedup_report.json`、`conflict_report.json`

## Scope

已完成：

- Cleaning Policy 配置化
- dataset 内 MD5 duplicate group + deterministic canonical
- label / spatial reconciliation（互补合并、相同去重、冲突上报）
- cross-dataset MD5 检查（当前 = 0，policy=`fail`）
- clean manifest + dedup_decisions + supervision_assignments
- raw **零修改**（仅 metadata decision）

未做：patient split、最终 train/val/test、训练。

## Sample cleaning

| Metric | Value |
|---|---:|
| samples_before | 22473 |
| samples_after | **22235** |
| duplicate aliases removed | **238** |
| multi-member duplicate groups | 222 |
| cross-dataset duplicates | **0** |

### Per dataset

| dataset | before | unique_md5 | after | aliases |
|---|---:|---:|---:|---:|
| biohit | 300 | 300 | 300 | 0 |
| tongueset3 | 1000 | 999 | 999 | 1 |
| tmc_tongue | 6719 | 6575 | 6575 | 144 |
| tonguedx | 5109 | 5092 | 5092 | 17 |
| tonguexpert | 5992 | 5992 | 5992 | 0 |
| tooth_marked | 1250 | 1250 | 1250 | 0 |
| dsct | 95 | 92 | 92 | 3 |
| stained_coating | 2008 | 1935 | 1935 | 73 |

`samples_before - samples_after = 238 = aliases`，可解释。

## Label / spatial

| Metric | before | after | note |
|---|---:|---:|---|
| labels | 91690 | **87736** | 主要来自 TMC 同图多框→image-level 去重；冲突事实未静默保留 |
| spatial | 24365 | **24363** | 几何完全一致去重 |
| label conflicts | — | **15** | 11 binary true/false + 4 value_mismatch（多为 TongueDx 同 MD5 标签对立） |
| spatial conflicts | — | **1605** | 同 task/label 多几何已报告，几何仍保留 |

TonguExpert labels：**30297 → 30297**（L1/L2 未互相折叠）。

## Supervision pools

| pool | count |
|---|---:|
| silver | 71160 |
| pseudo | 27124 |
| auxiliary | 9177 |
| gold_candidate | 4473 |
| external_holdout | 184 |

Guards：

- TonguExpert L2 → **仅 pseudo**（27124）
- DSCT → **仅 external_holdout**，`eligible_for_train=false`（184）
- Stained → 无 `coating.color.*` 监督资格
- TMC → 无“缺框推阴性”

## Validation

| Check | Result |
|---|---|
| `pytest -q` | **25 passed** |
| `validate-clean` | **PASS** |
| raw mutation | **0**（policy 禁止；D1 manifest 未覆盖写） |

## Outputs

```text
data/processed/v1/
  samples_clean.parquet
  labels_clean.parquet
  spatial_clean.parquet
  dedup_decisions.parquet
  supervision_assignments.parquet
  cleaning_metadata.json

reports/d2/
  dedup_report.json
  conflict_report.json
```

## Completion checklist

- [x] cleaning_policy_v1
- [x] duplicate groups + deterministic canonical
- [x] raw 0 mutation
- [x] complementary merge / identical dedupe / conflict report
- [x] spatial reconciliation
- [x] cross-dataset duplicate check
- [x] clean tables + decisions + supervision assignments
- [x] L2 pseudo / DSCT external_holdout / Stained quality-only
- [x] validators + tests + freeze report

## Stop

D2-A 已 Freeze。**未进入 D2-B/C Leakage-Safe Split。**
