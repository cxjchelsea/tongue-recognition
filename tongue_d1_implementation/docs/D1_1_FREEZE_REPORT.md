# D1.1 Freeze Report — Explicit Negative Supervision

## Versions

| Field | Value |
|---|---|
| Contract | **v1.1**（相对 D1 v1.0 的小版本修补） |
| Ontology | 1.0（未改 task type；`tongue_body.color` / `coating.color` 仍为 `multiclass_partial`） |
| TongueDx mapping | **1.1** |
| Manifest | **1.1** |
| Ingest | **1.1** |
| Package | 0.1.1 |

> D1 v1.0 仍然是历史冻结基线；D1.1 在兼容三表 Schema 的前提下补齐 explicit negative。

详细机器可读统计见同目录 `D1_1_FREEZE_STATS.json`（不含本机绝对路径与患者隐私字段）。  
`D1_1_FREEZE_STATS.json` 中的 `git_commit` 为构建时 HEAD（D1 v1.0 提交）；D1.1 代码变更在该次 build 时尚未单独提交。

## What changed

根因：`mapping_to_label_records()` 在仅有 `positive_label`、无 `negative_label` 时，对源值 `0` 直接丢弃。

修复策略（最小兼容）：

- **partial attribute**（`TonguePale` / `FurYellow`）：保留 `canonical_label=pale|yellow`，`value=0/1`
- **binary**（已有 `negative_label`）：保持 `canonical_label=true|false` 且 `value=1`
- **NA**：仍不生成监督行（`normalize_na("NA") is None`）
- 禁止 `not-pale → normal`、`not-yellow → white`

## Build summary

| Metric | D1 v1.0 | D1.1 |
|---|---:|---:|
| samples | 22473 | 22473 |
| labels | 82929 | **91690** |
| spatial | 24365 | 24365 |
| warnings | 0 | 0 |
| labels delta | — | **+8761**（= TonguePale 负监督 4485 + FurYellow 负监督 4276） |

### 8-source sample counts（不变）

| dataset | samples | labels | spatial |
|---|---:|---:|---:|
| biohit | 300 | 0 | 300 |
| tongueset3 | 1000 | 0 | 1000 |
| tmc_tongue | 6719 | 17073 | 17073 |
| tonguedx | 5109 | **40872** | 0 |
| tonguexpert | 5992 | 30297 | 5992 |
| tooth_marked | 1250 | 1250 | 0 |
| dsct | 95 | 190 | 0 |
| stained_coating | 2008 | 2008 | 0 |

## TongueDx supervision recovery

| Field | positive | explicit_negative | unavailable |
|---|---:|---:|---:|
| TonguePale | 624 | **4485**（v1.0 丢失） | 0 |
| FurYellow | 833 | **4276**（v1.0 丢失） | 0 |

Binary 回归（manifest 与源 CSV 一致）：

| Field | positive | explicit_negative |
|---|---:|---:|
| Crack | 4274 | 835 |
| Toothmark | 2984 | 2125 |
| Spot | 2431 | 2678 |
| Ecchymosis | 471 | 4638 |
| FurThick | 4973 | 136 |
| TipSideRed | 2243 | 2866 |

Guards：

- `FurYellow → white` = 0
- `TonguePale → normal` = 0

## Validation

| Check | Result |
|---|---|
| `validate-contract --strict` | PASS |
| 8-source build | PASS（warnings=0） |
| `validate-manifest` | PASS |
| `pytest -q` | **13 passed** |

## Completion checklist

- [x] TonguePale=0 不再丢失
- [x] FurYellow=0 不再丢失
- [x] NA 没有被转换为 0
- [x] 不把 not-pale 映射为 normal
- [x] 不把 not-yellow 映射为 white
- [x] existing binary mappings 无回归
- [x] TonguExpert L2 仍为 pseudo
- [x] validate-contract --strict PASS
- [x] 8-source build PASS
- [x] build warnings = 0
- [x] validate-manifest PASS
- [x] pytest 全部 PASS
- [x] D1.1 Freeze Report 已生成

## Boundary

D1.1 仅完成 explicit negative supervision patch。  
**未开始** D2 去重 / split / 训练。
