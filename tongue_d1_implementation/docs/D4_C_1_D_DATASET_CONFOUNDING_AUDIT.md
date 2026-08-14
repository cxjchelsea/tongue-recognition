# D4-C.1-D Dataset Confounding & Label Validity Audit

本阶段为**数据诊断**，未训练 ResNet / 未改 threshold / 未改 policy / 未访问 source TEST。

## Decision

| 字段 | 值 |
|---|---|
| SOURCE_CONFOUNDING_CONFIRMED | `true` |
| SOURCE_CONFOUNDING_LEVEL | `SEVERE` |
| EXISTING_DATA_RESCUABLE | `false` |
| RECOMMENDED_DATA_ACTION | `RECOLLECT_STAIN_DATASET` |

### 关键证据

1. **目录与标签完全绑定**：`染苔/` → positive_rate=1.0（n=892）；`非染苔/` → positive_rate=0.0（n=850）。
2. **Acquisition-only Logistic AUROC = 0.978**（`STRONG_CONFOUNDING_SIGNAL`）。
3. Color / Resolution / Geometry / Quality 各族 AUROC **均 ≥ 0.95**。
4. **Positive match rate 仅 13.2%**；matched pairs=118；matching 后 AUROC 仍 0.869。
5. Global acquisition **强于** local heterogeneity（0.978 vs 0.894）。

结论：现有 Stained 数据无法可靠解耦 acquisition 与 stain label。  
**不建议**仅靠 v4 模型技巧继续；应规划同协议下 stain+/stain- 成对采集。  
本阶段不实施补采/重训/policy 切换。

## Cohort

- n_audit (train+val): **1742**
- positive: **892** / negative: **850**

## Family AUROCs

| Feature set | AUROC |
|---|---:|
| Color-only | 0.9475 |
| Resolution-only | 0.9508 |
| Geometry-only | 0.9647 |
| Quality-only | 0.9582 |
| All-acquisition | **0.9785** (PR-AUC 0.9548) |

## Matching

- positive match rate: 0.132
- matched pairs: 118
- median / p95 distance: 1.187 / 1.442
- acquisition AUROC before→after: 0.978 → 0.869（Δ≈0.110）

## Notes

- Confounding = association, not causation.
- 解决混淆 ≠ 强行统一白平衡。
- STOP：不自动 v4 / 补采执行 / Final Freeze / phenotype。

产物：`reports/d4c1d/`、`docs/D4_C_1_D_AUDIT_STATS.json`。
