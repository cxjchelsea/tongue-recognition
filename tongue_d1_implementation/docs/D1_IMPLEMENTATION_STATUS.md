# D1 Implementation Status

## 已完成代码

- [x] Ontology v1
- [x] 8 个 Mapping
- [x] 三表 Schema
- [x] Adapter framework
- [x] TMC / TongueDx / TonguExpert / BioHit / TongueSet3 / Tooth-Marked / DSCT / Stained adapters
- [x] Manifest Builder
- [x] Validators
- [x] CLI
- [x] 自动测试

## 真实数据收尾

- [x] 将 example config 改成实际路径（`configs/datasets_v1.local.yaml`）
- [x] 在 8 个真实数据目录跑一次 build
- [x] 修正实际目录/列名与 Adapter 假设的差异（TongueDx 多 CSV、TMC 限定 shezhenv3-txt、BioHit/Tooth/Stained 嵌套路径）
- [x] 根据 DSCT 原始说明确认 0/1 语义（0=slight/mild，1=serious/severe；均衍生 crack.present）
- [x] 确认 TonguExpert `dark` 独立映射为 `tongue_body.color.dark`（不等于 purple）
- [x] 确认 TMC `huataishe`（滑苔）V1 `excluded`（不可等同腻苔）
- [x] 清理全部 unexpected/unmapped warning（build warnings_count=0）
- [x] `validate-contract --strict` 通过
- [x] `validate-manifest` 通过
- [x] Freeze D1 Contract v1.0（门槛已全部满足）
- [x] **D1.1** Explicit Negative Supervision Patch（Contract/Manifest v1.1）
  - TonguePale/FurYellow 的源值 0 以 `value=0` 保留
  - binary 映射语义不变；NA 仍不落盘
  - 详见 `docs/D1_1_FREEZE_REPORT.md`

## 最近一次真实 build 摘要（D1.1）

| dataset | samples | labels | spatial |
|---|---:|---:|---:|
| biohit | 300 | 0 | 300 |
| tongueset3 | 1000 | 0 | 1000 |
| tmc_tongue | 6719 | 17073 | 17073 |
| tonguedx | 5109 | 40872 | 0 |
| tonguexpert | 5992 | 30297 | 5992 |
| tooth_marked | 1250 | 1250 | 0 |
| dsct | 95 | 190 | 0 |
| stained_coating | 2008 | 2008 | 0 |
| **合计** | **22473** | **91690** | **24365** |

产物目录：`data/manifests/v1/`

## Freeze 门槛

- needs_review = 0
- unmapped source label = 0
- NA -> negative = 0
- TonguExpert L2 -> gold = 0
- raw mutation = 0
- referential integrity errors = 0
