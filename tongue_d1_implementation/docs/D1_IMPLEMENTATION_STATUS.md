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

## 真实数据收尾仍需执行

- [ ] 将 example config 改成实际路径
- [ ] 在 8 个真实数据目录跑一次 build
- [ ] 修正实际目录/列名与 Adapter 假设的差异
- [ ] 根据 DSCT 原始说明确认 0/1 语义
- [ ] 确认 TonguExpert `dark` 是否需要独立 canonical 语义
- [ ] 确认 TMC `huataishe` 是否以及如何进入 V1
- [ ] 清理全部 unexpected/unmapped warning
- [ ] `validate-contract --strict` 通过
- [ ] `validate-manifest` 通过
- [ ] Freeze D1 Contract v1.0

## Freeze 门槛

- needs_review = 0
- unmapped source label = 0
- NA -> negative = 0
- TonguExpert L2 -> gold = 0
- raw mutation = 0
- referential integrity errors = 0
