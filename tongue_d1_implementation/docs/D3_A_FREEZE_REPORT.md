# D3-A Freeze Report — Tongue Segmentation Dataset & Training Contract

## Versions

| Field | Value |
|---|---|
| Stage | **D3-A** |
| Input Contract | D1.1 |
| Cleaning Policy | v1.1（只读） |
| Split Policy | v1.0（只读，**不重划分**） |
| Segmentation Contract | **v1.0**（`configs/segmentation_v1.yaml`） |
| Package | 0.3.0 |
| Seed | **20260813** |
| Base code commit | `14463be`（D2-B/C；本阶段代码待单独 commit） |

机器可读：`docs/D3_A_FREEZE_STATS.json`

## Scope

本阶段完成：

- Segmentation dataset contract
- BioHit + TongueSet3 训练样本选择
- image-mask pairing / mask normalization
- letterbox resize + joint transform
- Dataset / DataLoader / metrics
- experiment config schema + reproducibility
- audit / smoke / validators / tests
- Freeze Report

**未做**：完整 baseline 训练、架构对比、学习率搜索、test 调参。

## Data selection

仅：

| Dataset | Role |
|---|---|
| BioHit | gold segmentation |
| TongueSet3 | gold segmentation |

明确排除：

TMC / TongueDx / Tooth-Marked / DSCT / Stained / TonguExpert masks（暂不作为 D3 baseline 主监督）

## Real-data counts

| Metric | Value |
|---:|---:|
| total_segmentation_samples | **1299** |
| train / val / test | **1039 / 130 / 130** |

### BioHit

| split | n |
|---|---:|
| train | **240** |
| val | **30** |
| test | **30** |
| total | **300** |

### TongueSet3

| split | n |
|---|---:|
| train | **799** |
| val | **100** |
| test | **100** |
| total | **999** |

Split 100% 继承自 `data/splits/v1/split_assignments.parquet`。

## Pairing / integrity

| Check | Result |
|---|---:|
| image-mask pairing | **100%**（1299/1299） |
| missing_images | **0** |
| missing_masks | **0** |
| shape_mismatches | **0** |
| empty_masks | **0** |
| full_masks | **0** |
| sample_leakage | **0** |
| md5_leakage | **0** |

### Multi-mask resolution

`tongueset3::102.jpg` 与 alias `1020.jpg` 同 MD5，D2 合并后出现 2 masks。

Policy：`prefer_canonical_origin`

- kept：`...tongue::0`（origin=`tongueset3::102.jpg`）
- dropped：alias origin=`tongueset3::1020.jpg`
- 已写入 audit，非静默跳过

## Mask pixels & normalization

Rule：**`mask > 0 → 1`**（禁止 `mask == 255`）

| Dataset | Observed raw values |
|---|---|
| BioHit | `{0,1}` 与 `{0,255}` 并存（含 bool 读） |
| TongueSet3 | `{0,1}`（多通道） |

两者均正确归一化为 binary `{0,1}`。

## Foreground ratio

| stat | value |
|---|---:|
| min | 0.0347 |
| p01 | 0.0623 |
| p10 | 0.1253 |
| median | 0.2559 |
| p90 | 0.5842 |
| p99 | 0.7863 |
| max | 0.8853 |
| mean | 0.3121 |

极端样本仅 warning（1 条），**不自动删除**。

## Resize / padding / augmentation

| Item | Policy |
|---|---|
| input | **384×384** |
| resize | **letterbox**（keep aspect + center pad） |
| image interp | bilinear |
| mask interp | **nearest** |
| image pad | 0 |
| mask pad | **0（background）** |
| train aug | hflip / small rotation±10° / scale 0.9–1.1 / mild brightness-contrast |
| val/test aug | **disabled** |
| geometry | image/mask **joint sync** |

## Metrics / training contract（预定义，本阶段不训练）

| Item | Value |
|---|---|
| primary metric | **Dice**（per-image mean） |
| also | IoU / Precision / Recall |
| threshold | 0.5（可配置） |
| loss (D3-B) | 0.5 BCE + 0.5 Dice |
| architecture candidate | ResNet34-UNet + ImageNet |
| optimizer | AdamW |
| device | auto（CUDA if available） |

### D3-B/C acceptance targets（工程 gate，非临床声明）

| Gate | Target |
|---|---|
| Minimum | overall test Dice **≥ 0.90** |
| Target | overall test Dice **≥ 0.95** |
| Required | 分别报告 BioHit / TongueSet3 test Dice |

Test set 纪律：只用于最终评估；模型选择仅看 val。

## Validation

| Check | Result |
|---|---|
| validate-segmentation | **PASS** |
| segmentation smoke test | **PASS**（BioHit/TongueSet3 × train/val/test） |
| pytest | **74 passed** |

## Outputs

```text
data/segmentation/v1/
  segmentation_manifest.parquet
  segmentation_metadata.json

configs/segmentation_v1.yaml

reports/d3/
  segmentation_dataset_audit.json
  segmentation_smoke_test.json

src/tongue_data/segmentation/
  config.py / manifest.py / dataset.py / transforms.py
  metrics.py / mask_ops.py / reproducibility.py
  builder.py / validators.py
```

## Freeze checklist

- [x] segmentation_v1.yaml 固定
- [x] 只使用 BioHit + TongueSet3 gold segmentation
- [x] D2 split 100% 继承
- [x] sample/MD5 leakage = 0
- [x] pairing / missing / empty / shape = 0
- [x] mask normalization 正确（含 TongueSet3 value=1）
- [x] resize 后 mask 仍 binary
- [x] train joint transform；val/test deterministic
- [x] Dataset/DataLoader smoke PASS
- [x] Dice/IoU metrics tests PASS
- [x] per-domain audit 已生成
- [x] pytest PASS
- [x] Freeze Report 已生成

## Stop

D3-A 完成。**不自动进入 D3-B**。

数据与训练契约已就绪，可在确认后进入 D3-B Baseline Model Implementation。
