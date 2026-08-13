# D3-E Freeze Report — Original-Image Inference + Unletterbox + ROI

## Status

**D3-E = PASS**

机器可读：`docs/D3_E_FREEZE_STATS.json`  
推理契约：`docs/D3_E_INFERENCE_CONTRACT.md`

## Versions / Checkpoint

| Field | Value |
|---|---|
| Stage | **D3-E** |
| Segmentation contract | v1.0（D3-A，geometry 复用） |
| Training contract | v1.0（只读） |
| Frozen checkpoint | `runs/segmentation/d3c/baseline/best.pt`（本地，gitignore） |
| Run ID | **d3c-resnet34-unet-seed20260813** |
| Config hash | **a26934531e6643f6** |
| Model | ResNet34-UNet（ImageNet encoder @ train） |
| Threshold | **0.5**（frozen，未改） |
| Input | 384×384 letterbox |
| Base commit before D3-E | `f97a926`（D3-C Freeze） |

## Discipline

本阶段 **未**：

- 重新训练 / fine-tune
- 修改 D3-C weights
- 修改 threshold / 做 threshold search
- 修改 D2 split / D3-A preprocess contract
- 进入 D4 Input Guard / phenotype

Test set 访问用途 = **engineering regression only**（验证 unletterbox 实现，非模型选择）。

## Inference Contract（摘要）

| Item | Contract |
|---|---|
| EXIF | `ImageOps.exif_transpose` → RGB |
| Letterbox | 与 D3-A 同一 rounding（`geometry.py`） |
| Restoration | probability → remove pad → **bilinear** → original → **threshold 0.5** |
| Mask | original H×W，`{0,1}` |
| BBox | **xyxy exclusive** `[x1,y1,x2,y2)` |
| ROI margin | **0.05**（同时输出 tight + roi） |
| Largest component | **default true**（4-connected） |
| Phenotype RGB | **仅** original RGB crop；禁止用 normalized 384 tensor |

## Real-image Integration

从 frozen test 选取 BioHit 5 + TongueSet3 5：

| Check | Result |
|---|---|
| all status=success | **PASS** |
| mask shape == original | **PASS** |
| ROI image/mask shape 一致 | **PASS** |

输出：`runs/segmentation/d3e/integration/`（gitignore）

## 130-image Original-Resolution Regression

报告：`reports/d3/d3e_inference_regression.json`

| Metric | D3-C (model-space) | D3-E (original-res) | Δ |
|---|---:|---:|---:|
| Overall Dice | 0.9748 | **0.9748** | **+0.00002** |
| BioHit Dice | 0.9539 | **0.9537** | −0.00026 |
| TongueSet3 Dice | 0.9811 | **0.9812** | +0.00010 |

| Gate | Limit | Result |
|---|---|---|
| Overall drop | ≤ 0.01 | **PASS** |
| BioHit drop | ≤ 0.02 | **PASS** |
| TongueSet3 drop | ≤ 0.02 | **PASS** |
| invalid_bbox | 0 | **0** |
| empty_roi (success) | 0 | **0** |
| empty_predictions | — | **0** |
| invalid_masks | 0 | **0** |

### ROI coverage（GT inside predicted ROI）

| Stat | Value |
|---|---:|
| mean | **0.9924** |
| median | **1.0** |
| p10 | **1.0** |
| sanity (>0.95) | **PASS** |

## Known Failure：`biohit::278.bmp`

报告：`reports/d3/d3e_known_failure_analysis.json`

| Field | Value |
|---|---|
| original_size | 768×576 |
| GT foreground ratio | ≈0.782 |
| Pred foreground ratio | ≈0.219 |
| Intersection pixels | 1050（几乎不重叠） |
| Dice | ≈**0.0047** |
| failure_category | **undetermined** |
| Geometry restore | shape OK；**非** unletterbox bug |

未为该样本调整 threshold / 管线。Overlay：`runs/segmentation/d3e/regression/known_failure/`

## Pytest

```text
131 passed
```

（含 D3-E geometry / EXIF / checkpoint fail-fast / deterministic 等）

## Acceptance Gate

| Gate | Result |
|---|---|
| frozen D3-C checkpoint 未修改 | PASS |
| model strict load | PASS |
| train/inference geometry 一致 | PASS |
| EXIF + RGB | PASS |
| letterbox metadata 完整 | PASS |
| landscape / portrait / odd-pad round-trip | PASS |
| restored mask shape + binary | PASS |
| original RGB 未 normalize 污染 | PASS |
| bbox / ROI margin / 不越界 | PASS |
| empty prediction 结构化 | PASS |
| connected-component 固定 | PASS |
| inference deterministic | PASS |
| checkpoint mismatch fail-fast | PASS |
| real-image integration | PASS |
| 130 regression + Dice gates | PASS |
| invalid bbox / empty ROI = 0 | PASS |
| biohit::278 单独记录 | PASS |
| pytest PASS | PASS |
| Freeze Report / Stats | PASS |

## 阶段判断

Geometry restoration / original-resolution regression / ROI extraction **均 PASS**。

**D3 segmentation stage 可正式完成。**

后续经确认后才进入：

- **D4 Input Guard + Quality Gate**

本阶段 **STOP**，未自动进入 D4 / 染苔 / phenotype / API。

## Outputs

```text
docs/D3_E_INFERENCE_CONTRACT.md
docs/D3_E_FREEZE_REPORT.md
docs/D3_E_FREEZE_STATS.json

# gitignored
runs/segmentation/d3e/
reports/d3/d3e_inference_regression.json
reports/d3/d3e_known_failure_analysis.json
```
