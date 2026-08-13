# D3-C Freeze Report — Full Baseline Training + Frozen Test Evaluation

## Versions

| Field | Value |
|---|---|
| Stage | **D3-C** |
| Data / Segmentation contract | v1.0（D3-A，只读） |
| Split policy | v1.0（D2-B/C，只读） |
| Training contract | v1.0（`configs/segmentation_train_v1.yaml`） |
| Run ID | **d3c-resnet34-unet-seed20260813** |
| Config hash | **a26934531e6643f6** |
| Seed | **20260813** |
| Base code commit | `a3ef144`（D3-B；本阶段代码待单独 commit） |

机器可读：`docs/D3_C_FREEZE_STATS.json`

## Discipline（必须）

Test set was **not** used for:

- training
- early stopping
- checkpoint selection
- hyperparameter tuning
- threshold tuning

Test was evaluated **only once** after `best.pt` was frozen.

`test_access_count = 1`  
`threshold = 0.5`（固定）

## Model / Environment

| Item | Value |
|---|---|
| Model | U-Net + ResNet34 ImageNet |
| Parameters | 24,436,369 |
| Device | **cuda** / NVIDIA GeForce RTX 4090 |
| Torch / CUDA | 2.9.0+cu126 / 12.6 |
| torchvision / smp | 0.24.0+cu126 / 0.5.0 |
| Batch size | **4** |
| AMP | true |
| Input | 384×384 letterbox |

## Training

| Item | Value |
|---|---|
| planned_epochs | 50 |
| actual_epochs | **50** |
| early_stopped | **false** |
| resumed | false |
| best_epoch | **44** |
| best_val_loss | **0.0224** |
| best_val_dice | **0.9844** |
| best_val_iou（epoch44） | ≈0.98（见 history） |
| BioHit val Dice @ best | **0.9875** |
| TongueSet3 val Dice @ best | **0.9835** |
| last_val_dice | 0.9843 |
| overfit_warning | true（启发式；末段 val 仍高位平台） |

Train samples=1039，Val=130；训练期 **未构建 test loader**。

## Frozen Test（once）

| Split | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| **Overall** | **0.9748** | **0.9585** | **0.9743** | **0.9761** |
| BioHit | **0.9539** | **0.9415** | 0.9563 | 0.9520 |
| TongueSet3 | **0.9811** | **0.9636** | 0.9797 | 0.9834 |

| Extra | Value |
|---|---:|
| domain_gap_dice | **0.0271**（良好） |
| empty_predictions | **0** |
| near_full_predictions | **0** |
| overall median Dice | 0.9865 |
| overall p10 Dice | 0.9701 |

### Foreground-size Dice

| Bucket | n | mean Dice |
|---|---:|---:|
| small | 43 | 0.9748 |
| medium | 43 | 0.9861 |
| large | 44 | 0.9638 |

### Worst cases（top）

| sample_id | dataset | dice | fg_ratio |
|---|---|---:|---:|
| biohit::278.bmp | biohit | **0.0052** | 0.7815 |
| tongueset3::1862.jpg | tongueset3 | 0.8202 | 0.0706 |
| tongueset3::126.jpg | tongueset3 | 0.8709 | 0.1281 |
| tongueset3::1293.jpg | tongueset3 | 0.9460 | 0.4067 |
| tongueset3::1648.jpg | tongueset3 | 0.9571 | 0.2718 |

说明：`biohit::278.bmp` 为极端失败个案，拉高 BioHit std；overall / domain 仍过 gate。

## Baseline Gate

| Gate | Criterion | Result |
|---|---|---|
| Overall target | Dice ≥ 0.95 | **PASS**（0.9748） |
| Domain min BioHit | ≥ 0.90 | **PASS**（0.9539） |
| Domain min TongueSet3 | ≥ 0.90 | **PASS**（0.9811） |

**baseline_status = TARGET_PASS**

（工程 gate，非临床有效性声明）

## Validation

| Check | Result |
|---|---|
| pytest | **111 passed** |
| preflight counts | PASS（与 D3-A Freeze 一致） |
| best reload | PASS |
| test evaluation | PASS（一次） |

## Outputs（gitignore）

```text
runs/segmentation/d3c/baseline/
  best.pt
  last.pt
  history.json
  run_metadata.json
  training_summary.json
  val_metrics.json
  test_metrics.json
  test_per_image_metrics.parquet
  failure_cases/worst_cases.json
  preflight.json

reports/d3/training_curve.json
```

## Recommendation（不自动继续）

当前已 **TARGET_PASS**，从指标上可进入：

- **D3-E ROI inference / unletterbox pipeline**（把 mask 映回原图）

或先做：

- **D3-D** 针对 `biohit::278.bmp` 等 failure cases 的改进实验

**未经确认不自动进入下一阶段。**

## Freeze checklist

- [x] D2/D3-A 契约未改；split/manifest 不变
- [x] full baseline training 完成（50 epochs）
- [x] 仅 train 训练；仅 val 选模
- [x] best = val Dice；reload PASS
- [x] test 仅 freeze 后访问 1 次；threshold=0.5
- [x] overall + BioHit + TongueSet3 metrics
- [x] domain gap / foreground-size / failure cases
- [x] empty/full prediction audit
- [x] baseline gate = TARGET_PASS
- [x] pytest PASS
- [x] Freeze Report 已生成
