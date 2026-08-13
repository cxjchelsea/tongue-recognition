# D3-B Freeze Report — ResNet34-UNet Trainer + Tiny Overfit

## Versions

| Field | Value |
|---|---|
| Stage | **D3-B** |
| Segmentation data contract | v1.0（D3-A，只读） |
| Training contract | **v1.0**（`configs/segmentation_train_v1.yaml`） |
| Package | 0.3.1 |
| Seed | **20260813** |
| Config hash | `8d49879293b3a655` |
| Base code commit | `2f55230`（D3-A；本阶段代码待单独 commit） |

机器可读：`docs/D3_B_FREEZE_STATS.json`

## Scope

本阶段证明 training pipeline 正确性：

- model forward / raw logits
- BCE + Dice loss / backward
- AMP / optimizer / scheduler
- checkpoint / resume
- metric pipeline（含 per-domain）
- **tiny-set overfit ≥ 0.95**

**未做**：完整 50 epoch 训练、test 评估、架构对比、超参搜索。

## Model

| Item | Value |
|---|---|
| Architecture | **U-Net** |
| Encoder | **resnet34** |
| Encoder weights | **imagenet**（失败不静默降级） |
| In / Out | 3 → 1 logits `[B,1,H,W]` |
| Total parameters | **24,436,369** |
| Trainable parameters | **24,436,369** |

Dependency：`segmentation-models-pytorch`（optional extra `train`）

## Loss / Optim

| Item | Value |
|---|---|
| Loss | 0.5·BCEWithLogits + 0.5·SoftDice（smooth=1e-6） |
| Soft Dice | continuous probability（**禁止 threshold 后反传**） |
| Optimizer | AdamW lr=1e-3, wd=1e-4 |
| Scheduler | ReduceLROnPlateau（monitor val_dice, mode=max） |
| Checkpoint monitor | **val_dice only**（不用 test） |

## Device / AMP

| Item | Value |
|---|---|
| Device | **cuda** |
| GPU | NVIDIA（本机 CUDA 可用） |
| Torch | 2.9.0+cu126 |
| AMP | **true**（CPU 自动关闭） |
| Metric dtype | float32（避免 AMP float16 在 384² reduce 溢出） |

## Smoke training（真实 D3-A dataset）

| Item | Value |
|---|---|
| train batches | 2 |
| val batches | 2 |
| epochs | 1 |
| result | **PASS** |
| resume | **PASS**（epoch 连续） |
| best/last.pt | 已写入 `runs/segmentation/d3b/smoke/`（gitignore） |

## Tiny overfit（核心门禁）

| Item | Value |
|---|---|
| samples | **16**（BioHit 8 + TongueSet3 8） |
| selection | deterministic（dataset+sample_id 排序） |
| augmentation | **disabled** |
| steps | **160** / max 500 |
| initial_loss | **0.6845** |
| final_loss | **0.0610** |
| final_dice | **0.9510** |
| target | ≥ 0.95 |
| result | **PASS** |

说明：tiny overfit 只验证 pipeline 可学习，不是泛化指标；未使用 val/test 做决策。

## Checkpoint / Resume

- `best.pt`：仅当 val_dice 提升时更新
- `last.pt`：每 epoch / overfit 结束更新
- 内容含：model / optimizer / scheduler / scaler / epoch / global_step / best_val_dice / config / config_hash / seed / history / torch_version
- resume 后 epoch 连续：单元测试 + smoke 均 **PASS**

## Per-domain metric pipeline

Validation aggregator 支持：

- overall
- biohit
- tongueset3

单元测试覆盖聚合正确性。正式域报告留给 D3-C。

## Important note（phenotype）

Segmentation training augmentation / normalized tensors **不得**作为后续舌色等 phenotype 输入。  
未来必须：预测 mask → 映射回 **original RGB** → 再提取 ROI。

## Validation

| Check | Result |
|---|---|
| unit tests (D3-B) | 24 passed |
| full pytest | **98 passed** |
| real smoke | **PASS** |
| tiny overfit | **PASS**（Dice≥0.95 且 loss 下降） |

## Outputs

```text
configs/segmentation_train_v1.yaml

src/tongue_data/segmentation/
  model.py
  train_config.py
  training/
    losses.py optimizer.py trainer.py
    checkpoint.py history.py evaluation.py

tests/data_contract/test_training_d3b.py

runs/segmentation/d3b/   # gitignored
  smoke/
  tiny_overfit/
```

## Freeze checklist

- [x] ResNet34-UNet factory
- [x] pretrained 行为明确（无静默降级）
- [x] raw logits
- [x] BCE+Dice / backward / finite grads
- [x] optimizer / scheduler / AMP
- [x] train + val loop
- [x] val Dice monitor + best/last
- [x] full checkpoint + resume
- [x] config hash + history
- [x] per-domain metrics infra
- [x] test set 未用于选择
- [x] real smoke PASS
- [x] tiny overfit ≥ 0.95 且 loss 下降
- [x] pytest PASS
- [x] Freeze Report

## Stop

D3-B 完成。**不自动进入 D3-C** 完整训练。

从 pipeline 正确性角度：已具备进入 D3-C Full Baseline Training + Evaluation 的条件（待确认）。
