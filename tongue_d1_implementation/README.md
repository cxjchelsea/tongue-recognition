# Tongue D1 Data Contract

这是舌诊视觉表型模型 V1 的 D1 可运行实现。  
当前契约小版本：**D1.1 / Contract v1.1**（显式负监督；见 `docs/D1_1_FREEZE_REPORT.md`）。  
清洗阶段：**D2-A.1**（Cleaning Policy v1.1；多实例 bbox ≠ conflict；见 `docs/D2_A_1_FREEZE_REPORT.md`）。尚未做 train/val/test split。

## 已实现

- Tongue Phenotype Ontology v1
- 8 个数据源 source mapping
- `samples / labels / spatial_annotations` 三表契约
- 8 数据源 Adapter
- Manifest Builder
- Contract / Manifest Validator
- NA 语义保护
- TonguExpert L1/L2 隔离
- TMC bbox 只产生正向 evidence，不从“无框”推断阴性
- CLI 与 pytest

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 校验数据契约

```bash
tongue-data validate-contract
```

当前会对待确认映射给 warning。正式冻结 D1：

```bash
tongue-data validate-contract --strict
```

严格模式必须在所有 `needs_review` 解决后才能通过。

## 配置真实数据

本仓库已提供本机配置：

```text
configs/datasets_v1.local.yaml
```

（由 `configs/datasets_v1.example.yaml` 按实际目录生成；TongueDx 使用 fold1 train/val + test。）

## 构建统一 Manifest

```bash
tongue-data build --config configs/datasets_v1.local.yaml --output data/manifests/v1
```

输出：

```text
samples.parquet
labels.parquet
spatial_annotations.parquet
dataset_statistics.json
mapping_statistics.json
build_metadata.json
```

## 校验构建产物

```bash
tongue-data validate-manifest --manifest-dir data/manifests/v1
```

## D2-A 清洗（不改 raw，不生成最终 split）

```bash
tongue-data clean \
  --manifest-dir data/manifests/v1 \
  --policy configs/cleaning_policy_v1.yaml \
  --output data/processed/v1 \
  --report-dir reports/d2

tongue-data validate-clean \
  --processed-dir data/processed/v1 \
  --policy configs/cleaning_policy_v1.yaml
```

## 测试

```bash
pytest -q
```

## D3-E：原图分割推理 + ROI（不训练）

需要本地提供 frozen D3-C checkpoint（不进 Git）：

```text
runs/segmentation/d3c/baseline/best.pt
```

单图推理：

```bash
tongue-data segmentation-infer \
  --image path/to/image.jpg \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --output runs/segmentation/d3e/inference
```

契约与 Freeze：`docs/D3_E_INFERENCE_CONTRACT.md`、`docs/D3_E_FREEZE_REPORT.md`。

原则：下游表型必须使用 **original RGB + original-resolution mask**，禁止直接用 384×384 normalized tensor。

## D4-A：Input Guard 契约（不训练 QC）

```bash
tongue-data validate-input-guard --policy configs/input_guard_v1.yaml
```

契约与 Freeze：`docs/D4_A_INPUT_GUARD_CONTRACT.md`、`docs/D4_A_FREEZE_REPORT.md`。

## D4-B：信号质量规则（engineering heuristic）

```bash
tongue-data input-guard-calibrate \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --segmentation-dir data/segmentation/v1 \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --policy configs/input_guard_v1.yaml \
  --output runs/input_guard/d4b/calibration

tongue-data input-guard-run \
  --image path/to/image.jpg \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --policy configs/input_guard_v1.yaml
```

Policy **v1.1**；阈值仅用 train+val 校准，**不是**临床标准。  
`evaluation_complete` / `guard_ready` 仍为 false（缺 color_cast / occlusion / stain）。  
Freeze：`docs/D4_B_FREEZE_REPORT.md`。

## D1/D2 边界

D1 不物理删除 raw 重复数据，也不生成最终 train/val/test。

D2 再处理：
- MD5 去重
- 标签冲突
- patient-level split
- leakage-safe split
- Gold/Aux/Pseudo pools
