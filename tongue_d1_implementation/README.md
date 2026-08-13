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

## D1/D2 边界

D1 不物理删除 raw 重复数据，也不生成最终 train/val/test。

D2 再处理：
- MD5 去重
- 标签冲突
- patient-level split
- leakage-safe split
- Gold/Aux/Pseudo pools
