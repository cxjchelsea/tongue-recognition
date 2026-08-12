# Manifest Spec v1 / v1.1

Contract v1.1 在列结构不变的前提下，明确支持 **explicit negative supervision**。

## samples.parquet

一行对应一张 canonical image。保存 sample_id、dataset、source path、MD5、尺寸、patient ID、原始 split 等。

## labels.parquet

只保存“确实有监督信息”的 label row。`value ∈ {0, 1}`，且 `label_available = true`。

因此：
- `NA` 不生成监督记录（unavailable）；
- 明确的源值 `0` 才能生成 explicit negative；
- partial multiclass / attribute（如 `pale` / `yellow`）：
  - `canonical_label=pale, value=1` → pale positive
  - `canonical_label=pale, value=0` → pale explicit negative（**不是** normal）
  - `canonical_label=yellow, value=0` → yellow explicit negative（**不是** white）
- binary task（配置了 `negative_label`）可继续使用：
  - `canonical_label=true/false, value=1`
- 每条记录保留 source_dataset/source_field/source_label；
- supervision tier 不能丢失。

## spatial_annotations.parquet

保存 bbox/mask。

TMC 的 bbox 可衍生一条 positive image-level evidence，但绝不从“未出现 bbox”自动推断 negative。

## Provenance

任何 canonical label 均可追溯至 source label 和 mapping version。
