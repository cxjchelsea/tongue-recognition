# Manifest Spec v1

## samples.parquet

一行对应一张 canonical image。保存 sample_id、dataset、source path、MD5、尺寸、patient ID、原始 split 等。

## labels.parquet

只保存“确实有监督信息”的 label row。

因此：
- `NA` 不生成 negative；
- 明确的 0 才能生成 negative；
- 每条记录保留 source_dataset/source_field/source_label；
- supervision tier 不能丢失。

## spatial_annotations.parquet

保存 bbox/mask。

TMC 的 bbox 可衍生一条 positive image-level evidence，但绝不从“未出现 bbox”自动推断 negative。

## Provenance

任何 canonical label 均可追溯至 source label 和 mapping version。
