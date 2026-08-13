# D3-E Inference Contract — Original-Image Segmentation + ROI

## 目的

D3-A/B/C 验证 **384×384 letterbox 分割模型**有效。  
D3-E 定义如何对 **任意尺寸原始用户图片** 做推理，并恢复为下游可用的 **原图级** 结果。

本阶段 **不训练**、不修改 D3-C frozen weights、不进入 D4 Input Guard / phenotype。

## 核心原则（必须）

后续舌色 / 苔色 / 颜色统计 / 局部表型 **必须**基于：

```text
original RGB image  +  original-resolution predicted mask
→ original RGB tongue ROI
```

**禁止**把下列对象当作 phenotype 输入：

- 384×384 model input tensor
- ImageNet-normalized tensor
- color-jitter / train augmentation 图像

Model-space mask（384×384）仅用于 debug / visualization / traceability。

## 输入

支持：

- `path`
- `PIL.Image`
- `numpy.ndarray`（H×W×3 RGB，或灰度 / RGBA 显式转换）

内部统一 **RGB uint8**。

### EXIF Orientation

真实手机图可能带 EXIF orientation。推理必须：

1. load
2. `PIL.ImageOps.exif_transpose()`
3. convert RGB（记录 `original_mode`）
4. 读取 final `original_width/height`
5. letterbox

未 transpose 会导致竖拍坐标与 mask 错位。

## Letterbox（复用 D3-A）

**不得**另写 `cv2.resize(..., (384,384))`。

共享实现：`segmentation/geometry.py`

- `compute_letterbox_metadata`
- `letterbox_image`
- `letterbox_mask`

Rounding contract（与训练一致）：

```text
scale = min(input_w / orig_w, input_h / orig_h)
resized_w = round(orig_w * scale)  # >= 1
resized_h = round(orig_h * scale)
pad_left = total_pad_w // 2
pad_right = total_pad_w - pad_left
pad_top = total_pad_h // 2
pad_bottom = total_pad_h - pad_top
```

Metadata 至少包含：

```json
{
  "original_size": [W, H],
  "input_size": [384, 384],
  "scale": ...,
  "resized_size": [rw, rh],
  "padding": {"left": ..., "right": ..., "top": ..., "bottom": ...}
}
```

## 模型加载

`load_frozen_segmentation_model(checkpoint, data_config, train_config, device)`

校验：

- architecture / encoder / classes
- `config_hash` 与 train config 一致
- `model_state_dict` 存在
- `strict=True`（禁止 `strict=False` 静默跳过）

默认 checkpoint（本地，不进 Git）：

```text
runs/segmentation/d3c/baseline/best.pt
```

路径由 CLI / 调用方传入，**不硬编码**在 `model.py`。

## Inference 模式

- `model.eval()`
- `torch.inference_mode()`
- CUDA 可用 AMP autocast；**probability / geometry 转 float32**

## Threshold / Restoration（v1 固定）

读取 frozen train config：`metrics.mask_threshold = 0.5`  
**禁止**另设 0.45 / 0.6 或 threshold search。

### Restoration strategy（策略 A）

```text
logits
→ sigmoid → model probability (384×384, float32)
→ remove letterbox padding
→ bilinear restore probability → original size
→ threshold 0.5
→ original binary mask {0,1}
```

文档固定：

> threshold applied in **original-resolution probability space**.

Binary nearest unletterbox 仅作调试对比（策略 B），phenotype 主路径用策略 A。

## Restored mask contract

| 字段 | 约定 |
|---|---|
| shape | `[H_original, W_original]` |
| dtype | `uint8`（内部亦可 bool） |
| values | `{0,1}` |
| 0 | background |
| 1 | tongue |

写 PNG 时：`1 → 255`；读回：`>0 → 1`。

## BBox

从 `original_binary_mask` 计算：

- `bbox_tight`：紧包围盒
- `bbox_roi`：外扩后的 ROI 框

**约定：`xyxy` exclusive** → `[x1, y1, x2, y2)`  
等价于 Python：`image[y1:y2, x1:x2]`

### ROI margin

默认：`roi_margin_ratio = 0.05`

```text
margin_x = bbox_width * ratio
margin_y = bbox_height * ratio
向外扩展后 clip 到图像边界
```

原因：裂纹 / 齿痕 / 舌边可能贴边；过紧 crop 会丢 context。

## ROI 输出

| 输出 | 来源 |
|---|---|
| `tongue_roi_rgb` | exif-transposed **original RGB** 按 `bbox_roi` 裁剪 |
| `tongue_roi_mask` | original binary mask 同区域裁剪 |
| `masked_tongue_rgb` | 可选；非舌像素置 0 |

禁止对 ROI 做 ImageNet normalize / brightness / white-balance。

## Connected components

默认：`keep_largest_component = true`

- 4-连通
- 仅保留最大前景组件
- 报告 `component_count_before`、`largest_component_ratio`

**不做** erosion/dilation/CRF/GrabCut。

## 失败 / 警告状态

| 情况 | 行为 |
|---|---|
| empty mask | `status = "no_tongue_detected"`；bbox/ROI = null；不 crash |
| `foreground_ratio > 0.95` | warning: `near_full_prediction`（不在本阶段 fail） |

完整用户重拍 / 合格判定 → **D4 Input Guard**（本阶段不做）。

## Result schema

`TongueSegmentationResult` 至少包含：

- status / sample_id / original_mode
- sizes / threshold / restoration_strategy / bbox_convention
- mask foreground stats / component stats
- bbox_tight / bbox_roi / roi_size
- original_binary_mask / optional probability & model-space masks
- tongue_roi_rgb / tongue_roi_mask / optional masked ROI
- letterbox_metadata / model_metadata / warnings
- D4 预留：foreground_ratio、bbox ratios、touches_image_border、mean/max probability

JSON metadata **不**内嵌 base64 大图；图像单独存文件。

## CLI

```bash
tongue-data segmentation-infer \
  --image path/to/image.jpg \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --output runs/segmentation/d3e/inference
```

工程回归（**仅验证实现**，禁止据此调参）：

```bash
tongue-data segmentation-infer-regression \
  --checkpoint runs/segmentation/d3c/baseline/best.pt \
  --segmentation-dir data/segmentation/v1 \
  --data-config configs/segmentation_v1.yaml \
  --train-config configs/segmentation_train_v1.yaml \
  --output runs/segmentation/d3e/regression \
  --allow-test
```

## 与 D3-C / D4 边界

- D3-E test 访问用途 = **engineering regression only**
- 禁止根据 regression 调 threshold / 换模型 / 重训练
- 不在本阶段判断：舌头过小、模糊、过曝、染苔、伸舌不全等（D4）
