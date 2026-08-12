# D1 Tongue Ontology + Unified Manifest 实施方案

## 1. 阶段定位

### 1.1 D1 的核心目标

D1 要解决的不是“如何训练舌诊模型”，而是：

> **把当前 8 个来源不同、标签体系不同、监督形式不同的舌诊数据集，转换成一套统一、可追溯、可验证、可被后续模型稳定消费的数据契约。**

完成 D1 后，后续 D2 数据清洗、D3 分割、D4 Input Guard、D5 多任务舌象模型均只依赖 D1 产出的统一数据接口，不再直接解析各个原始数据集。

---

## 2. D1 输入条件

当前已经完成 8 个数据集的基础审计，包括 BioHit、TongueSet3、DSCT、TMC-Tongue、TongueDx、TonguExpert、Tooth-Marked Tongue 和染苔数据。

这些数据的监督形式并不统一：

| 数据集 | 主要监督类型 |
|---|---|
| BioHit | 舌体像素级 Mask |
| TongueSet3 | 舌体像素级 Mask |
| TMC-Tongue | 多类目标检测标注 |
| TongueDx | 图像级多标签 |
| TonguExpert | L1 人工部分标签 + L2 模型预测标签 + Mask/连续表型 |
| Tooth-Marked Tongue | 齿痕二分类 |
| DSCT | 裂纹专项小样本标签 |
| 染苔数据 | 染苔/非染苔场景分类 |

这一事实决定了 D1 不能简单做“8个CSV拼接”。

当前审计还发现：

- TMC 存在内容重复、跨原始 split 的重复图片以及类名漂移；
- TongueDx 有 patient ID，后续必须患者级划分；
- TonguExpert L1 与 L2 监督来源不同；
- 8 个数据集当前未发现跨数据集 MD5 内容碰撞。

这些问题在 D1 中需要被**记录进数据契约**，但实际去重和重新划分放到 D2 完成。

---

# 3. D1 非目标

D1 明确不做：

```text
模型训练
超参数选择
模型架构实验
train / val / test 最终划分
TMC 物理删除重复图片
TongueDx 数据采样
TonguExpert L2 伪标签训练
类别平衡处理
数据增强
疾病/证候推理
```

D1 只负责：

```text
定义标准语言
↓
解释各数据集标签
↓
建立映射关系
↓
建立统一数据表示
↓
保证来源可以追溯
↓
建立自动校验
```

---

# 4. D1 总体架构

```text
                     Raw Datasets
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
       TMC           TongueDx          TonguExpert
        │                 │                 │
      ...其余5个数据集
                          │
                          ▼
                  Source Adapters
                          │
                          ▼
                 Source Label Layer
                          │
                          ▼
             Tongue Phenotype Ontology v1
                          │
                          ▼
                 Canonical Mapping
                          │
                          ▼
              Unified Manifest Package
               ┌──────────┼──────────┐
               ↓          ↓          ↓
            samples     labels    spatial
                          │
                          ▼
                 Automatic Validation
                          │
                          ▼
                Dataset Contract v1
```

---

# 5. D1-A：建立 Tongue Phenotype Ontology v1

## 5.1 Ontology 的作用

Ontology 是整个舌诊项目的“标准语言”。

以后：

- 数据集；
- 模型；
- API；
- 数据库；
- 前端；
- 四诊融合 Agent；

都不直接使用：

```text
hongshe
TonguePale
labels_zhi
FurYellow
```

这类数据源字段。

统一使用：

```text
tongue_body.color.red
tongue_body.color.pale
coating.color.yellow
features.crack.present
```

---

# 6. Ontology v1 建议结构

建议文件：

```text
ontology/
└── tongue_phenotype_v1.yaml
```

第一版定义：

```yaml
version: "1.0"

tongue_body:

  color:
    task_type: multilabel_or_partial_multiclass

    labels:
      pale:
      normal:
      red:
      dark:
      purple:

  shape:
    task_type: multilabel

    labels:
      enlarged:
      thin:

coating:

  color:
    task_type: multiclass_partial

    labels:
      white:
      light_yellow:
      yellow:
      black:

  properties:
    task_type: multilabel

    labels:
      thick:
      peeling:
      greasy:

features:

  crack:

    present:
      task_type: binary

    severity:
      task_type: ordinal
      labels:
        mild:
        severe:

  tooth_mark:

    present:
      task_type: binary

    severity:
      task_type: ordinal
      labels:
        mild:
        severe:

  red_spot:
    present:
      task_type: binary

  ecchymosis:
    present:
      task_type: binary

quality:

  stain_suspected:
    task_type: binary

segmentation:

  tongue:
    task_type: binary_segmentation
```

---

# 7. Ontology 设计原则

## 7.1 只描述视觉表型

V1 Ontology 中允许：

```text
红
紫
淡
胖大
瘦薄
白苔
黄苔
裂纹
齿痕
红点
瘀斑
```

暂时不允许：

```text
脾虚
湿热
肝火旺
肾虚
某疾病
某脏器异常
```

---

## 7.2 不强行合并语义不同的标签

例如：

```text
TonguExpert.dark
```

不能未经医学定义确认直接映射：

```text
tongue_body.color.purple
```

正确处理：

```yaml
source_label: dark
canonical_label: null
mapping_status: needs_review
```

---

## 7.3 Unknown 不是训练类别

推理阶段允许：

```text
unknown
```

但训练阶段：

```text
unknown ≠ negative
unknown ≠ normal
```

Unknown 是：

> 当前信息不足，模型不应该被迫回答。

---

## 7.4 NA 与 0 必须彻底区分

这是 D1 最重要的数据规则之一。

```text
0
```

表示：

> 数据明确标注该特征不存在。

而：

```text
NA
```

表示：

> 该数据没有告诉我们这个特征存在还是不存在。

例如 TonguExpert L1 大量字段为 NA。

因此：

```text
NA → label_available = false
```

绝不能：

```text
NA → value = 0
```

---

# 8. D1-B：建立 Source Label Mapping

建议：

```text
ontology/
└── mappings/
    ├── tmc_v1.yaml
    ├── tonguedx_v1.yaml
    ├── tonguexpert_v1.yaml
    ├── tooth_marked_v1.yaml
    ├── dsct_v1.yaml
    ├── stained_v1.yaml
    ├── biohit_v1.yaml
    └── tongueset3_v1.yaml
```

---

# 9. Mapping Schema

每个映射至少保存：

```yaml
source_dataset:
source_field:
source_label:

canonical_task:
canonical_label:

mapping_status:
mapping_confidence:

annotation_type:
label_source:

note:
```

其中：

### mapping_status

限定：

```text
exact
compatible
partial
needs_review
excluded
```

含义：

| 状态 | 含义 |
|---|---|
| exact | 原标签与 canonical 含义基本一致 |
| compatible | 可以用于同一任务，但定义存在轻微差异 |
| partial | 只覆盖 canonical 概念的一部分 |
| needs_review | 暂时无法可靠映射 |
| excluded | 明确不进入 V1 |

---

# 10. TMC 初始映射

例如：

```yaml
- source_label: hongshe
  canonical_task: tongue_body.color
  canonical_label: red
  mapping_status: exact

- source_label: zishe
  canonical_task: tongue_body.color
  canonical_label: purple
  mapping_status: exact

- source_label: pangdashe
  canonical_task: tongue_body.shape
  canonical_label: enlarged
  mapping_status: exact

- source_label: shoushe
  canonical_task: tongue_body.shape
  canonical_label: thin
  mapping_status: exact

- source_label: hongdianshe
  canonical_task: features.red_spot.present
  canonical_label: true
  mapping_status: exact

- source_label: liewenshe
  canonical_task: features.crack.present
  canonical_label: true
  mapping_status: exact

- source_label: chihenshe
  canonical_task: features.tooth_mark.present
  canonical_label: true
  mapping_status: exact

- source_label: baitaishe
  canonical_task: coating.color
  canonical_label: white
  mapping_status: exact

- source_label: huangtaishe
  canonical_task: coating.color
  canonical_label: yellow
  mapping_status: exact

- source_label: heitaishe
  canonical_task: coating.color
  canonical_label: black
  mapping_status: exact
```

TMC 当前的五脏凹凸类标签统一：

```yaml
mapping_status: excluded
reason: outside_v1_visual_phenotype_scope
```

TMC 审计中部分相关类别实例只有 2 或 9，而且原始 classes/yaml 还存在命名漂移，因此不应该进入 V1。

---

# 11. TongueDx 初始映射

候选：

```text
TonguePale
→ tongue_body.color.pale

Crack
→ features.crack.present

Toothmark
→ features.tooth_mark.present

Spot
→ features.red_spot.present

Ecchymosis
→ features.ecchymosis.present

FurThick
→ coating.properties.thick

FurYellow
→ coating.color.yellow
```

`TipSideRed` 不直接等价于：

```text
tongue_body.color.red
```

因为“舌尖/舌边红”和“整个舌质红”不是完全相同概念。

推荐：

```yaml
mapping_status: partial
```

必要时 Ontology 后续单独增加：

```text
tongue_body.regional_color.tip_side_red
```

而不是强行合并。

TongueDx 五脏字段：

```text
Heart
Lung
Spleen
Liver
Kidney
```

全部：

```text
excluded
```

不进入 V1。

---

# 12. TonguExpert 初始映射

必须分：

```text
L1
L2
```

### L1

```text
label_source = human
supervision_tier = gold_candidate
```

### L2

```text
label_source = model_prediction
supervision_tier = pseudo
```

L2 永远不能被 D1 转换成：

```text
label_source = human
```

TonguExpert 当前审计显示 L1 只覆盖部分图像，而 L2 覆盖 5992 张，但 L2 是预测结果。

---

# 13. D1-C：统一 Manifest 设计

这里不建议使用“一张超级宽 CSV”。

因为未来任务会继续增加：

```text
舌色
舌形
苔色
裂纹
Mask
Bounding Box
严重程度
质量标签
……
```

更稳定的方法是建立三个标准表。

---

# 14. samples.parquet

负责描述：

> 这张图片是谁。

Schema：

```text
sample_id
dataset
source_sample_id
source_image_path
md5
width
height
patient_id
patient_id_available
source_split
duplicate_group_id
dataset_version
ingest_version
```

示例：

```text
sample_id:
tmc::000001

dataset:
tmc_tongue

source_image_path:
data/raw/tmc/...

md5:
abc123...

patient_id:
null

patient_id_available:
false

source_split:
train
```

注意：

`source_split` 只是记录原数据提供的 split。

D1 不认为：

```text
source_split = final_split
```

---

# 15. labels.parquet

负责：

> 这张图片有什么非空间标签。

Schema：

```text
sample_id

canonical_task
canonical_label
value

label_available

source_dataset
source_field
source_label

annotation_type
label_source
supervision_tier

mapping_status
mapping_version

confidence

note
```

例如：

```text
sample_id:
tonguedx::00128

canonical_task:
features.crack.present

canonical_label:
true

value:
1

label_available:
true

source_label:
Crack

annotation_type:
image_level

label_source:
human_or_dataset_annotation

supervision_tier:
gold_candidate
```

---

# 16. spatial_annotations.parquet

TMC 的 Bounding Box 与 BioHit/TongueSet3 的 Mask 不适合塞到 labels 表。

因此建立：

```text
spatial_annotations.parquet
```

Schema：

```text
sample_id

annotation_id
annotation_task
canonical_label

annotation_type

x_min
y_min
x_max
y_max

mask_path

source_dataset
source_label
label_source

mapping_version
```

支持：

```text
bbox
mask
polygon
```

---

# 17. 为什么采用三表结构

最终关系：

```text
samples
   │
   ├──────── labels
   │
   └──────── spatial_annotations
```

这样未来一个样本可以同时拥有：

```text
1个舌体 Mask
4个红点 box
1个裂纹标签
1个齿痕标签
1个苔色标签
```

而不用不断修改一个巨大的 CSV Schema。

---

# 18. D1-D：Source Adapter

每个数据集独立实现解析器：

```text
src/data/adapters/
├── base.py
├── tmc.py
├── tonguedx.py
├── tonguexpert.py
├── biohit.py
├── tongueset3.py
├── tooth_marked.py
├── dsct.py
└── stained.py
```

统一接口：

```python
class DatasetAdapter:

    def scan_samples(self):
        ...

    def parse_labels(self):
        ...

    def parse_spatial_annotations(self):
        ...

    def validate_source(self):
        ...
```

这样以后新增第9个数据集时：

> 只增加 Adapter + Mapping。

而不是修改整个训练管线。

---

# 19. Source Adapter 不能修改 raw

Adapter：

```text
读取 raw
↓
解析
↓
生成 manifest
```

禁止：

```text
移动图片
修改图片
删除图片
重命名 raw
覆盖原标签
```

Raw 数据始终只读。

---

# 20. D1-E：Provenance 设计

每一个 canonical label 必须能够回答：

> 这个标签最初从哪里来的？

例如模型最终使用：

```text
features.crack.present = true
```

必须能够向后追溯：

```text
canonical label
↓
mapping version
↓
source label
↓
source dataset
↓
source annotation file
↓
source image
```

推荐字段：

```text
source_dataset
source_sample_id
source_field
source_label
annotation_type
label_source
supervision_tier
mapping_version
```

---

# 21. supervision_tier

建议标准化：

```text
gold_candidate
silver
pseudo
weak
excluded
```

### gold_candidate

例如：

- 专家人工；
- 明确人工 Mask；
- 来源较可靠的人工标签。

注意这里叫：

> candidate

而不是直接叫 gold。

是否真正进入最终 Gold Test Set，要在 D2/D9 再决定。

### pseudo

例如：

```text
TonguExpert L2
```

### weak

例如来源和协议较弱的文件夹分类。

---

# 22. D1-F：自动校验

D1 必须配套自动测试，不能靠肉眼检查 YAML 和 CSV。

建议：

```text
tests/data_contract/
├── test_ontology.py
├── test_mapping.py
├── test_manifest.py
├── test_provenance.py
├── test_na_semantics.py
└── test_source_integrity.py
```

---

# 23. Ontology 自动检查

检查：

```text
canonical task 是否存在
canonical label 是否存在
task_type 是否合法
重复 label
非法路径
ontology version
```

---

# 24. Mapping 自动检查

要求：

> 每一个发现的 source label 都必须有明确处理结果。

可以是：

```text
exact
compatible
partial
needs_review
excluded
```

但不能：

```text
无人处理
```

因此测试：

```text
source_labels
-
mapped_source_labels
=
0
```

---

# 25. NA 语义测试

这是必须单独写的测试。

例如：

```text
TonguExpert L1:
NA
```

生成结果必须：

```text
label_available = false
```

测试中明确禁止：

```text
NA → 0
```

---

# 26. L1 / L2 隔离测试

自动断言：

```text
TonguExpert L2

label_source != human
supervision_tier == pseudo
```

如果将来任何代码把 L2 写成 gold：

> 测试必须直接失败。

---

# 27. 空间标注检查

针对：

```text
bbox
mask
```

验证：

```text
0 <= x_min < x_max <= width
0 <= y_min < y_max <= height

mask尺寸合法
mask路径存在
sample_id存在
```

---

# 28. Manifest 完整性检查

检查：

```text
sample_id 唯一
source_path 存在
md5 非空
width / height 合法
dataset 合法
mapping_version 非空
所有 label 都能找到 sample
所有 spatial annotation 都能找到 sample
```

---

# 29. 审计数量回归检查

D1 Manifest 构建完成以后，应重新生成统计结果。

例如：

```text
TMC canonical 图片数
TongueDx patient 数
TonguExpert L1 行数
TongueSet3 mask 配对数
```

必须能够与当前 audit 的事实一致，或者能解释差异原因。

不能发生：

```text
原始有 5000
manifest 只剩 4300
但没有任何日志解释
```

---

# 30. D1-G：统一 Manifest Builder

最终提供命令：

```bash
python -m tongue_data.build_manifest \
    --config configs/datasets_v1.yaml \
    --ontology ontology/tongue_phenotype_v1.yaml \
    --output data/manifests/v1
```

输出：

```text
data/manifests/v1/
├── samples.parquet
├── labels.parquet
├── spatial_annotations.parquet
├── dataset_statistics.json
├── mapping_statistics.json
└── build_metadata.json
```

---

# 31. build_metadata.json

记录：

```json
{
  "manifest_version": "1.0",
  "ontology_version": "1.0",
  "mapping_version": "1.0",
  "datasets": {},
  "build_timestamp": "...",
  "code_commit": "...",
  "source_hashes": {}
}
```

这样未来：

> “这个模型到底用了哪版数据？”

可以明确回答。

---

# 32. D1 项目目录

建议：

```text
tongue-diagnosis/
│
├── data/
│   ├── raw/
│   ├── audit/
│   └── manifests/
│       └── v1/
│
├── ontology/
│   ├── tongue_phenotype_v1.yaml
│   └── mappings/
│       ├── tmc_v1.yaml
│       ├── tonguedx_v1.yaml
│       ├── tonguexpert_v1.yaml
│       ├── biohit_v1.yaml
│       ├── tongueset3_v1.yaml
│       ├── tooth_marked_v1.yaml
│       ├── dsct_v1.yaml
│       └── stained_v1.yaml
│
├── src/
│   └── data/
│       ├── schema.py
│       ├── manifest.py
│       ├── validators.py
│       └── adapters/
│
├── configs/
│   └── datasets_v1.yaml
│
├── tests/
│   └── data_contract/
│
└── docs/
    ├── ontology_v1.md
    ├── mapping_report_v1.md
    └── manifest_spec_v1.md
```

---

# 33. D1 子阶段拆分

## D1-A：Ontology 定义

完成：

```text
tongue_phenotype_v1.yaml
ontology_v1.md
```

验收：

- V1 所有任务有唯一 canonical path；
- 不包含证候/疾病；
- 支持 binary / multilabel / multiclass / ordinal / segmentation；
- 所有标签均有定义。

---

## D1-B：Source Mapping

完成：

```text
8个 mapping YAML
```

验收：

- 所有 source label 都有处理状态；
- 无静默丢弃；
- 有歧义的标记 needs_review；
- V1 排除标签明确 excluded。

---

## D1-C：Manifest Schema

完成：

```text
samples schema
labels schema
spatial annotations schema
```

验收：

- 支持缺失标签；
- 支持 provenance；
- 支持 bbox/mask；
- 支持 pseudo label；
- 支持 patient ID。

---

## D1-D：8个 Adapter

完成：

```text
raw dataset
→ unified intermediate representation
```

验收：

所有数据集均可以：

```text
读取
解析
生成记录
```

而不修改原始文件。

---

## D1-E：Manifest Builder

完成：

```text
8 datasets
→ samples.parquet
→ labels.parquet
→ spatial_annotations.parquet
```

---

## D1-F：Automatic Validation

完成：

```text
ontology validation
mapping validation
NA semantics
L1/L2 isolation
spatial validation
manifest referential integrity
audit count regression
```

---

## D1-G：D1 Freeze

最终生成：

```text
Tongue Dataset Contract v1.0
```

并冻结：

```text
Ontology v1.0
Mapping v1.0
Manifest Schema v1.0
```

后续任何变化通过：

```text
v1.1
v1.2
v2.0
```

管理，而不是直接改旧版本。

---

# 34. D1 验收标准

只有以下条件全部满足才能进入 D2。

| 验收项 | 要求 |
|---|---|
| Ontology | V1全部任务已定义 |
| Source Labels | 100% 有 mapping status |
| 原始数据 | 0 修改 |
| Sample ID | 全局唯一 |
| MD5 | 全部可追溯 |
| Patient ID | 有则保留，无则明确 null |
| NA | 0 个被错误转换成阴性 |
| TonguExpert L2 | 0 个被标记成 human/gold |
| Spatial Annotation | 坐标/Mask引用合法 |
| Provenance | 每个标签可追溯到 source |
| Manifest | 可重复构建 |
| Audit Regression | 与现有审计事实一致或有明确解释 |
| 自动测试 | 全部通过 |
| Train/Val/Test | D1 不生成最终 split |

---

# 35. D1 明确禁止的实现捷径

以下做法如果出现，应视为 D1 不合格：

```text
把所有图片复制到一个 images 文件夹
```

然后：

```text
把所有标签强行拼成一个 CSV
```

或者：

```text
没有标签 → 0
```

或者：

```text
TonguExpert L2 → 普通监督标签
```

或者：

```text
dark → purple
```

这种未经审核的语义强行映射。

也不能：

```text
为了标签统一
直接改 raw dataset
```

---

# 36. D1 完成以后会得到什么

完成 D1 后，项目会第一次拥有：

```text
统一的数据语言
+
统一的数据结构
+
统一的标签来源记录
+
统一的自动校验
```

以后模型代码不需要知道：

```text
这张图来自 TMC
还是 TongueDx
还是 TonguExpert
```

模型训练只看到：

```text
sample
+
canonical labels
+
availability mask
+
supervision tier
```

例如：

```text
sample_id = DX::000123

tongue_body.color.pale:
    value = 1
    available = true

features.crack.present:
    value = 1
    available = true

features.tooth_mark.present:
    value = 0
    available = true

coating.color.white:
    available = false
```

这才是真正能够进入多任务模型的数据。

---

# 37. D1 与 D2 的边界

D1 回答：

> **这些数据是什么意思？**

D2 回答：

> **哪些数据最终可以进入训练、验证和测试？**

因此 D1 只记录：

```text
duplicate_group
patient_id
source_split
label_source
mapping
```

D2 再执行：

```text
去重
冲突解决
patient-level grouping
重新 split
类别统计
训练集构建
Gold Test Set 构建
```

两者不能混在一起。

---

# 38. D1 完成后的下一阶段

D1：

```text
Ontology
+
Mapping
+
Manifest
```

完成后进入：

> **D2 Dataset Cleaning + Leakage-Safe Split**

D2 的重点将是：

```text
TMC MD5去重
TMC跨split泄漏消除
TMC类别名漂移处理

TongueDx patient-level split

TonguExpert L1/L2隔离

其他数据集重复和标签冲突处理

Gold / Auxiliary / Pseudo 数据分层

最终生成：
train_v1
val_v1
test_v1
```

然后 D3 才第一次真正开始训练：

> **Tongue Segmentation Baseline**

---

# 39. D1 最终交付物

```text
D1 Tongue Dataset Contract v1.0

├── Tongue Phenotype Ontology v1
├── 8 × Source Mapping
├── samples.parquet
├── labels.parquet
├── spatial_annotations.parquet
├── dataset_statistics.json
├── mapping_statistics.json
├── build_metadata.json
├── Source Adapters
├── Manifest Builder
├── Validators
├── Automated Tests
├── ontology_v1.md
├── mapping_report_v1.md
└── manifest_spec_v1.md
```

D1 的最终成功标准不是：

> “数据能读取。”

而是：

> **任意一条进入未来模型训练的监督信号，都可以明确知道它是什么意思、从哪里来、属于什么可信等级、是否真的有标签，并且整个转换过程可以重复执行和自动验证。**

做到这一点之后，再进入 D2 和模型训练，后面的舌诊系统才有可靠的数据基础。