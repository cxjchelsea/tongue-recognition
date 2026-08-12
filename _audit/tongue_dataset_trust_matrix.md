# 舌诊数据源可信度与用途矩阵

生成依据：

- 事实列：[`matrix_facts.json`](matrix_facts.json) / [`summary.md`](summary.md)（`tongue_dataset_audit.py` @ 2026-08-07，含 Tongueset3）
- 判断列：按下述定级标准人工填定；包内均未检出 LICENSE 文件（TMC 依 Dryad 平台默认 CC0）

## 定级标准

| 维度 | 等级含义 |
|------|----------|
| 来源可信度 | **A**=DOI/机构库+可复核论文；**B**=高校/平台发布但授权或文档不全；**C**=来源弱/仅网盘式整理 |
| 标签可信度 | **A**=专家/多人人工且协议清晰；**B**=人工但无协议细节或仅文件夹/结构化表标签；**C**=含模型预测标签或弱监督；**D**=不可用于主监督 |
| 是否可商用 | 有明示商用许可/**CC0**=是；仅研究/CC 非商用/未找到 LICENSE=**未明示，默认不可商用** |
| 训练角色 | `seg-pretrain` / `multi-label-main` / `detection-main` / `binary-aux` / `domain-hard-neg` / `holdout-eval-only` / `exclude` |

规模列优先写 **unique_md5**（内容去重后）；括号内为 canonical 扫描文件数。

## 矩阵

| 数据集 | 来源可信度 | 标签可信度 | 规模(唯一图,审计后) | 是否患者级ID | 是否人工标签 | 是否可商用 | 训练角色 | 备注风险 |
|--------|------------|------------|---------------------|--------------|--------------|------------|----------|----------|
| BioHit Tongue Image Dataset | B | A（分割） | **300** (300) | 否 | 是（mask） | 未明示，默认不可商用 | `seg-pretrain` | 图-mask 配对率 100%；尺寸统一 768×576；无 LICENSE；受控采集环境 |
| Tongueset3 | B（TongueSAM / arXiv:2308.06444） | A（分割人工） | **999** (1000) | 否 | 是（Labelme mask） | 未明示，默认不可商用 | `seg-pretrain`（野外域） | `img`↔`gt` 配对 100%；统一 400×400；**无病症类标签**；mask 为 RGB 且舌体像素值=1（非 255，预览常全黑）；1 对内容重复（102/1020）；来源 AI Studio 拼图，授权链不清 |
| DSCT裂纹舌数据集 | C | B | **92** (95) | 否 | 是（expert 0/1） | 未明示，默认不可商用 | `binary-aux`（裂纹） | 权威目录仅 `data/data/expert data`；全树冗余约 **5.06×**；体量过小，不宜主训 |
| TMC-Tongue | A（Dryad DOI `10.5061/dryad.1c59zw48r`） | B | **6575** (6719 txt) | 否 | 是（据 Dryad README） | **是（Dryad 默认 CC0）** | `detection-main` | 只用 `shezhenv3-txt`；split 5594/572/553 与 README 一致；**集内 MD5 重复 144** 且存在 train↔val 同图；classes.txt 与 yaml 在 id 18/19 命名漂移；极稀有类（如 14/18 仅 2 实例） |
| TongueDx | B | B | **5263** (6344 origin) | **是**（csv `id`，约 4650） | 倾向是（结构化多标签） | 未明示，默认不可商用 | `multi-label-main` | `test/` 与编号桶 basename 重复；集内 MD5 多余文件 **1081**；须按 **patient id** 划分；Spleen/FurThick 极度不平衡 |
| TonguExpertDatabase | B | **L1=A / L2=C** | **5992** (5992) | Sample ID（`TE…`，非严格患者） | L1 是（1747）；L2 否（预测） | 未明示，默认不可商用 | L1→`multi-label-main` 候选；L2→禁止金标准 | Raw/Mask 配对 100%；L1 大量 `NA`；**禁止把 L2 预测标签当主监督** |
| Tooth-Marked Tongue | B | B | **1250** (1250) | 弱（时间戳文件名） | 是（文件夹） | 未明示，默认不可商用 | `binary-aux`（齿痕） | marked 546 / unmarked 704；无细粒度定位标注 |
| 中医舌诊染苔数据 | B | B（场景标签） | **1935** (2008) | 否 | 是（染苔/非染苔） | 未明示，默认不可商用 | `domain-hard-neg` | 染苔=食物染色，**不得当病苔正样本**；集内 MD5 重复 73（同图多名） |

## 标签分布（审计事实）

主表只写可信度/角色；**各类别计数在这里**（原名保留，旁注中文）。原始 JSON：[`matrix_facts.json`](matrix_facts.json)、[`per_dataset/`](per_dataset/)。

### BioHit（分割，无病症类）

| 项 | 值 |
|----|---:|
| 原图 | 300 |
| mask（分割掩膜） | 300 |
| 配对率 | 100% |
| mask 前景占比 p10 / p50 / p90 | 见 `per_dataset/BioHit.json` |

无分类标签；监督信号为二值**舌体轮廓 mask**（舌头区域 vs 背景）。受控设备采集。

### Tongueset3（分割，无病症类；野外/手机域）

| 项 | 值 |
|----|---:|
| img（原图） | 1000 |
| gt（分割真值） | 1000 |
| 配对率 | 100% |
| 内容唯一图 | 999 |
| 尺寸 | 全部 400×400 |
| mask 像素取值 | 全部为 {0,1}（舌体=1，背景=0） |
| mask 前景占比 p10 / p50 / p90 | 0.1134 / 0.2959 / 0.6101 |
| 前景占比>0.5 的 mask 数 | 220（近景舌头占画面大半，属正常） |

无分类标签；与 BioHit 同类任务，但是**非标准拍摄环境**，适合做分割域泛化。训练时勿把 mask 当 0/255。

### DSCT（裂纹二分类，canonical=`expert data`）

| 标签 | 中文含义 | 图片数 |
|------|----------|------:|
| 0 | 无裂纹 / 非正例 | 53 |
| 1 | 有裂纹 | 42 |
| 合计（文件） | — | 95 |
| 内容唯一 | — | 92 |

### TMC-Tongue（检测实例数，YOLO txt；类名以 yaml 为准）

划分：train **5594** / val **572** / test **553**（图-标配对 6719/6719）。

| id | 原名(yaml) | 中文 | 实例数 |
|---:|------------|------|------:|
| 0 | jiankangshe | 健康舌 | 23 |
| 1 | botaishe | 薄苔舌 | 620 |
| 2 | hongshe | 红舌 | 1545 |
| 3 | zishe | 紫舌 | 252 |
| 4 | pangdashe | 胖大舌 | 703 |
| 5 | shoushe | 瘦舌 | 289 |
| 6 | hongdianshe | 红点舌 | 3725 |
| 7 | liewenshe | 裂纹舌 | 1949 |
| 8 | chihenshe | 齿痕舌 | 1542 |
| 9 | baitaishe | 白苔舌 | 5167 |
| 10 | huangtaishe | 黄苔舌 | 1156 |
| 11 | heitaishe | 黑苔舌 | 125 |
| 12 | huataishe | 滑苔舌 | 291 |
| 13 | shenquao | 肾区凹 | 507 |
| 14 | shenqutu | 肾区凸 | 2 |
| 15 | gandanao | 肝胆凹 | 292 |
| 16 | gandantu | 肝胆凸 | 9 |
| 17 | piweiao | 脾胃凹 | 187 |
| 18 | xinfeitu | 心肺凸 | 2 |
| 19 | xinfeiao | 心肺凹 | 377 |

说明：13–19 为舌面脏腑分区的凹/凸形态标注（数据集自用命名，非通用国标术语）。  
注意：`classes.txt` 多出 `piweitu`（脾胃凸），与 yaml 在 id 18/19 错位（见审计 `name_drift`）。

### TongueDx（多标签，CSV 唯一 image_path 上的 0/1）

患者 id 约 **4650**；下列为标签值计数（同一路径去重后）。`0`=无/阴性，`1`=有/阳性。

| 原名 | 中文 | 0（无） | 1（有） |
|------|------|-------:|-------:|
| TonguePale | 舌淡白 | 4485 | 624 |
| TipSideRed | 舌尖/舌边红 | 2866 | 2243 |
| Spot | 斑点 | 2678 | 2431 |
| Ecchymosis | 瘀斑 | 4638 | 471 |
| Crack | 裂纹 | 835 | 4274 |
| Toothmark | 齿痕 | 2125 | 2984 |
| FurThick | 苔厚 | 136 | 4973 |
| FurYellow | 苔黄 | 4276 | 833 |
| Heart | 心（脏腑相关） | 2716 | 2393 |
| Lung | 肺（脏腑相关） | 2057 | 3052 |
| Spleen | 脾（脏腑相关） | 32 | 5077 |
| Liver | 肝（脏腑相关） | 1946 | 3163 |
| Kidney | 肾（脏腑相关） | 1662 | 3447 |

### TonguExpert L1（人工，1747 行；大量 NA=未标注）

| 字段原名 | 中文 | 分布（中文） |
|----------|------|----------------|
| labels_tai | 苔色 | 未标注 1369 / 淡黄 162 / 白 112 / 黄 104 |
| labels_zhi | 舌质色 | 未标注 1408 / 暗 140 / 正常 108 / 淡 91 |
| labels_fissure | 裂纹 | 未标注 1175 / 轻 306 / 重 266 |
| labels_tooth_mk | 齿痕 | 未标注 1091 / 轻 430 / 重 226 |

取值对照：`NA`=未标注；`light_yellow`=淡黄；`white`=白；`yellow`=黄；`dark`=暗；`light`=淡/轻；`regular`=正常；`severe`=重。  
L1 覆盖 Raw：**29.16%**（1747/5992）。

### TonguExpert L2（模型预测，5992 行；**禁止当金标准**）

| 字段原名 | 中文 | 分布（中文） |
|----------|------|----------------|
| coating_label | 腻苔 | 腻 5342 / 厚腻 532 / 非腻 118 |
| tai_label | 苔色 | 白 3349 / 淡黄 2284 / 黄 359 |
| zhi_label | 舌质色 | 正常 2998 / 暗 1594 / 淡 1400 |
| fissure_label | 裂纹 | 无 4017 / 轻 1230 / 重 745 |
| tooth_mk_label | 齿痕 | 无 3393 / 轻 1903 / 重 696 |

取值对照：`greasy`=腻；`greasy_thick`=厚腻；`non_greasy`=非腻；`None`=无；其余同 L1。

### Tooth-Marked（文件夹二分类）

| 原名 | 中文 | 图片数 |
|------|------|------:|
| marked | 有齿痕 | 546 |
| unmarked | 无齿痕 | 704 |
| 合计 | — | 1250 |

### 中医舌诊染苔数据

| 标签 | 中文含义 | 图片数（文件） |
|------|----------|---------------:|
| 染苔 | 食物等外源性染色苔 | 1001 |
| 非染苔 | 非染色 | 1007 |
| 内容唯一合计 | — | 1935 |

染苔食物来源 Top10：彩虹糖 55、芒果 44、巧克力 42、蓝莓干 34、黑葡提 31、黑莓 30、蓝莓 25、橘子 24、金桔盐金枣 23、黑巧 22。

## 跨集事实

- 跨数据集 MD5 碰撞：**0**（8 集 canonical 扫描范围内无跨集内容重复；Tongueset3 与 BioHit 无同图）
- 全库损坏图：**0**
- 本地 8 集已齐，无需再按“下载顺序”排队；以下仅定**训练纳入顺序**

## 列值与审计字段对照

| 矩阵列 | 审计来源 |
|--------|----------|
| 规模 | `images.unique_md5` / `scanned_images` |
| 是否患者级ID | `labels.patient_id_evidence` + TongueDx `unique_patient_ids` |
| 是否人工标签 | `labels.human_label_evidence`；TonguExpert 拆 L1/L2 |
| 是否可商用 | `license_files_found` + Dryad 平台政策（仅 TMC 升为“是”） |
| 备注风险 | 各集 `labels.*`、intra-dup、name_drift、类别分布 |
