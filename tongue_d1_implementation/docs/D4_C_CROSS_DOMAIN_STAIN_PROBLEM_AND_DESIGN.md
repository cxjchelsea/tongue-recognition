# D4 染苔跨域捷径：问题复盘与解决方案设计

> 文档性质：项目复盘用设计说明书（不是 Freeze Report，也不是 Final Freeze）。  
> 整理日期：2026-08-14  
> 当前阶段：`D4-C.1-C`（representation-level domain invariance）进行中，**未切换线上 detector**。  
> 线上仍生效：D4-C v1 染苔模型 + Input Guard Policy **1.3**（`t_clear=0.95` / `t_retake=0.96`）。

本文把「为什么 Input Guard 不能终裁、为什么不能进 D5 舌象训练」写成一份可独立阅读的材料。  
数字与结论均来自已冻结/已落盘的报告，不另做新实验。

---

## 1. 一句话结论

D4 要回答的不是「这张舌头是什么舌象」，而是：

> 这张照片是否足够可靠，可以继续进入舌象视觉表型分析？

11 项质控里，**染苔检测**（`quality.stain_suspected`）在源域 `stained_coating` 上几乎完美，但一放到真实采集域 **TongueSet3** 就出现灾难性高分饱和：大量干净手机舌图被判成「疑似外源染色」并触发 `RETAKE`。

根因不是阈值拍脑袋，也不是预处理 bug，而是模型学到了 **采集风格 / 颜色捷径（COLOR_ACQUISITION_STYLE）**，把「像不像 stained 数据集」当成了「有没有染色」。

因此：

- D4 **不能 Final Freeze**
- **不能进入 D5 多任务舌象模型**
- 当前只允许继续做更强的 domain-robust 方案，且 **禁止给外部域打伪标签**

---

## 2. 这个问题在整条 V1 链路里的位置

V1 正式推理链路：

```text
用户原图
  → D3 舌头分割（U-Net + ResNet34，frozen）
  → 原图 RGB + 原分辨率 mask / ROI
  → D4 Input Guard（11 项质控）
        ├─ RETAKE  → 停止，返回重拍建议
        ├─ WARNING → 可继续，但必须保留 reason
        └─ PASS    → 才允许进入 D5 表型模型
  → D5 多任务舌象（舌色 / 苔色 / 裂纹 / 齿痕 / …）
```

D4 的 11 项检查：

| # | check | 实现方式 | 阶段 | 当前可信度 |
|---|---|---|---|---|
| 1 | 舌头是否存在 | 规则 | D4-B | 可用 |
| 2 | 舌头尺度 | 规则 | D4-B | 可用 |
| 3 | 舌头是否完整 | 规则 | D4-B | 可用 |
| 4 | 分割是否可信 | 规则 | D4-B | 可用 |
| 5 | 清晰度 | 规则 | D4-B | 可用 |
| 6 | 曝光 | 规则 | D4-B | 可用 |
| 7 | 光照均匀性 | 规则 | D4-B | 可用 |
| 8 | 分辨率 | 规则 | D4-B | 可用 |
| 9 | **染苔嫌疑** | **学习模型** | **D4-C** | **跨域未过关** |
| 10 | 色偏 | 规则（舌外中性参考） | D4-D | 工程可用 |
| 11 | 遮挡 | 规则（多证据） | D4-D | 工程可用 |

D4-D 把 11 项拼成统一 Guard 后，工程门禁曾标为 `TARGET_PASS`，但随后的 **D4-D.1 联调审计** 证明：  
统一 Guard 的高 RETAKE 几乎全部来自第 9 项 stain，而不是色偏或遮挡。

这就是当前阶段还停在 D4、没有训舌象模型的直接原因。

---

## 3. 染苔检测本来要做什么

### 3.1 唯一合法问题

`quality.stain_suspected` 只回答：

> 这张舌头照片是否可能受到食物、饮料、药物、色素等**外源性染色**影响，从而使后续苔色分析不可靠？

三态输出：

| finding | 含义 | 对 Guard 的影响 |
|---|---|---|
| `false` | 无明显外源染色嫌疑 | 不阻挡 |
| `uncertain` | 证据不足，进入安全缓冲 | `WARNING` + `STAIN_SUSPECTED` |
| `true` | 外源染色嫌疑成立 | `RETAKE` + `STAIN_SUSPECTED` |

阈值合同（只允许用 source val 校准）：

```text
p <= t_clear              → false
t_clear < p < t_retake    → uncertain
p >= t_retake             → true
```

D4-C v1 冻结阈值：`t_clear = 0.95`，`t_retake = 0.96`。

### 3.2 明确不回答

- 苔是什么颜色（`coating.color`：白 / 淡黄 / 黄 / 黑）
- 舌色、裂纹、齿痕等任何表型
- 证候或疾病
- 「这张图来自哪个数据集」

禁止出现的 reason：`YELLOW_COATING_STAIN`、`BLACK_TONGUE`，以及任何把染色等同黄苔/黑苔的映射。

### 3.3 监督数据

| 项 | 约定 |
|---|---|
| 唯一有 gold 的源 | D2 冻结后的 `stained_coating`（1935 张；train/val/test = 1548/194/193） |
| 角色 | `quality-only auxiliary supervision` |
| 划分 | **100% 继承 D2 split**，禁止重分 |
| BioHit / TongueSet3 | **没有 stain gold**，只能当 unlabeled 外部域 |
| 禁止 | 从苔色标签反推染色；给外部域打伪标签；用 test 选模或调阈值 |

这一条是后续所有方案的硬约束：外部域没有「干净/染色」真值，因此 **TongueSet3 高分不能写成 false positive**，只能写成「跨域概率偏移 / 饱和」。

---

## 4. 问题是怎么被发现的

### 4.1 D4-C 源域看起来已经成功

D4-C v1（ResNet18，black-masked ROI 224，只在 `stained_coating` 上监督）：

| 集合 | AUROC | 备注 |
|---|---:|---|
| source val | 0.9988 | 选模唯一依据 |
| source test（只评一次） | 0.9918 | 未参与训练/阈值 |

源域内几乎可分。当时政策升到 1.2，`stain_suspected` 接入 Guard，但 `color_cast` / `occlusion` 尚未实现，故 `guard_ready=false`。

### 4.2 D4-D 拼上统一 Guard 后，外部测试 RETAKE 暴涨

D4-D 补齐色偏、遮挡后，在 **BioHit + TongueSet3 test 共 130 张** 上做了一次只读审计（未调参）：

| 决策 | 数量 |
|---|---:|
| pass | 36 |
| warning | 14 |
| retake | **80** |

单独看 RETAKE 高，还不能判死刑。D4-D.1 用同一 130 张、复用各 check 冻结输出，做了四级消融：

| 组合 | pass | warning | retake |
|---|---:|---:|---:|
| A 仅 D4-B 信号规则 | 74 | 43 | 13 |
| B A + stain | 39 | 11 | **80** |
| C B + color_cast | 36 | 14 | 80 |
| D 全量 Guard | 36 | 14 | 80 |

增量：

- B − A（接上 stain）= **+67 RETAKE**
- C − B（色偏）= 0
- D − C（遮挡）= 0

新增 67 张全部是 `stain_only`。聚合器没有算错，stain 三态映射也没有 bug。

### 4.3 两个外部域表现完全相反

同一套 v1 阈值 `p >= 0.96`：

| 域 | n | `p>=0.96` 比例 | 中位 `p_stain` | 相对 stained 负样本中位偏移 |
|---|---:|---:|---:|---:|
| stained 负样本（源域） | 94 | 极低 | 0.00059 | 基准 |
| BioHit test | 30 | **3.3%** | 0.0033 | 小 |
| TongueSet3 test | 100 | **78%** | **0.998** | 接近 1.0 |

TongueSet3 的分数分布几乎贴在 1 上；BioHit 则更接近源域负样本。  
这不是「所有外部图都判染」，而是 **某一个采集域被系统性当成染色正样本**。

D4-D.1 结论：

- `recommendation = D4C_CROSS_DOMAIN_CONCERN`
- `guard_ready` 建议改为 `SET_FALSE_PENDING_DOMAIN_FIX`
- 本阶段 **不改 runtime / 阈值 / checkpoint / split**

---

## 5. 根因诊断（D4-C.1-A）

诊断报告明确：BioHit / TongueSet3 无 gold，不得写成「误报率」。  
结论：`primary_shortcut_hypothesis = COLOR_ACQUISITION_STYLE`。

### 5.1 模型学到了数据集身份

用手工统计特征做 LogisticRegression，5-fold CV 区分数据集的准确率约 **0.958**。  
说明仅靠颜色、亮度、尺度等统计量，就能把「这张图来自哪个库」分得很开。模型有充分动机走这条捷径。

### 5.2 为什么 TongueSet3 崩、BioHit 不崩

BioHit vs TongueSet3 差异主要在：

1. 模型自己的 `p_stain`（结果本身已经分家）
2. `mean_b` / 亮度 / `mean_l` / `mean_g` / `mean_r`
3. ROI 短边（分辨率）
4. 背景占比、Lab / 饱和度

即：**颜色 / 白平衡 / 亮度 / 采集风格 + 分辨率**，不是单纯黑边填充。

### 5.3 黑边填充不是 TongueSet3 高分的主因

同一批图换成 black / gray / bbox 三种 representation，TongueSet3 中位 `p_stain` 分别为 **0.998 / 1.000 / 0.976**，三种都极高。  
若主因是 letterbox 黑边，换成 bbox 应明显下降；没有。  
gray fill 会让 BioHit / 源域负样本分数飙升，只说明模型对填充色敏感，不能反过来说 TongueSet3 靠黑边。

### 5.4 注意力在舌面，不在 padding

TongueSet3 高分图 Grad-CAM 均值：

| 区域 | 能量 |
|---|---:|
| 舌面内部 | **0.665** |
| mask 外填充 | 0.223 |
| padding | 0.086 |
| 边界 | 0.026 |

模型看的是舌面外观，不是 letterbox 条带。

### 5.5 Embedding 几何与捷径矩阵

质心距离：

- TongueSet3 ↔ stain 正样本 ≈ **16.8**
- TongueSet3 ↔ stain 负样本 ≈ **35.0**
- BioHit ↔ stain 负样本 ≈ **12.0**

TongueSet3 的表征更靠近「染色正样本」；BioHit 更靠近「未染色」。

| 因素 | 强度 |
|---|---|
| 颜色分布 | 强 |
| 白平衡 | 强 |
| 数据集身份 | 强 |
| 亮度 | 中 |
| 分辨率 | 中 |
| 局部染色证据 | 中 |
| mask 几何 / letterbox / 模糊 | 弱 |

三角证据（分布差 + 反事实 representation + embedding/CAM）一致，排除预处理实现错误，也排除「给外部域当负样本训过」这种数据泄漏。

### 5.6 源域内部也有采集混淆（C.1-C 审计补充）

`stained_coating` 正负样本本身就不是同分布采集：

| 统计 | 正样本中位 | 负样本中位 |
|---|---:|---:|
| 高度 | 950.5 | 1512.5 |
| 宽度 | 900 | 2599 |
| 亮度 | 112.8 | 17.3 |
| 红色通道 | 143.8 | 22.4 |

源域「染色 vs 未染色」与「亮/红/小图 vs 暗/大图」高度纠缠。  
模型在源域把这两件事一起学会，迁到 TongueSet3（另一套亮度和白平衡）时，就会把「像源域正样本的采集风格」当成染色。

---

## 6. 这个问题为什么必须先修

### 6.1 对产品语义的破坏

Input Guard 的 `RETAKE` 含义是：

> 显著影响表型判断，不应继续正式 inference，应请用户重拍。

若 78% 的 TongueSet3 风格手机图被 stain 挡下：

- 真实用户场景（自然光、手机、非实验室）会大规模无法进入表型
- 后续舌色/苔色模型永远看不到这类图，指标会虚高且不可泛化
- 用户会把「采集风格不同」理解成「你的舌头有问题」

### 6.2 对科学纪律的破坏

若此时强行进 D5：

- 训练集会被 Guard 按捷径过滤，等于按数据集身份抽样
- 苔色任务会与「像不像 stained 库」缠在一起
- 以后无法区分「表型模型差」还是「入口就把域切歪了」

### 6.3 对 D4 终裁的破坏

D4-D 工程门禁可以过，但 D4-D.1 明确：高 RETAKE 本身不是自动 FAIL，**归因 + 跨域概率偏移**才构成 FAIL。  
在 stain 跨域未解决前，宣称 `guard_ready=true` 并 Final Freeze，属于虚假完工。

---

## 7. 设计原则与硬禁令

后续所有方案共用同一组原则。复盘时若有人提议「先调阈值 / 先上大模型 / 先给 TongueSet3 打伪标」，应直接对照本节否决。

### 7.1 必须遵守

1. **v1 资产只读。** `runs/input_guard/d4c/stain/best.pt` 与 `0.95/0.96` 不得覆盖。新实验写到 `d4c1b/`、`d4c1c/`。
2. **不从 v1/v2 继续 fine-tune。** 避免把捷径权重当初始化。
3. **外部域无 stain gold。** BioHit / TongueSet3 不得写成负样本，不得伪标，不得熵最小化 / self-training。
4. **选模只用 source val AUROC。** 外部域指标只做鲁棒性门禁，不做 checkpoint 选择。
5. **test 是一次性资产。** 候选未过 acceptance 前，禁止碰 source TEST 与 known external 130。
6. **不换更大模型碰运气。** 问题在捷径与表示，不在容量。
7. **不自动切线上。** 即使某候选过门禁，也必须人工确认后才改 active policy。
8. **不进 D5 / D4-E UI。** stain 未过关则整条 D4 未终裁。

### 7.2 明确不采用的「捷径修复」

| 诱惑方案 | 为什么不行 |
|---|---|
| 把 TongueSet3 阈值单独调高 | 等于承认模型在认数据集；换一个手机品牌仍会崩 |
| 把 TongueSet3 全部当负样本再训 | 无 gold；其中真实染色会被写成阴性，污染语义 |
| 用 VLM 零样本当 stain 法官 | 无法校准、不可复现、会把黄苔/滤镜/灯光说成染色 |
| 关掉 stain，先训表型 | 苔色会被食物染色污染，且 Guard 契约不完整 |
| 只看源域 AUROC 宣布成功 | 正是 D4-C 已经犯过的错 |

---

## 8. 总体解决方案：分阶段收紧表示，而不是换任务

目标函数始终不变：源域仍要能分开「外源染色 vs 无明显染色」；同时表示里应去掉「数据集身份 / 采集风格」。

分三层，由浅到深：

```text
D4-C v1
  只在 stained_coating 上做源域监督
  → 源域成功，跨域崩

D4-C.1-A
  只诊断，不改模型
  → 锁定 COLOR_ACQUISITION_STYLE

D4-C.1-B（输入层 + 一致性）
  源域监督
  + 源域 style consistency
  + 外部无标 consistency
  + 受控采集风格增广
  → 饱和略降，间隙几乎不动，FAIL

D4-C.1-C（表示层）
  继承 B 的 consistency / style
  + MixStyle（打乱风格统计）
  + 域对抗 GRL（抑制数据集可分性）
  + 三域均衡采样
  → 进行中；C1/C2 已训，均未过 dual-gate
```

判断「修好了」必须 **双门禁同时过**，不能只看其中一个：

| 门 | 要证明的事 | 典型指标 |
|---|---|---|
| Source gate | 还认识染色 | source val AUROC / PR-AUC ≥ 0.95 |
| Domain gate | 不再认数据集 | 域 logit 间隙下降、TongueSet3 高分率下降、embedding 域探针准确率下降 |

过了 acceptance 才允许动 test；过了 test 且人工确认，才允许切换 active detector。

---

## 9. D4-C.1-B 设计（已做完，未过关）

### 9.1 假设

若捷径主要是采集风格，则在训练时：

- 对源域图做有限的通道增益 / gamma / 曝光 / 对比度扰动，迫使模型在风格变化下仍给出同一 stain 标签；
- 对无标外部图做弱-强一致性，迫使同一张 TongueSet3 / BioHit 在风格扰动下分数稳定。

这样不必给外部域发明标签。

### 9.2 训练设计

| 项 | 选择 |
|---|---|
| 骨干 | ResNet18，ImageNet 初始化，**不**从 v1 fine-tune |
| 输入 | 与 v1 相同：black-masked ROI + letterbox 224 + ImageNet normalize |
| 源域损失 | BCEWithLogits（stained gold） |
| 源域 consistency | 权重 0.5，warmup 5 epoch |
| 外部 consistency | 权重 0.5；弱视图 stop-gradient 当 target |
| 伪标签 / 熵最小 | **关闭** |
| 风格增广范围（仅 train） | RGB gain [0.90, 1.35]；gamma [0.80, 1.25]；exposure [0.80, 1.25]；contrast [0.85, 1.20] |
| 禁止增广 | 灰度、反色、solarize、极端色相、强饱和度、ColorJitter |
| 外部采样 | BioHit : TongueSet3 = 1 : 1 |
| 选模 | 仅 source val AUROC |
| seed | 20260813 |

外部 consistency 的含义：模型对同一张无标图的两个视图应给出相近 logit，**不**把该图当成 0 或 1。

### 9.3 过关标准（预注册）

相对 v1：

- TongueSet3 高分率（`p >= t_retake`）降到 **≤ 0.50**
- BioHit 与 TongueSet3 的 median-logit 间隙缩小 **≥ 50%**
- 源域 test 相对 v1 的 AUROC 下降 **≤ 0.03**（仅当外部 VAL 先过，才允许评 test）

### 9.4 实际结果

训练 8 epoch 早停。源域 val AUROC 0.9978，判别力仍在。

| 指标 | v1 | v2 | 目标 |
|---|---:|---:|---|
| TongueSet3 中位 p | 0.9985 | 0.9895 | 明显离开 1 |
| TongueSet3 高分率 | 0.84 | **0.74** | ≤ 0.50 |
| TongueSet3 中位 logit | 6.50 | 4.54 | 大幅下降 |
| BioHit 中位 p | 0.0069 | 0.0022 | 保持低 |
| 域 median-logit 间隙 | 11.52 | 10.65 | 缩小 ≥50% |
| 间隙降幅 | — | **7.5%** | ≥ 50% |
| 灾难性饱和是否解除 | — | **否** | 是 |

解释：consistency + 风格增广让分数略温和，但模型仍能用颜色/风格把 TongueSet3 推到高分。  
**输入层扰动不够，捷径在更深的表示里。**

按协议：外部 VAL 未过 → source TEST、known 130、统一 Guard 复测全部 SKIP。  
v2 合同存在，**未切换 active detector**。

---

## 10. D4-C.1-C 设计（当前阶段）

### 10.1 假设升级

B 证明：只扰动像素统计、只要求分数一致，不足以拆掉数据集身份。  
C 改为直接改 **中间表示**：

1. **MixStyle**：在特征图上混合不同域的通道均值/方差，使分类头更难依赖风格统计。
2. **GRL 域对抗**：加一个三分类域头（stained / biohit / tongueset3）。特征经梯度反转后再被域头识别，训练目标是「域头变差、stain 头仍好」。
3. **三域均衡采样**：每个 batch 三个域等量，避免 TongueSet3 数量主导域头。

推理时 **不需要** 域标签。域头只存在于训练。

### 10.2 仍然继承的部分

- 同一 ROI 合同、同一 v1 只读资产
- 同一 style augmentation 范围（不重新搜索）
- 同一 source / external consistency，且伪标仍禁止
- 选模仍只看 source val AUROC
- 新 run 目录：`runs/input_guard/d4c1c/`

### 10.3 预注册的四个候选（禁止结果出来后再加搜参）

| 候选 | MixStyle | GRL | consistency | 目的 |
|---|---|---|---|---|
| C0 | — | — | — | 冻结的 B/v2 对照，不重训 |
| C1 | 开 | 关 | 开 | 只打乱风格统计 |
| C2 | 关 | 开 | 开 | 只对抗数据集身份 |
| C3 | 开 | 开 | 开 | 两者叠加；**仅当 C1 或 C2 先出现有效信号才跑** |

C3 的「有效信号」预注册为：间隙降幅 ≥ 0.15 **或** TongueSet3 高分率 < 0.65。  
避免一上来就组合两个未验证模块，无法归因。

### 10.4 MixStyle 合同

| 项 | 值 | 用意 |
|---|---|---|
| 插入层 | `layer1` | 只动浅层风格，少伤语义 |
| p | 0.5 | 一半 batch 混合 |
| alpha | 0.1 | 保守，避免把染色颜色本身洗掉 |
| 策略 | `cross_domain` | 跨域混，不在同类里混 |
| label mixup | **禁止** | 只混特征统计，不混 stain 标签 |
| 推理 | 关闭 | 部署无随机性 |

### 10.5 GRL 合同

| 项 | 值 | 用意 |
|---|---|---|
| 域类别 | 3（stained / biohit / tongueset3） | 域身份是元数据，不是 stain 监督 |
| hidden | 256 | 足够但不大 |
| `lambda_max` | 0.3（moderate） | 对抗太强会毁源域染色可分性 |
| warmup | 5 epoch 线性升到 lambda_max | 先让 stain 头站稳 |
| 域损失权重 | 1.0 | 与 stain BCE 同量级，靠 lambda 调度压强度 |
| 推理 | 丢掉域头 | 不依赖域标签 |

### 10.6 损失（概念）

```text
L =
    L_stain_source                         # 唯一有监督的 stain BCE
  + 0.5 * L_consistency_source             # 源域风格扰动一致性
  + 0.5 * L_consistency_external           # 外部无标一致性（无伪标）
  + λ(t) * L_domain_adversarial            # 仅 C2/C3；λ 从 0 升到 0.3
```

其中 `L_domain_adversarial` 的梯度在特征上反转：域头想分对域，encoder 想让域头分不对。

### 10.7 Dual-gate（预注册，写进 train config，禁止事后改）

**Source（必须）**

- val AUROC ≥ 0.95，PR-AUC ≥ 0.95
- 目标档：AUROC ≥ 0.97

**Domain robustness（必须，相对 v2）**

| 指标 | 最低过关 | 目标档 |
|---|---|---|
| 域 median-logit 间隙降幅 | ≥ 40% | ≥ 50% |
| TongueSet3 高分率 | < 0.50 | < 0.30 |
| 域探针准确率相对 v2 下降 | ≥ 0.10 | ≥ 0.20 |
| 风格敏感度下降 | ≥ 0.30（观察） | — |

v2 对照冻结值：

- 域间隙 ≈ **10.65**
- TongueSet3 高分率 = **0.74**
- v2 域探针准确率 ≈ **0.878**

**候选排序（仅 PASS 者参与）**

1. 间隙降幅更大
2. TongueSet3 高分率更低
3. 域探针下降更多
4. 风格敏感度更低
5. source val AUROC 更高

**Test 门（仅 acceptance 通过后评一次）**

- source test AUROC / PR-AUC ≥ 0.95
- 相对 v1 的 AUROC 下降 ≤ 0.03
- 自信染色精确率 / 召回 / 干净纯度 ≥ 0.90

任一候选 FAIL，则该候选不得碰 test。

### 10.8 当前实验事实（截至本文整理时）

C1、C2 已训练并审计；C3 尚未跑。两者 **source 都很好，dual-gate 都 FAIL**。线上仍是 v1。

| 指标 | v2 对照 | C1 MixStyle | C2 GRL | 最低过关 |
|---|---:|---:|---:|---|
| source val AUROC | 0.9978 | **0.9994** | **0.9985** | ≥ 0.95 |
| 域 logit 间隙 | 10.65 | 11.11（变差 4%） | **2.43（降 77%）** | 降 ≥ 40% |
| TongueSet3 高分率 | 0.74 | **0.45** | **0.47** | < 0.50 |
| TongueSet3 中位 p | 0.989 | 0.945 | 0.945 | 离开饱和 |
| BioHit 中位 p | 0.002 | 0.0003 | **0.604** | 保持低 |
| 域探针准确率 | 0.878 | 0.808（只降 0.07） | 0.800（只降 0.08） | 至少再降 0.10 |
| 风格敏感度降幅 | — | 0.43 | 0.52 | 观察项 |
| stain 正负质心距 | 36.4（v1）/ 81.9（v2 probe） | 33.3 | 75.5 | 不能塌掉 |
| acceptance | — | FAIL | FAIL | — |
| 是否已评 test | 否 | 否 | 否 | — |

读法：

- **C1** 把 TongueSet3 高分率压到 0.45（过了高分门），但域间隙没缩小，BioHit 与 TongueSet3 仍被风格分开。MixStyle 减了「极端饱和」，没拆掉域身份。
- **C2** 把域间隙砍掉 77%（过了间隙门），高分率也到 0.47；但 BioHit 中位 p 被抬到 0.60，说明对抗在「拉近两个外部域」时，把 BioHit 从「很像未染色」推向中间带。域探针仍约 0.80，数据集身份还在。
- 两者都有 `meaningful_signal`，按合同 **允许启动 C3**，但 C3 不是自动成功保证。
- 源域内部正负样本的采集混淆（亮度/分辨率）仍在，C3 也消不掉这块标注偏差，只能尽量不让它主导跨域决策。

---

## 11. 系统落地方案（修好之后怎么接回去）

本节是「若某候选过关」后的接入设计，不是当前已发生的事实。

### 11.1 运行时数据流

```text
原图
  → D3-E：原分辨率 mask + bbox ROI
  → stain 输入：mask 外填 0，letterbox 224，ImageNet normalize
  → stain 模型：单个 logit → sigmoid → p
  → 三态映射（t_clear / t_retake，只允许用 source val 重校准）
  → Input Guard 聚合：PASS < WARNING < RETAKE
  → 若 RETAKE 且 primary_reason=STAIN_SUSPECTED
        返回重拍建议（清洁口腔 / 避免刚进食染色食物 / 重新拍摄）
  → 若 PASS/WARNING 且 evaluation_complete
        才把 original RGB + original mask 交给 D5
```

硬规则：下游表型 **禁止** 直接吃 384×384 归一化 tensor。

### 11.2 切换 active detector 的清单

必须全部满足，缺一不可：

1. 候选 dual-gate = `MINIMUM_PASS` 或 `TARGET_PASS`
2. source TEST 只评一次，且不低于第 10.7 节 test 门
3. known external 130 复测：TongueSet3 不再灾难性饱和；BioHit 不出现新的系统性高分
4. 统一 Guard 复测：stain 不再单独贡献数十张「采集域误伤」
5. v1 目录未被覆盖，新旧阈值文件可并列回滚
6. **人工确认** 后才改 `input_guard` policy 的 stain checkpoint 指针
7. 写入新的 Freeze Report；否则不得宣称 D4 Final Freeze

### 11.3 即使过关，stain 仍保持 quality-only

过关后的模型仍然：

- 不输出苔色
- 不参与 D5 损失
- 不把 `STAIN_SUSPECTED` 解释成病理
- 阈值仍是工程启发式，不是临床标准

### 11.4 若 C3 仍失败：预留的下一层，而不是现改

合同允许讨论、但 **本文不授权开工** 的后续：

- 更强的表示约束（仍禁止伪标）
- 源域内采集混淆的分层抽样 / 重加权（先审计，再改训练）
- 把 stain 从「单点 RETAKE」改成「与苔色联合的不确定度」（那是 D5/D7 的事，不能用来掩盖跨域失败）

明确继续禁止：换更大 backbone、接 VLM、按数据集设不同阈值、给 TongueSet3 打伪标。

---

## 12. 与相邻模块的边界

| 模块 | 边界 |
|---|---|
| D3 分割 | stain 只用其 ROI/mask；不得为了 stain 去改 D3 checkpoint |
| D4-B 信号规则 | 模糊/曝光/尺度等阈值冻结；不靠它们「补」stain 的跨域失败 |
| D4-D 色偏 | 只看舌 **外** 中性参考，禁止舌面均值 RGB（那会变成舌色捷径） |
| D4-D 遮挡 | 不得把裂纹/齿痕当遮挡；与 stain 无关 |
| D5 表型 | 消费者，不是修复手段；Guard 未就绪则不准开工 |
| 大模型 | 可写说明文字，不可当 stain 或舌象的主判官 |

---

## 13. 关键产物索引（复盘时按这个找）

### 问题发现与诊断

| 文件 | 内容 |
|---|---|
| `docs/D4_D_FREEZE_REPORT.md` | 统一 Guard 工程通过 |
| `docs/D4_D_1_INTEGRATION_AUDIT.md` | RETAKE 67 张全是 stain；跨域偏移 |
| `docs/D4_C_1_A_SHORTCUT_DIAGNOSIS.md` | 捷径定性为采集风格 |
| `reports/d4c1/d4c1a_*.json` | 诊断用分布、CAM、embedding 原始数 |

### v1 / v2

| 文件 | 内容 |
|---|---|
| `docs/D4_C_STAIN_CONTRACT.md` | v1 语义与三态阈值 |
| `docs/D4_C_FREEZE_REPORT.md` | 源域 TARGET_PASS |
| `docs/D4_C_1_B_DOMAIN_ROBUST_STAIN_CONTRACT.md` | v2 合同摘要 |
| `docs/D4_C_1_B_FREEZE_REPORT.md` | v2 FAIL，间隙只降 7.5% |
| `configs/stain_train_v2.yaml` | v2 训练超参 |
| `runs/input_guard/d4c/stain/` | **线上仍在用的 v1** |
| `runs/input_guard/d4c1b/stain_v2/` | v2 实验，未激活 |

### v3（当前）

| 文件 | 内容 |
|---|---|
| `configs/stain_detection_v3.yaml` | 表示层合同 |
| `configs/stain_train_v3.yaml` | C1/C2/C3 与 dual-gate |
| `src/tongue_data/stain/domain_invariant_model.py` | MixStyle + GRL |
| `src/tongue_data/stain/domain_balanced.py` | 三域均衡采样 |
| `src/tongue_data/stain/v3_train.py` / `v3_audit.py` | 训练与门禁 |
| `reports/d4c1c/candidate_c1.json` | C1 FAIL |
| `reports/d4c1c/candidate_c2.json` | C2 FAIL |
| `reports/d4c1c/source_confounding_audit.json` | 源域正负采集纠缠 |
| `runs/input_guard/d4c1c/c1_mixstyle/` | C1 权重 |
| `runs/input_guard/d4c1c/c2_grl/` | C2 权重 |

---

## 14. 复盘时建议记住的决策

1. **先归因，再改模型。** D4-D.1 用消融证明是 stain 而不是色偏/遮挡/聚合 bug，才开 C.1-A；C.1-A 用三角证据锁定期径，才开重训。
2. **源域好看不等于能上线。** v1 AUROC 0.99 仍然制造了 67 张跨域误伤。
3. **没有 gold 的域不能当负样本。** 这是整条修复线最贵的纪律，也是唯一避免把真实染色写成阴性的办法。
4. **test 是一次性的。** v2 / C1 / C2 失败时主动 SKIP test，是为了以后真过关时还有一次干净评估。
5. **B 失败的价值是排除「只做 consistency 就够」。** C 才上表示层，而不是加参数。
6. **C2 的间隙下降伴有 BioHit 分数上移。** 以后若 C3 过关，必须同时看「两个外部域是否被一起抬进染色带」，不能只看间隙一个数。
7. **当前不宣称 Guard 已就绪。** runtime 里 `guard_ready` 曾因 D4-D 工程门禁被置 true，审计建议应视为 `false`，直到域问题关闭。

---

## 15. 当前状态与下一步（截至 2026-08-14）

| 项 | 状态 |
|---|---|
| 问题是否定义清楚 | 是 |
| 根因是否锁定 | 是（采集风格捷径 + 源域采集混淆） |
| 线上 stain | 仍为 D4-C v1 |
| v2 | 已训，FAIL，未激活 |
| C1 / C2 | 已训，FAIL，未激活，未碰 test |
| C3 | 合同允许启动，尚未作为完成态交付 |
| D4 Final Freeze | **未做** |
| D5 舌象训练 | **未开始，且当前不应开始** |

下一步只应是：在现有 C.1-C 合同内做完 C3（或等价的预注册表示方案），用 dual-gate 判定，失败则停并写实验报告，成功才走第 11.2 节切换清单。  
不要平行开启 D5，也不要为了赶进度把 TongueSet3 阈值单独抬高。
