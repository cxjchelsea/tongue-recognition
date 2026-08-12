# 舌诊数据集：训练纳入 / 排除清单与推荐顺序

依据：[`tongue_dataset_trust_matrix.md`](tongue_dataset_trust_matrix.md) + 审计事实层（2026-08-07，含 Tongueset3）。

默认场景：**研究试点 / 产品预研**。除 TMC-Tongue（Dryad CC0）外，其余集均为「未明示可商用」——上线商用模型前须单独完成授权核验或替换为自有标注数据。

## 纳入 / 排除

| 数据集 | 决策 | 角色 | 条件 |
|--------|------|------|------|
| BioHit | **纳入** | `seg-pretrain` | 仅用 `dataset`↔`groundtruth/mask`；300 对全量可训分割（受控环境） |
| Tongueset3 | **纳入** | `seg-pretrain`（野外域） | `img`↔`gt` 1000 对；去掉 102/1020 内容重复 1 份；mask 按 `>0` 二值化（像素值=1）；与 BioHit 联合训分割 |
| TongueDx | **纳入** | `multi-label-main` | 按 csv `id` 做 train/val/test；去重 basename（忽略 `origin/test` 与桶内重复拷贝）；勿把 seg 当第二份原图混进分类 |
| TMC-Tongue | **纳入** | `detection-main` | **只使用 `shezhenv3-txt`**；训练前按 MD5 去掉跨 split 重复；统一以 yaml 或 classes.txt 之一为类名真源（修 18/19 漂移）；长尾类需过采样或合并评估 |
| Tooth-Marked | **纳入** | `binary-aux` | 齿痕二分类辅助头 / 微调；不可替代检测框监督 |
| DSCT | **有限纳入** | `binary-aux` / 小样本评估 | 只用 `data/data/expert data/{0,1}`（92 唯一图）；不作主训骨干 |
| 中医舌诊染苔数据 | **纳入（特殊）** | `domain-hard-neg` | 仅作染苔鉴别 / 域偏移负样本；**禁止**当作黄苔/厚苔病理正例 |
| TonguExpert L1 | **有限纳入** | `multi-label-main` 候选 | 仅 1747 条人工行；大量 NA 列需 mask loss；SID 按样本划分 |
| TonguExpert L2 | **排除出主监督** | — | 预测标签可作伪标签实验，**不得**进主损失或最终评测金标准 |

## 推荐训练顺序

1. **BioHit + Tongueset3 分割** — 受控环境 + 野外/手机域，建立舌体 ROI  
2. **TongueDx 多标签** — 有患者级 id，主任务最干净  
3. **TMC 检测（txt only）** — 20 类病理/体征检测；先清 MD5 跨 split 泄漏  
4. **Tooth-Marked + DSCT** — 齿痕 / 裂纹辅助头或课程微调  
5. **染苔 hard-neg** — 鲁棒性与误报抑制  
6. **TonguExpert L1 only** — 扩展苔色/舌质/裂纹/齿痕词表；L2 不用  

```text
BioHit+Tongueset3(seg) → TongueDx(id-split multilabel) → TMC(txt det, dedup)
    → ToothMark + DSCT(aux) → Stain(hard-neg) → TonguExpert(L1 only)
```

## 开训前必做清洗（来自审计）

1. TMC：删除或合并 144 个多余 MD5 副本，确保 train/val/test 无同内容交叉  
2. TongueDx：以 patient `id` 为单位划分；同一 basename 只保留一份 origin  
3. DSCT：忽略 `code/`、`index data/`、`__MACOSX` 与 HOG 镜像目录  
4. 染苔：同 MD5 多名文件只留一条；标签语义固定为「食物染色」  
5. Tongueset3：去掉 `img/102.jpg` 与 `img/1020.jpg` 之一；mask 用 `>0` 而非 `==255`  
6. 全库：当前跨集 MD5 碰撞为 0（含 Tongueset3）；后续增补需重跑 audit 

## 重跑审计

```bash
python "d:\project\4.0\智能体检\scripts\tongue_dataset_audit.py" ^
  --base "d:\project\4.0\智能体检\舌象更新" ^
  --out "d:\project\4.0\智能体检\舌象更新\_audit"
```
