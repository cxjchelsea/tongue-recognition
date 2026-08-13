# D4-C.1-A Stain Cross-Domain Shortcut Diagnosis

> Diagnosis Report（**不是** Final Freeze）。  
> BioHit / TongueSet3 **没有** stain gold label，不得表述为 false positive。

## Verdict

- **recommendation**: `PROCEED_D4C1B_DOMAIN_ROBUST_RETRAINING`
- **ready_for_d4c1b**: `true`
- **primary_shortcut_hypothesis**: `COLOR_ACQUISITION_STYLE`
- **preprocessing_equivalence**: `PASS`
- **runtime / thresholds / checkpoint**: 未修改

## 1. 核心问题回答

### 模型是否学到了 dataset identity？

**是（强证据）。**

手工统计特征 LogisticRegression 5-fold CV accuracy ≈ **0.958**  
→ `dataset_identity_signal = strong`

### TongueSet3 为何崩、BioHit 为何不崩？

BioHit vs TongueSet3 最大分布差异（effect size 排序）集中在：

1. `p_stain`（结果本身）
2. `mean_b` / `luminance_mean` / `mean_l` / `mean_g` / `mean_r`
3. `roi_short_side`
4. `bg_ratio` / `mean_b_lab` / `mean_s`

即：**颜色 / 亮度 / 采集风格 + 分辨率**，而不是单纯 black-fill。

### Black-fill 是否强化 dataset identity？

**对 TongueSet3 高分：不是主因。**

Representation ablation（median `p_stain`）：

| group | black | gray | bbox |
|---|---:|---:|---:|
| Stained negative | 0.00045 | 0.895 | 0.00034 |
| BioHit | 0.0036 | 0.999 | 0.232 |
| TongueSet3 | **0.998** | **1.000** | **0.976** |

TongueSet3：black / gray / bbox **均极高** → 更像 **tongue appearance / acquisition color shortcut**。  
（gray/mean_fill 对 BioHit/negatives 的飙升属于 OOD fill，不能当正式准确率，只说明模型对 fill 色敏感；不能据此说 TongueSet3 靠黑边。）

### ResNet18 关注哪里？

TongueSet3 high-score Grad-CAM 均值：

- inside tongue ≈ **0.665**
- boundary ≈ **0.026**
- background/fill ≈ **0.223**
- padding ≈ **0.086**

主要在舌面，其次有一定 fill 能量；**不是** padding/letterbox 主导。

### Embedding

Centroid L2：

- TongueSet3 ↔ stain_pos ≈ **16.8**
- TongueSet3 ↔ stain_neg ≈ **35.0**
- BioHit ↔ stain_neg ≈ **12.0**

→ TongueSet3 embedding **更靠近 stain positive**；BioHit 更靠近 stain negative。

## 2. Shortcut Evidence Matrix

| factor | status |
|---|---|
| color_distribution | **strong** |
| white_balance | **strong** |
| luminance | moderate |
| mask_fill_geometry | weak |
| ROI_geometry | weak |
| letterbox_padding | weak |
| resolution | moderate |
| blur | weak |
| compression | undetermined |
| dataset_identity | **strong** |
| local_stain_evidence | moderate |

## 3. 其他审计摘要

- n_manifest = **3234**（stained 1935 + biohit 300 + tongueset3 999）
- black_pixel_ratio median：stain_pos≈0.375 / stain_neg≈0.911 / BioHit≈0.416 / TongueSet3≈0.405  
  （BioHit 与 TongueSet3 接近；stain_neg 因小舌/黑边更高）
- padding_ratio median：BioHit≈0.125 / TongueSet3≈0.138（接近）
- p_stain strongest Spearman：`mean_s`, `mean_a`, `roi_short_side(-)`, `black_pixel_ratio(-)`, `mean_b_lab`
- uncertain band：仍几乎未利用（known limitation；本阶段不重校准）
- D4-D.1 audit 文件 hash：未修改

## 4. 进入 D4-C.1-B 的条件

**具备。**

理由：

1. 无 preprocessing bug  
2. 无伪标 external negative  
3. 三角证据一致：distribution + counterfactual + embedding/CAM  
4. 主因指向跨域采集颜色/风格，需要 domain-robust retraining / representation redesign  

**禁止在本阶段**：改 threshold、重训、改 runtime、Final Freeze、进 phenotype。

等待确认后再进入 **D4-C.1-B Domain-Robust Retraining**。
