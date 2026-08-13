# D4-D.1 Unified Guard Integration Audit

> 本报告用于决定是否允许 D4 Final Freeze；**不是** Freeze Report。

- stage: `D4-D.1`
- n_samples: `130`
- dataset_counts: `{'tongueset3': 100, 'biohit': 30}`
- recommendation: **`D4C_CROSS_DOMAIN_CONCERN`**
- guard_ready_recommendation: `SET_FALSE_PENDING_DOMAIN_FIX`

## 1. 为什么 RETAKE 从 13 增加到 80？

四级 ablation（同一 130 sample set，复用 per-check frozen outputs）：

- A D4-B only: `{'pass': 74, 'warning': 43, 'retake': 13}`
- B +stain: `{'pass': 39, 'warning': 11, 'retake': 80}`
- C +color_cast: `{'pass': 36, 'warning': 14, 'retake': 80}`
- D full: `{'pass': 36, 'warning': 14, 'retake': 80}`

- Δ retake B−A (stain): `67`
- Δ retake C−B (color_cast): `0`
- Δ retake D−C (occlusion): `0`

## 2. 新增 RETAKE attribution

- newly_rejected_n: `67`
- stain_only: `67`
- color_cast_only: `0`
- occlusion_only: `0`
- multiple: `0`
- stain_trigger (any): `67`
- color_cast_trigger (any): `0`
- occlusion_trigger (any): `0`

## 3–4. Stain finding counts

- overall: `{'clear': 49, 'uncertain': 2, 'stain': 79, 'missing': 0}`
- BioHit: `{'clear': 29, 'uncertain': 0, 'stain': 1, 'missing': 0}`
- TongueSet3: `{'clear': 20, 'uncertain': 2, 'stain': 78, 'missing': 0}`

## 5. Cross-domain probability shift

- possible_cross_domain_probability_shift: `True`
- dataset_identity_shift_suspected: `True`
- overall p>=0.96 rate: `0.6076923076923076`
- BioHit p>=0.96 rate: `0.03333333333333333`
- TongueSet3 p>=0.96 rate: `0.78`
- uncertain count: `2`
- domain compare: `{'available': True, 'd4c_negative_quantiles': {'count': 94, 'min': 1.4921465452122362e-10, 'p01': 9.185507442305286e-09, 'p05': 3.139511598249102e-07, 'p10': 1.3100696151013843e-06, 'p25': 0.00011448202894825954, 'median': 0.0005870272871106863, 'p75': 0.0006665117107331753, 'p90': 0.0010549935977905993, 'p95': 0.0018805701867677254, 'p99': 0.007100727234501043, 'max': 0.054588377475738525, 'mean': 0.001129189678469172, 'std': 0.005578518764015532}, 'biohit_vs_d4c_neg': {'median_shift': 0.0027190346736460924, 'mean_shift': 0.1324012185996785, 'ks': 0.5347517730496454, 'wasserstein': 0.1324012185996785}, 'tongueset3_vs_d4c_neg': {'median_shift': 0.9973649273160845, 'mean_shift': 0.91509284415515, 'ks': 0.9893617021276596, 'wasserstein': 0.9150928441551499}, 'd4c_negative_median': 0.0005870272871106863, 'biohit_median': 0.0033060619607567787, 'tongueset3_median': 0.9979519546031952}`

## 6–7. Color cast / Occlusion

- color_cast: findings=`{'acceptable': 115, 'null': 10, 'suspected': 5}` retake=`0` warning=`5` unavailable=`10`
- occlusion: findings=`{'none': 127, 'possible_occlusion': 3}` retake=`0` warning=`3` unavailable=`0`

## 8–9. Integrity

- aggregation_bugs: `0`
- stain_mapping_bugs: `0`

## 10. evaluation_complete=false

`{'count': 10, 'reason_counts': {'color_cast_unavailable': 10, 'occlusion_unavailable': 0, 'stain_unavailable': 0, 'other': 0}, 'details': [{'sample_id': 'tongueset3::1128.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1285.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::137.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1419.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1531.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1561.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1721.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1758.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1808.jpg', 'flags': ['color_cast_unavailable']}, {'sample_id': 'tongueset3::1822.jpg', 'flags': ['color_cast_unavailable']}]}`

## 11. biohit::278

`{'sample_id': 'biohit::278.bmp', 'D4B_decision': 'pass', 'stain_probability': np.float64(0.006855750922113657), 'stain_finding': 'false', 'color_cast_finding': 'acceptable', 'occlusion_finding': 'none', 'unified_decision': 'pass', 'primary_reason': None, 'all_reason_codes': ''}`

## 12. guard_ready recommendation

- system guard_ready remains true in runtime; audit recommendation = `SET_FALSE_PENDING_DOMAIN_FIX`

## Full retake attribution (80)

- by_source: `{'d4b_only': 1, 'stain_only': 67, 'color_cast_only': 0, 'occlusion_only': 0, 'd4b_plus_stain': 12, 'd4b_plus_cast': 0, 'd4b_plus_occlusion': 0, 'stain_plus_cast': 0, 'stain_plus_occlusion': 0, 'cast_plus_occlusion': 0, 'three_or_more': 0, 'none': 0, 'other': 0}`
- unique_trigger_count: `{'D4B': 13, 'stain': 79, 'color_cast': 0, 'occlusion': 0}`

## Reason / primary reason

- reason_code_counts: `{'STAIN_SUSPECTED': 81, 'TONGUE_SLIGHTLY_SMALL': 17, 'IMAGE_BLUR': 12, 'TONGUE_TOUCHES_FRAME': 11, 'UNDEREXPOSED': 11, 'UNEVEN_LIGHTING': 10, 'IMAGE_RESOLUTION_TOO_LOW': 10, 'OVEREXPOSED': 8, 'COLOR_CAST_SUSPECTED': 5, 'TONGUE_OCCLUDED': 3, 'STRONG_SHADOW': 3, 'TONGUE_RESOLUTION_TOO_LOW': 3, 'SHADOW_CLIPPING': 2, 'TONGUE_TOO_SMALL': 2, 'HIGHLIGHT_CLIPPING': 2, 'TONGUE_CROPPED': 1, 'TONGUE_BLUR': 1}`
- primary_reason_counts: `{'STAIN_SUSPECTED': 67, 'null': 36, 'UNDEREXPOSED': 5, 'TONGUE_TOUCHES_FRAME': 3, 'TONGUE_SLIGHTLY_SMALL': 3, 'STRONG_SHADOW': 3, 'COLOR_CAST_SUSPECTED': 3, 'SHADOW_CLIPPING': 2, 'HIGHLIGHT_CLIPPING': 2, 'OVEREXPOSED': 2, 'TONGUE_TOO_SMALL': 1, 'TONGUE_RESOLUTION_TOO_LOW': 1, 'TONGUE_CROPPED': 1, 'TONGUE_BLUR': 1}`

## Decision note

High RETAKE rate alone is **not** an automatic FAIL. The recommendation is based on trigger attribution and cross-domain probability shift evidence.

本阶段未修改 runtime / thresholds / checkpoints / splits。
