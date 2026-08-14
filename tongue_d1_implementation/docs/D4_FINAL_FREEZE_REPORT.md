# D4 Final Freeze Report

**D4_FINAL_STATUS = `PARTIAL_PASS_WITH_KNOWN_LIMITATION`**

## Capability table

| Check | Status |
|---|---|
| tongue_presence | ACTIVE |
| tongue_scale | ACTIVE |
| tongue_completeness | ACTIVE |
| segmentation_integrity | ACTIVE |
| focus | ACTIVE |
| exposure | ACTIVE |
| illumination_uniformity | ACTIVE |
| resolution | ACTIVE |
| color_cast | ACTIVE |
| occlusion | ACTIVE |
| stain_suspected | **DEFERRED** |

## Provenance story

1. D4-C v1: in-domain TARGET_PASS; cross-domain FAIL
2. D4-D.1: D4C_CROSS_DOMAIN_CONCERN (stain-triggered retake surge)
3. D4-C.1-A: COLOR_ACQUISITION_STYLE shortcut
4. D4-C.1-B: style aug + consistency NEEDS_IMPROVEMENT_STOP
5. D4-C.1-C: representation invariance FAILED
6. D4-C.1-D: SOURCE_CONFOUNDING_SEVERE; EXISTING_DATA_RESCUABLE=false
7. D4-E: stain deferred; production Input Guard partial freeze

## Production unified (n=130, stain disabled)

- pass/warning/retake = 70/47/13
- RETAKE=13 与 D4-B baseline 一致；全部可追溯到 active checks（无 stain attribution）
- stain invocations = 0
- stain-triggered warning/retake = 0/0
- evaluation_complete true/false = 120/10
- evaluation_complete=false 原因：`color_cast_unavailable` ×10（deferred stain 不计入）
- guard_ready = true
- full_capability_coverage = false
- audit artifact：`reports/d4/d4e_production_unified_audit.json`（本地；`reports/d4/` gitignore）

## biohit::278.bmp

- decision = pass
- reasons = []
- evaluation_complete = true
- stain_evaluation_state = not_evaluated / finding = null
- 不因 D4-E 调参；D3 known failure 语义保持

## Versions

- Input Guard Contract = 1.1
- Input Guard Policy = 1.4
- D3 checkpoint `config_hash` = a26934531e6643f6（unchanged）
- D4-B / color_cast / occlusion thresholds unchanged
- stain research v1 thresholds 0.95/0.96 preserved

## Gate results

- validator = PASS
- pytest = PASS（301 passed, 1 skipped；含 d4e 40）

## Known limitation

- `STAIN_DETECTION_DEFERRED` / `SOURCE_DATASET_CONFOUNDING_SEVERE`
- D5 coating-color must not claim external staining is excluded.
- Capture guidance 仅为 acquisition fallback，不是算法排除。

## STOP

Do not auto-enter D5 without confirmation.  
Commit plan：`docs/D4_COMMIT_PLAN.md`（未经确认勿 push）。
