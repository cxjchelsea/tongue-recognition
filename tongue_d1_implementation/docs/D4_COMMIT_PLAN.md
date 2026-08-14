# D4 Commit Plan（建议，未经确认勿 push）

当前 `HEAD = 9c9e885`（已含 D4-C/D4-D + D4-C.1-A）。工作区另有大量未提交变更，建议按逻辑拆分；**若无法安全重放历史 diff，不要强行拆 commit，先整包或按现状分批。**

## 建议提交顺序

1. **D4-C stain baseline**（若尚未完整在 9c9e885）— v1 research checkpoint metadata / thresholds 0.95/0.96 相关文档与测试收口  
2. **D4-D unified guard** — color_cast / occlusion / unified runtime（多数已在 9c9e885）  
3. **D4-D.1 integration audit** — cross-domain concern 报告与测试  
4. **D4-C.1-A shortcut diagnosis** — 已在 9c9e885；补遗文档若有  
5. **D4-C.1-B robust experiment** — `stain_*_v2`、`style_augment`、`robust_*`、`reports/d4c1b`、`docs/D4_C_1_B_*`、`test_input_guard_d4c1b.py`  
6. **D4-C.1-C representation experiment** — MixStyle/GRL、`reports/d4c1c`、`docs/D4_C_1_C_*`、`test_input_guard_d4c1c.py`  
7. **D4-C.1-D confounding audit** — `d4c1d_*`、`reports/d4c1d`、`docs/D4_C_1_D_*`、`test_input_guard_d4c1d.py`  
8. **D4-E final partial freeze** — policy 1.4 / contract 1.1、runtime deferred、`d4e_audit`、`docs/D4_E_*` / `D4_FINAL_*`、`test_input_guard_d4e.py`、`reports/d4/d4e_production_unified_audit.json`

## 明确排除

- `_audit/_semantic_check/`
- 根目录临时笔记（如需另议）
- 任何重新训练产物覆盖 v1 best.pt

## 操作约束

- 不自动 `git push`
- 不 `reset` / `clean` / 丢弃本地修改
- 需用户确认后再执行实际 commit
