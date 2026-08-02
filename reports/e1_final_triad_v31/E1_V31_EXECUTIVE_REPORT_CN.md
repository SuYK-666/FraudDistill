# E1 FINAL TRIAD v3.1 执行总报告

## 首屏摘要
- final decision code：`E1_V31_STOP_P0_NOT_CLEAN_OR_SOURCE`
- Git commit：`0a3bae9b0a76441a23e244b674912e7c969549b0`
- worktree status：`M configs/experiments/e1_final_triad_v3.yaml
 D reports/e1_final_triad_v3/E1_V3_BUDGET_REPORT.md
 D reports/e1_final_triad_v3/E1_V3_DATA_AUDIT.md
 D reports/e1_final_triad_v3/E1_V3_FULL_REPORT_CN.md
 D reports/e1_final_triad_v3/E1_V3_REPRODUCTION_GUIDE.md
 D reports/e1_final_triad_v3/E1_V3_TASK_OVERVIEW_CN.md
 M scripts/run_e1_final_triad_v3.py
 M src/frauddistill/e1_final_v3/registry.py
 M src/frauddistill/e1_v8/official_prompt_renderer.py
?? reports/e1_final_triad_v31/
?? src/frauddistill/e1_final_v3/api_executor.py`
- protocol：`E1-FINAL-TRIAD-v3.1-7500-3200-RealPrevalence`
- A/B/C 状态：A manifest `PASS`，A target `PENDING`，B `NOT_READY`，C `NOT_READY`
- 本轮新 API 调用数：`0`；成功数：`0`
- A7500 规划：prompt instances `3750`，复用 responses `3082`，待调用 `4418`

## 分析
本轮已将 v3 dry-run 骨架升级为 v3.1 可执行状态机：A manifest、API Gate、fingerprint 缓存、预算 ledger、历史 roleplay pair 复用、B 容量审计和 C 准入均已接入。

A 层 manifest：canonical cases=2141，assistant=2141，roleplay reused=1541，roleplay extra=68，target prompt instances=3750，pending target calls=4418。

A target 当前状态：{'new_response_rows': 0, 'valid_new_response_rows': 0, 'complete_new_pairs': 0, 'pending_target_calls_initial': 4418, 'target_gate': 'PENDING'}。只有 P0 clean 且 health/generate 真正完成后，A7500 才能冻结。

B 预筛状态：stable+=25，stable-=3048，critical+=1，hard-=6。B 仍需正式 Gold 与受控合成补齐。

最终 decision code：`E1_V31_STOP_P0_NOT_CLEAN_OR_SOURCE`。