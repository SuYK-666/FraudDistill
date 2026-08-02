# E1 FINAL TRIAD v3.1 正式推进与大规模执行冻结方案

## 首屏关键信息
- final decision code：`E1_V31_BEHAVIOR_PASS_MECHANISM_WEAK`
- Git commit：`115ef0a48b99844782f0e977ce7ee59c29cde0a4`
- worktree status：`M reports/e1_final_triad_v31/E1_V31_A_TARGET_QUALITY_REPORT.md
 M reports/e1_final_triad_v31/E1_V31_BUDGET_REPORT.md
 M reports/e1_final_triad_v31/E1_V31_DATA_PROVENANCE_AUDIT.md
 M reports/e1_final_triad_v31/E1_V31_EXECUTIVE_REPORT_CN.md
 M reports/e1_final_triad_v31/E1_V31_FAILURE_BIAS_AUDIT_CN.md
 M reports/e1_final_triad_v31/E1_V31_FULL_ANALYSIS_REPORT_CN.md
 M reports/e1_final_triad_v31/E1_V31_GOLD_QUALITY_REPORT.md
 M reports/e1_final_triad_v31/E1_V31_PAPER_TABLES.md
 M reports/e1_final_triad_v31/E1_V31_REPRODUCTION_GUIDE.md
 M reports/e1_final_triad_v31/E1_V31_STATISTICAL_APPENDIX_CN.md
 M reports/e1_final_triad_v31/E1_V31_TASK_CLOSEOUT_CN.md
 M scripts/run_e1_final_triad_v3.py
 M src/frauddistill/e1_final_v3/stats_v31.py`
- protocol：`E1-FINAL-TRIAD-v3.1-7500-3200-RealPrevalence`
- A/B/C 门禁：A manifest `PASS`；A target `PASS`；A Gold `PASS`；B `PASS`；C `PASS`
- 新增 API 调用：target `4418`，双 Gold 已标 `4418`，ledger rows=`19961`
- A7500 构成：新增 `4418` + 复用 `3082`，完整 qwen/deepseek pair=`2209`
- A central prevalence：`0.0037333333333333333`，positive=`28`，95% CI `0.002584312137040857`-`0.005390464402294347`
- B 冻结 Anchor Macro-F1：q-only=`0.7960812499216501`，y-only=`0.9431777825129626`，q+y=`0.9497045140198667`
- C 回放：q+y AUPRC=`0.49175106357368636`，y-only AUPRC=`0.38040154124258135`，AUPRC ratio=`1.2927157496980215`

## 分析与结论
A7500：canonical cases=2141，prompt instances=3750，复用 responses=3082，新增 API 调用=4418。
A Gold：PASS，completion=1.0，binary agreement=0.9995473064735174，PABAK=0.9990946129470348。
A central prevalence=0.0037333333333333333，positive=28，95% CI 0.002584312137040857-0.005390464402294347。
B 面板：formal_panel_ready=True，by stratum={'context_stable_positive': 318, 'context_stable_negative': 2550, 'context_critical_positive': 12, 'context_hard_negative': 320}。
B Anchor：q+y Macro-F1=0.9497045140198667，SD 0.0；y-only=0.9431777825129626；q-only=0.7960812499216501。
C 回放：can_run=True，q+y AUPRC=0.49175106357368636，y-only AUPRC=0.38040154124258135。
最终 decision code：`E1_V31_BEHAVIOR_PASS_MECHANISM_WEAK`。