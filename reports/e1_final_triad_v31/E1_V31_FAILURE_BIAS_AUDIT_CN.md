# E1 v3.1 失败偏差与反快捷方式审计

```json
{
  "analysis": "A7500：canonical cases=2141，prompt instances=3750，复用 responses=3082，新增 API 调用=4418。\nA Gold：PASS，completion=1.0，binary agreement=0.9995473064735174，PABAK=0.9990946129470348。\nA central prevalence=0.0037333333333333333，positive=28，95% CI 0.002584312137040857-0.005390464402294347。\nB 面板：formal_panel_ready=True，by stratum={'context_stable_positive': 318, 'context_stable_negative': 2550, 'context_critical_positive': 12, 'context_hard_negative': 320}。\nB Anchor：q+y Macro-F1=0.9497045140198667，SD 0.0；y-only=0.9431777825129626；q-only=0.7960812499216501。\nC 回放：can_run=True，q+y AUPRC=0.49175106357368636，y-only AUPRC=0.38040154124258135。\n最终 decision code：`E1_V31_BEHAVIOR_PASS_MECHANISM_WEAK`。",
  "b_rejects": 0,
  "shortcut_audits": {
    "provenance_shortcut_auc": 0.9135804033365009,
    "nuisance_baseline_auc": 0.9393348115299336,
    "wrong_q_perm_orig_macro_f1": 0.9572266640263373,
    "wrong_q_perm_permuted_macro_f1": 0.8109954124073049,
    "wrong_q_perm_drop": 0.1462312516190324
  },
  "c_note": "E1-C is NOT an unseen generalization experiment; it replays the frozen B detector on the A7500 real distribution."
}
```
