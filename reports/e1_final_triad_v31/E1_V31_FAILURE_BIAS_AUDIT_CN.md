# E1 v3.1 失败与偏差审计

本轮已将 v3 dry-run 骨架升级为 v3.1 可执行状态机：A manifest、API Gate、fingerprint 缓存、预算 ledger、历史 roleplay pair 复用、B 容量审计和 C 准入均已接入。

A 层 manifest：canonical cases=2141，assistant=2141，roleplay reused=1541，roleplay extra=68，target prompt instances=3750，pending target calls=4418。

A target 当前状态：{'new_response_rows': 4418, 'valid_new_response_rows': 4418, 'complete_new_pairs': 2209, 'pending_target_calls_initial': 4418, 'target_gate': 'PASS'}。只有 P0 clean 且 health/generate 真正完成后，A7500 才能冻结。

B 预筛状态：stable+=25，stable-=3048，critical+=1，hard-=6。B 仍需正式 Gold 与受控合成补齐。

最终 decision code：`E1_V31_PENDING_B_PANEL`。
