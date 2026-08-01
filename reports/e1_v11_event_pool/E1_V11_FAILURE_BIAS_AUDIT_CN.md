# E1 V11 失败与偏差审计

{
  "decision": {
    "decision": "E1_V11_STOP_INVALID",
    "a_gate": "PASS",
    "b1_gate": "EXPLORATORY",
    "b2_gate": "EXPLORATORY",
    "b3_gate": "EXPLORATORY",
    "c_gate": "EXPLORATORY",
    "hard_gates_ok": false,
    "b1_qy_macro_f1": 0.0,
    "b1_delta_qy_y": 0.0,
    "c_auprc_relative_improvement": 0
  },
  "known_biases": [
    "B1 是风险富集 case-control，不代表自然 prevalence。",
    "候选检索可能造成 spectrum bias，已通过 screen-negative audit 披露。",
    "Qwen-Max 同时参与 adjudication 和主 evaluator，因此 primary headline 只使用 consensus-core Gold。",
    "若 C positive_n 不足，C 只能解释为低基率趋势。"
  ],
  "budget": {
    "qwen_cny": 35.261947382400024,
    "deepseek_cny": 12.416672996639997,
    "total_cny": 47.67862037904002,
    "over_hard_cap": false,
    "hard_caps": {
      "qwen": 48.0,
      "deepseek": 48.0
    }
  }
}
