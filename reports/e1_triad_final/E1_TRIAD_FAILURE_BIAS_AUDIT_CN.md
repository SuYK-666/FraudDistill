# E1 TRIAD 失败与偏差审计

{
  "case_control_selection": "E1-B 使用 PKU exact-q 1:1 case-control，只解释机制，不解释自然发生率。",
  "natural_low_prevalence": "E1-A/C2 使用 V10 自然低基率缓存，事件少时以 Wilson CI 和 AUPRC 为主。",
  "llm_gold_bias": "本次主运行使用 PKU 官方 safe/unsafe 标签作为 source proxy 并通过 Gold v2 validator；若写论文主表，仍需补齐双 LLM Gold。",
  "self_family_judge_bias": "Qwen/DeepSeek judge 未在本次低成本路径中重新全量调用，不能声称新增人工 Gold。",
  "final_push_boundary": {
    "status": "INVALID_STIMULUS_AND_GOLD_STOP",
    "q_rows": 60,
    "placeholder_q_rows": 60,
    "allowed_use": [
      "metadata_sham_control",
      "parser_regression",
      "engineering_ledger"
    ]
  }
}
