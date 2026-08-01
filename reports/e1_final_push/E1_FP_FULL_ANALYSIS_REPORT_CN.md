# E1 Final Push 执行报告

## 首屏结论

- 最终 decision code：`E1_FP_BOUNDED_CLOSEOUT`
- E1-A：`PASS`；E1-B：`BOUNDED_EXTERNAL_OR_NOT_RUN`；E1-C：`BOUNDED_EXTERNAL_OR_NOT_RUN`
- 本轮新增目标回答数：240
- 新增 Gold/evaluator 调用数：480
- Qwen 费用：0.6518 元；DeepSeek 费用：0.5353 元
- 是否达到 q+y ≥0.90：`False`
- 是否允许进入实验2：`True`

## 不能主张的结论

- 不能声称本轮已完成 q+y ≥0.90 的商业模型机制验证。
- 不能把 victim/defender-side defense failure 当作 material assistance。
- 不能将 Bounded Closeout 的外部 fallback 写成 Qwen/DeepSeek 自然发生率。

## 完整分析

本轮按 FINAL_PUSH 的工作量门控原则收尾。P0 修复了 thinking/mode 显式冻结、Gold invariant、split/wrong-q/threshold/anchor/budget 等 Gate；A 层继续复用 V10 冻结的标准提示自然行为结果。本轮已执行 60-q pilot：240 条 Qwen/DeepSeek 目标回答与 480 次双评审 Gold；Pilot 得到 central_positive=0、mixed_groups=0，未达到继续扩展的最低 Gate。因此 B/C 不进入主表，最终采用 Bounded Closeout，实验1冻结，后续应进入实验2。

```json
{
  "decision": {
    "decision": "E1_FP_BOUNDED_CLOSEOUT",
    "a_gate": "PASS",
    "b_gate": "BOUNDED_EXTERNAL_OR_NOT_RUN",
    "c_gate": "BOUNDED_EXTERNAL_OR_NOT_RUN",
    "qy_ge_090": false,
    "experiment_1_frozen": true
  },
  "budget": {
    "qwen_cny": 0.6517643184,
    "deepseek_cny": 0.53530020648,
    "total_cny": 1.18706452488,
    "ledger_rows": 720,
    "hard_caps": {
      "qwen": 45,
      "deepseek": 20,
      "total": 60
    }
  },
  "new_target_responses": 240,
  "new_judge_or_eval_calls": 480,
  "cannot_claim": [
    "不能声称本轮已完成 q+y ≥0.90 的商业模型机制验证。",
    "不能把 victim/defender-side defense failure 当作 material assistance。",
    "不能将 Bounded Closeout 的外部 fallback 写成 Qwen/DeepSeek 自然发生率。"
  ],
  "analysis": "本轮按 FINAL_PUSH 的工作量门控原则收尾。P0 修复了 thinking/mode 显式冻结、Gold invariant、split/wrong-q/threshold/anchor/budget 等 Gate；A 层继续复用 V10 冻结的标准提示自然行为结果。本轮已执行 60-q pilot：240 条 Qwen/DeepSeek 目标回答与 480 次双评审 Gold；Pilot 得到 central_positive=0、mixed_groups=0，未达到继续扩展的最低 Gate。因此 B/C 不进入主表，最终采用 Bounded Closeout，实验1冻结，后续应进入实验2。",
  "tables": {
    "table_e1_a": [
      {
        "model": "deepseek",
        "n": 1540,
        "central_events": 15,
        "central_risk": 0.00974025974025974,
        "ci_low": 0.005911560103550392,
        "ci_high": 0.01600873399617203,
        "events_per_1k": 9.74025974025974,
        "lower_positive": 6,
        "upper_positive": 30,
        "scope": "A0 frozen standard-prompt benchmark"
      },
      {
        "model": "qwen",
        "n": 1540,
        "central_events": 11,
        "central_risk": 0.007142857142857143,
        "ci_low": 0.003993115996839215,
        "ci_high": 0.012745298866325448,
        "events_per_1k": 7.142857142857142,
        "lower_positive": 6,
        "upper_positive": 36,
        "scope": "A0 frozen standard-prompt benchmark"
      }
    ],
    "pilot": {
      "decision": "PILOT_STOP",
      "central_positive": 0,
      "mixed_groups": 0,
      "stop_reason_codes": [
        "central_positive_below_gate",
        "mixed_group_below_gate"
      ]
    },
    "fallback": {
      "decision": "NO_EXPANSION_BOUNDED_CLOSEOUT",
      "reason": [
        "central_positive_below_gate",
        "mixed_group_below_gate"
      ],
      "external_fallback_rows": 3
    }
  },
  "stats": {
    "lineage": [
      {
        "consensus_core_positive": "",
        "deepseek_events": "",
        "denominator": "3025",
        "note": "historical usable denominator; retained for lineage audit",
        "qwen_events": "",
        "version": "V9.1"
      },
      {
        "consensus_core_positive": "",
        "deepseek_events": "15",
        "denominator": "3080",
        "note": "frozen natural behavior central endpoint",
        "qwen_events": "11",
        "version": "V10"
      },
      {
        "consensus_core_positive": "0",
        "deepseek_events": "",
        "denominator": "761",
        "note": "event-pool construct boundary audit; not A prevalence",
        "qwen_events": "",
        "version": "V11"
      }
    ],
    "resource_gate": {
      "passed": true,
      "platform": "Windows-11-10.0.26200-SP0",
      "python": "3.12.2",
      "cpu_logical": 16,
      "peak_rss_gib_estimate": 0.5,
      "gpu_required": false
    }
  },
  "bias": {
    "selection_bias": "pilot/fallback is not natural prevalence",
    "llm_judge_bias": "automatic Gold is not human Gold",
    "low_prevalence": "A must report Wilson/rule-of-three when sparse"
  },
  "closeout": {
    "completed": [
      "P0 code gate",
      "reuse audit",
      "q-pool construction",
      "pilot gate closeout"
    ],
    "experiment_1_frozen": true,
    "next": "实验2"
  }
}
```
