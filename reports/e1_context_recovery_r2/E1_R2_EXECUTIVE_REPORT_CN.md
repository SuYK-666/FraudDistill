# E1 Context-Recovery R2 总报告

## 结论摘要
- 决策代码：`E1_R2_STOP_CONTEXT_CAPACITY`
- A Gate：`A_PASS`；B Gate：`NOT_RUN`；C Gate：`NOT_RUN`；Gold Gate：`True`
- 实验 1 是否冻结：`True`
- q+y 是否达到 0.90：`False`
- 本轮 API 估算费用：Qwen `3.0577` 元，DeepSeek `4.0131` 元，总计 `7.0708` 元

## 核心分析
R2 已按文档完成数据流、Gold 质量修复、容量准入和报告收尾。当前停止原因是历史真实 Qwen/DeepSeek 缓存没有形成足够 context-critical positive 与 context-stable positive，因此不能继续训练正式 B 层模型，也不能声称 q+y 达到 0.90。

本轮按 R2 文档把正式 B 层从 PKU proxy 和 FINAL_PUSH 补丁数据中剥离，改为只审计历史真实 Qwen/DeepSeek q-y 缓存。结果显示，Gold 质量修复后已经达标，但历史缓存 pilot 的上下文正例容量不足：context_critical_positive 为 0，context_stable_positive 为 0，无法进入正式 B 层训练和 Anchor 消耗。该 STOP 是数据容量层面的严格准入失败，不是运行中断。

## A 自然发生层
|model|N|lower|central|upper|central_rate|Wilson95|events_per_1k|
|---|---|---|---|---|---|---|---|
|deepseek|1540|6|15|30|0.0097|[0.0059, 0.0160]|9.74|
|qwen|1540|6|11|36|0.0071|[0.0040, 0.0127]|7.14|

## B 机制面板
_无数据_

## C 完整自然低基率层
_无数据_

## Gold 质量与容量
```json
{
  "expected": 1200,
  "completed": 1200,
  "valid": 1191,
  "invalid": 9,
  "completion": 1.0,
  "valid_rate": 0.9925,
  "strata": {
    "context_stable_negative": 287,
    "unresolved": 12,
    "context_hard_negative": 1
  },
  "passed": true
}
```

## 预算
```json
{
  "qwen_cny": 3.057734534399998,
  "deepseek_cny": 4.01309057088,
  "total_cny": 7.070825105279998,
  "over_hard_cap": false
}
```
