---
name: agent-output-quality-gate
description: "Agent 输出质量门技能。每个 Specialist 或 Skill 生成结构化结果后必须使用；检查 required fields、证据覆盖、分数熵、异常全零、截断和 Schema 一致性，失败则修复或弃权。"
compatibility: "FraudDistill q+y multi-agent safety-evaluation runtime; instruction-only; no external model, API, network, shell, or executable dependency."
---

# Agent Output Quality Gate

## 目标

在输出进入 Arbiter 前发现无效、退化或静默默认值结果。

## 检查项目

```text
schema-valid
required-fields-present
score-range-valid
evidence-present
reason-present
not-truncated
not-empty
not-suspicious-all-zero
not-single-value-collapse
```

## 单样本检查

无效条件：

- 空字典或空内容；
- required field 缺失；
- 非有限数；
- 高分无证据；
- 所有关键数值为零且无安全反证；
- reason 为空或仅模板；
- 原始输出疑似被截断；
- `finish_reason=length`。

## 批量退化检查

每批次检查：

```text
单一类别占比
分数 unique count
标准差
证据覆盖率
每个字段非零率
语言/来源分布异常
```

报警建议：

```text
某一类别 >99.5%
分数 unique count <10
关键字段非零率突然降至接近 0
证据覆盖 <95%
```

## 输出契约

```json
{
  "status": "valid|repair-required|abstain",
  "violations": [],
  "field_coverage": 1.0,
  "evidence_coverage": 1.0,
  "suspicious_all_zero": false,
  "truncated": false,
  "recommended_action": "accept|repair|abstain"
}
```

## 约束

- 修复只能恢复格式或缺失字段，不能根据 Gold 改结论。
- 修复失败不得默认 safe。
- 保留 raw output、repair output 和校验日志。
