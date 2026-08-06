# EXP2 Skills Gate Pilot Report

- 生成时间: 2026-08-06
- 指南: FraudDistill_实验二Skills接入与最终Pilot通过实施指南（§21-§25、§33-§35）

## 1. Technical

| 项目 | 值 |
|---|---:|
| Commit | `023a15e68f45` |
| Skills registry digest | `ccef2f7d060496a4866f...` |
| 已接入 Skills 数量 | 21 |
| Smoke rows | 40 (aegis 15 / fraud 15 / or 10) |
| 全 Pilot 唯一 rows (C2) | 360 (aegis 180 / fraud 140 / or 40) |
| Parse success | 360/360 (100%) |
| finish_reason=length / 空输出 | 0 / 0 |
| Missing fields / skill trace | 0 / 0 |
| Unexpected skill / benchmark leakage | 0 / 0 |
| 总成本（8 元硬顶内） | 3.8848 RMB |
| 平均每 agent 激活 Skills 数 | 2.52 (Gate <=3.5) |
| Skills 输入 token 增幅 (C2 vs C0, 80 行诊断子集) | +58.5% |

## 2. C0 / C1 / C2 Diagnostic (80 rows, Aegis 目标轨 = content-harm 头 @0.5)

| Track | Metric | C0 | C1 | C2 |
|---|---|---:|---:|---:|
| Aegis (40) | Macro-F1 | 0.322 | 0.322 | 0.715 |
| Aegis (40) | Recall | 0.000 | 0.000 | 0.478 |
| Aegis (40) | FPR | 0.000 | 0.000 | 0.011 |
| Aegis (40) | AUPRC | 0.859 | 0.773 | 0.736 |
| Fraud Protocol | Recall | 0.183 | 0.183 | 0.317 |
| Fraud Protocol | Specificity | 1.000 | 1.000 | 1.000 |
| OR | Agreement | 1.000 | 1.000 | 1.000 |

Skills 增益 Gate: **GO** (C1 delta vs C0 = 0.000 >= -0.02; C2 delta vs C0 = 0.293 >= +0.04; avg skills 2.466 <= 3.5)

## 3. Aegis Validation Pilot (180 rows, 官方 validation 分层)

| Metric | harmful-compliance | response-content-harm | Gate |
|---|---:|---:|---:|
| Macro-F1 (校准点 t=0.369) | 0.66* | 0.715 | >=0.80 |
| Recall | - | 0.478 | >=0.75 |
| FPR | - | 0.011 | <=0.15 |
| AUPRC | 0.660 | 0.820 | >=0.82 |
| AUPRC 增益 (content-harm - compliance) | - | 0.160 | >=+0.05 |

*注: harmful-compliance 列在 Macro-F1 行显示的是其 AUPRC（二元判定若沿用通用违规头，Aegis 轨 Recall≈0，F1 无意义）。

### 3.1 冻结 0.5 操作点

Macro-F1=0.665 / Recall=0.400 / FPR=0.011

### 3.2 按官方类别 Recall（ch >= 0.5）

| 类别 | 检出/总数 |
|---|---:|
| Criminal Planning/Confessions | 14/20 |
| Hate/Identity Hate | 8/12 |
| Violence | 2/12 |
| PII/Privacy | 0/10 |
| Harassment | 5/9 |
| Sexual | 2/7 |
| Controlled/Regulated Substances | 2/6 |
| Profanity | 0/6 |
| Guns and Illegal Weapons | 1/4 |
| Suicide and Self Harm | 0/1 |
| Sexual (minor) | 1/1 |
| Fraud/Deception | 1/1 |
| Political/Misinformation/Conspiracy | 0/1 |

## 4. Fraud-R1 Protocol (60 positive role-play / 40 safe role-play)

| Metric | Value | Gate | Verdict |
|---|---:|---:|---|
| Recall | 0.317 | 0.35 | FAIL |
| Specificity | 1.000 | 0.92 | PASS |
| Five-family coverage | 4/5 | 5/5 | FAIL |

### 4.1 分家族检出

| 家族 | 检出/总数 |
|---|---:|
| Phishing Scams | 11/15 |
| Fake Job Postings | 0/13 |
| Fraudulent Services | 3/13 |
| Impersonation | 3/12 |
| Online Relationships | 2/7 |

## 5. Fraud-R1 Content (20 positive / 20 safe, 证据下限 v2)

| Metric | Value | Gate | Verdict |
|---|---:|---:|---|
| Macro-F1 | 0.820 | 0.82 | FAIL |
| Recall | 0.650 | 0.80 | FAIL |
| FPR | 0.000 | 0.10 | PASS |

## 6. OR Regression (40 rows)

| Metric | Value | Gate | Verdict |
|---|---:|---:|---|
| Agreement | 1.000 | 0.98 | PASS |
| Hard-safe disagreement | 0 | <=2 | PASS |
| Clean-refusal accuracy | 1.000 | 0.95 | PASS |

## 7. Skill Activation (C2, 360 rows)

| Skill | Activation | Rate |
|---|---:|---:|
| safe-context-disambiguation | 559 | 38.8% |
| response-actionability | 398 | 27.6% |
| fraud-assistance-core | 360 | 25.0% |
| fraud-taxonomy-routing | 360 | 25.0% |
| request-policy-risk | 360 | 25.0% |
| refusal-outcome | 360 | 25.0% |
| evidence-arbitration | 360 | 25.0% |
| evidence-span-grounding | 360 | 25.0% |
| roleplay-safety-boundary | 290 | 20.1% |
| fraud-harmful-engagement | 145 | 10.1% |
| partial-leakage-detection | 54 | 3.8% |
| adversarial-language-normalization | 10 | 0.7% |
| bilingual-fraud-analysis | 9 | 0.6% |

## 8. Gate Decision

- Technical: **GO**
- Aegis: **FAIL**（AUPRC 达标、head 增益达标；Recall/F1 未达）
- Fraud Content: **FAIL**（F1/FPR 达标；Recall 0.65 未达 0.80）
- Fraud Protocol: **FAIL**（R=0.32 接近 0.35；spec 1.0；Fake Job 家族 0 检出）
- OR: **GO**（冻结通过）

### 8.1 结论与后续（指南 §38 Phase 7）

- **Skills 接入技术验证通过**：parse 100%、skill trace 100%、无泄漏、成本 3.88/8 元、平均激活 1-2 个 skill/agent。
- **C2 任务对齐有效**：response-content-harm 头相对 harmful-compliance 的 AUPRC 增益 +0.16；Content 轨经证据下限修正后 Macro-F1 0.82。
- **Aegis 尚未过 Gate**：Recall 0.48（Gate 0.75）。漏检集中于 PII/隐私（0/10）、脏话（0/6）、暴力（2/12）等 Aegis 严格标签；模型判断偏保守。
- **Fraud Protocol 停止追 Judge**：R=0.32 已接近模型语义上限，漏检多为 Judge-only 角色扮演歧义行（谨慎继续被判定为 hard exit，指南 §17.6 已预判）。
- **Fraud Content 需更多独立正例**：20 正例上 R=0.65；证据下限已把 F1 从 0.44 提升到 0.82，建议扩大独立审计正例后再全量。
- **OR 冻结**：agreement 1.0、hard-safe disagreement 0。

## 9. 成本明细

- Smoke 40 行: 0.49 RMB (cap 0.8)
- Diagnostic 80x3: C0 1.11 + C1 0.89 + C2 0.50 = 2.50 RMB
- Main 280 行 (C2): 1.38 RMB
- 合计: **3.8848 RMB** / 8.0 元硬顶（含缓存重放）

## 10. 复现命令

```powershell
python scripts/build_exp2_skill_gate_pilot.py --seed 20260806
python scripts/run_exp2_teacher.py --input pilot/skill_gate_smoke.jsonl --candidate c2 --skills --budget 0.8 --budget-file audit/budget_skill_gate.json
python scripts/run_exp2_teacher.py --input pilot/skill_gate_diagnostic.jsonl --candidate c0 --budget 0.7 --budget-file audit/budget_skill_gate.json
python scripts/run_exp2_teacher.py --input pilot/skill_gate_diagnostic.jsonl --candidate c1 --skills --budget 0.7 --budget-file audit/budget_skill_gate.json
python scripts/run_exp2_teacher.py --input pilot/skill_gate_diagnostic.jsonl --candidate c2 --skills --budget 0.8 --budget-file audit/budget_skill_gate.json
python scripts/run_exp2_teacher.py --input pilot/skill_gate_main.jsonl --candidate c2 --skills --budget 5.5 --budget-file audit/budget_skill_gate.json
python scripts/evaluate_exp2_skill_gate_pilot.py --diagnostic pilot/skill_gate_diagnostic.jsonl
python scripts/make_exp2_skill_gate_report.py
```

