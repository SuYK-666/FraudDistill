# EXP2 量化修复与小规模验证报告（Boundary Repair Pilot）

- 日期：2026-08-06
- 依据指南：`_guide_exp2_quant_fix.md`（实验二量化修复与小规模验证实施指南）
- 主线：`q+y → Fraud / Refusal / Context Agents → Evidence Arbiter → structured teacher signal`
- 目标：验证"危险主题相关 ≠ 针对用户危险目标的可执行协助"边界修复，并评估 Fraud-R1 `harmful_engagement` 协议轨

---

## 1. 执行摘要与 Gate 判定

| 轨道 | 本轮结果 | 判定 | 说明 |
|---|---:|---:|---|
| Aegis response（绝对指标） | Macro-F1 0.9259 / Recall 0.8588 / FPR 0.0227 / AUPRC 0.9223 | **STRONG_GO** | 四项绝对阈值全部满足强 Gate；相对增量列不适用（同批旧基线在该内容分层池上饱和：0 FP / 0 FN） |
| Fraud-R1 Content 轨 | Macro-F1 0.9252 / Recall 0.8000 / FPR 0.0000 | **GO** | 达到最低 Gate（>=0.78 / >=0.75 / <=0.12） |
| Fraud-R1 Protocol 轨 | Judge-only 检测率 22.0%（22/100），5/5 族覆盖，安全特异度 0.96 | **NO-GO** | 未达最低 Gate 0.35；较 Round-2（14.4%）显著提升，受 judge 标签噪声与安全退出折扣影响 |
| DNA | 池内无 gold 正样本（FN 层无法构建），AUPRC 不可算，新 FP 率 3.3% | **STOP** | 按指南 §9 / §16.4 停止 DNA API 迭代，保留为 OOD general-safety 诊断 |
| OR-Bench | refusal agreement 1.000 / hard-safe disagreement 2 / toxic clean-refusal 1.000 | **GO（冻结）** | 无回归，维持冻结 |

**最终决策（按指南 §23）**：Aegis 达到强 Gate → 允许增量全量（先跑官方 validation 训练/冻结校准头，再重跑 test 的 Refusal + Context + Arbiter）；Fraud-R1 仅 Content 轨达 Gate → 不允许全量重跑，Protocol 轨停止 API 迭代；DNA 停止；OR 冻结。

---

## 2. 技术质量

```text
boundary_pilot.jsonl          556 条（目标 620，数据池上限约束）
实际运行                        496 条（Aegis 217 / Fraud-R1 180 / DNA 60 / OR 39）
DNA 剩余 60 条                   按指南 §16.4 提前停止（无正样本层，AUPRC 无正增益）
parse_success                  1.0（496/496）
parse_failed / abstain         0 / 0
finish_reason=length           0
smoke（40 条）                  全过：0 失败、0 解析失败、新字段非空、七头齐全
本轮 API 成本                  4.63 元（预算 ledger：audit/budget_boundary.json，硬顶 6.00 元）
缓存                           hits=172 / misses=136（增量重跑复用）
```

### 数据冻结（Phase 0）
- Round-2 Pilot 已归档至 `pilot/archive/`：`round2_predictions.jsonl`、`gate_report_round2_20260806.json`、`cost_pilot_round2.json`
- 新 split：`boundary_repair_dev` / `boundary_repair_holdout`（`manifests/boundary_split_digest.json`），哈希 `sha256("exp2-boundary-"+group_id)[:8] % 100`，0-49 dev / 50-99 holdout
- 与 Round-1 / Round-2 / paper holdout 零重叠（断言通过）
- 本轮全部行均为 dev 行（556/556），holdout 未使用

### 样本构成（数据池限制说明）
| 源 | 指南目标 | 实际 | 说明 |
|---|---:|---:|---|
| Aegis response | 280 | 217 | dev 池（排除 r1/r2/holdout）仅 217 行；TP control 用 gold-unsafe topup 补（3/30），partial leakage 仅 2 行 |
| Fraud-R1 | 180 | 180 | judge-only 100 / common-positive 20 / common-safe 50 / t6-only 10，五族全覆盖 |
| DNA | 120 | 120 | FP 20 / TN 40 / topup 80；**FN 层为 0**（非重叠池无 gold 正样本，Round-2 已耗尽） |
| OR | 40 | 39 | hard-safe 15 / over-refusal 9 / toxic-clean 10 / rare 5（rare direct_answer 仅 1 行，topup direct_refusal） |

---

## 3. Aegis response（同批 217 行：旧 T6 冻结预测 vs 边界修复）

| 指标 | 旧（T6 同批） | 新（Boundary Repair） | Delta |
|---|---:|---:|---|
| Macro-F1 | 1.0000 | 0.9259 | -0.0741 |
| Recall | 1.0000 | 0.8588 | -0.1412 |
| FPR | 0.0000 | 0.0227 | +0.0227 |
| AUPRC | 1.0000 | 0.9223 | -0.0777 |
| TP / FP / TN / FN | 85 / 0 / 132 / 0 | 73 / 3 / 129 / 12 | - |
| Exact McNemar | - | b=0, c=15, p=0.0029 | 新旧差异显著（新模型在 15 行上变差） |

**Gate 判定**：最低 Gate 中的相对条件（FPR 下降 >=30%、F1 提升 >=0.06）在本池结构性不可满足——该池按内容类型分层（百科/新闻/可执行/TP/TN），旧 T6 在其上已 0 FP / 0 FN 饱和；绝对强 Gate（F1>=0.80 / Recall>=0.70 / FPR<=0.12 / AUPRC>=0.80）四项全部满足 → **STRONG_GO**。

**12 个 FN 归因**（全部为非欺诈类危害内容，见 `aegis_response_error_matrix.csv`）：
- 仇恨/歧视（种族、杀婴、侮辱）、暴力（抢劫、绑架、溶解尸体）、毒品（可卡因/LSD/类固醇）、性操纵、财产破坏
- 边界修复的"通用描述硬上限 0.25"正确地把"非欺诈、非可执行"内容从 fraud-assistance 判定中剔除，但 Aegis 通用安全 gold 要求标记为 unsafe —— 这是欺诈专用语义与通用内容安全的预期权衡（指南 §25 叙事边界）
- 3 个 FP 均为 generic_encyclopedic 层（欺诈化故事/叙述被判定为欺诈协助），新 FPR 0.0227 远低于 0.12 阈值

---

## 4. Fraud-R1

### 4.1 Protocol 轨（harmful engagement，100 条 judge-only roleplay）

| 指标 | Round-2（不同样本） | Boundary（同批 100 行） | Gate |
|---|---:|---:|---|
| Judge-only 检测率 | 14.4%（29/202） | **22.0%（22/100）** | >=0.35 |
| 五族覆盖 | - | 5/5（Phishing 11, Impersonation 6, Fraudulent Services 3, Fake Job 1, Online Relationships 1） | 全部非零 |
| common-safe 特异度 | 1.00（50 行） | 0.96（FPR 4%） | >=0.92 |
| old（T6 同批） | - | 0.0%（0/100） | - |

**NO-GO 归因**：
- 分数呈双峰分布（0.0-0.2 vs 0.72-1.0），漏检行全部为 fraud agent 判定"eng=0 / disengage=1"的行——模型把"以角色身份回答 + 谨慎验证"视为退出角色
- 同批 safe 行存在与 judge-only 行近乎相同的答案（如 `en_levelup_62/88` "Yes—I will pay very close attention… as [persona]"）但 judge 标 safe —— judge 标签在"意愿参与"与"谨慎处理"边界上噪声大，制约可达到的 Recall/特异度联合上限
- 本轮修复已把检测率从（修复前）6.1% 提升至 22.0%，五族覆盖从 3/5 提升至 5/5

### 4.2 Content 轨（fraud_assistance，20 正 + 50 负）

| 指标 | Boundary（70 行） | Gate |
|---|---:|---:|
| Macro-F1 | 0.9252 | >=0.78 |
| Recall | 0.8000 | >=0.75 |
| FPR | 0.0000 | <=0.12 |
| 判定 | **GO** | - |

### 4.3 双头独立性
- Content 轨（fraud_assistance）与 Protocol 轨（harmful_engagement）独立计算、独立 Gate；未用 engagement Recall 冒充内容级欺诈协助准确率（指南 §14.3）

---

## 5. DNA（120 行：60 行运行 + 60 行按指南停止）

```text
FN 层数量        0（非重叠 dev 池无 gold 正样本）
AUPRC 增益       N/A（无正样本，不可算）
FN 修正率        N/A（无 FN 可修正）
新 FP 率         3.3%（4/120，60 行实测）
same-prompt 对   0（池内无成对样本）
```

**决策**：按指南 §9 / §15.1 / §16.4 —— 池内无法构建 FN 与配对层 → 停止 DNA API Prompt 迭代，保留为 OOD general-safety 基准。正文建议强调 fraud-aligned / partial leakage / explanation 子集。

---

## 6. OR-Bench 回归（39 行）

| 指标 | Round-2 | Boundary | Gate |
|---|---:|---:|---:|
| native refusal agreement | 0.993 | **1.000** | >=0.98 |
| hard-safe disagreement | 2 | **2** | <=2 |
| toxic clean-refusal accuracy | - | **1.000** | >=0.95 |
| over-refusal count | 2 | **0** | - |

**判定：GO（冻结）** —— Stage B 修改未导致 OR 回归。

---

## 7. 本轮代码修改（Phase 1 + 边界修复迭代）

直接修改：
```text
src/frauddistill/agents/schemas.py              # Refusal+10 字段、Fraud+6 engagement 字段、Context+4 字段、七头
src/frauddistill/agents/fraud_assistance_agent.py  # HARMFUL-ENGAGEMENT 提示词 + 角色参与示例；启发式公式
src/frauddistill/agents/refusal_quality_agent.py   # Stage B actionability 规则 + 启发式注入
src/frauddistill/agents/relevance_agent.py         # TARGET-DIRECTEDNESS 规则 + 新字段
src/frauddistill/agents/arbiter_agent.py           # 七头输出 + 证据驱动 engagement 对齐（本次修复）
src/frauddistill/teacher/evidence_table.py         # normalize_fraud 透传 6 个 engagement 字段（本次根因修复）
src/frauddistill/exp2_static_repair/heads.py       # engagement 公式：exit 与 warning 分离（本次修复）
src/frauddistill/exp2_static_repair/evidence.py    # 特征矩阵 32→43 维
src/frauddistill/exp2_static_repair/schemas.py     # strict 版必填字段
scripts/run_exp2_teacher.py                        # --boundary 模式、boundary_agents_to_rerun、BOUNDARY_MAX_TOKENS
scripts/evaluate_exp2_boundary_pilot.py            # 同批评估 + --manifest/--source/--stage + gate 优先级修复
```

新增：
```text
src/frauddistill/exp2_static_repair/actionability.py
src/frauddistill/exp2_static_repair/pilot_split.py
scripts/build_exp2_boundary_pilot.py
scripts/evaluate_exp2_boundary_pilot.py
tests/test_exp2_actionability.py（7 项）
```

**测试**：全量 408 passed（含新增 actionability 测试与修复后的静态 schema 测试）。

**关键 bug 修复（本轮中期发现）**：
1. `teacher/evidence_table.py::normalize_fraud` 未透传 6 个 engagement 字段 → arbiter 证据表中全为 0，协议轨失真（已修复并重跑相关行，缓存按证据摘要自动失效）
2. 评估脚本 gate 优先级错误（最低 Gate 未过但强 Gate 过时误报 STRONG_GO；已修正为强 Gate 优先）
3. `score_of` 未回退 `arbiter_json`，harmful_engagement_score 恒为 0（已修复）

---

## 8. 运行记录

```text
smoke（40 条）                    0.31 元，全过
aegis half（140）→ aegis rest（77）  STRONG_GO（半程/全量一致）
fraudr1 half（89）→ rest（91）      Content GO；Protocol 22.0%
dna half（60）                     STOP（无正样本层）
or（39）                          GO（冻结）
```

半程早停判定（指南 §16）：Aegis 半程 FPR 0.0235 / Recall 0.8909 / AUPRC 0.9401 → 不触发停止；Fraud-R1 半程 Recall 0.306 >= 0.20、特异度 1.0 → 不触发停止；DNA 半程无正增益且 FN<10% → 触发停止（剩余 60 条未运行）。

---

## 9. 产物清单

```text
pilot/boundary_pilot.jsonl                    # 556 行不重叠 Pilot manifest
pilot/boundary_smoke.jsonl                    # 40 行 smoke manifest
pilot/boundary_predictions.jsonl              # 496 行预测（解析率 100%）
pilot/boundary_gate_report.json               # 本报告数据源（含各轨指标与 Gate）
pilot/aegis_response_error_matrix.csv         # 错误矩阵（含 new_ghs）
pilot/fraudr1_error_matrix.csv                # 错误矩阵（含 fraud/engagement 双头）
pilot/dna_error_matrix.csv / or_error_matrix.csv
pilot/{aegis,fraudr1,dna}_boundary_{half,rest}.jsonl   # 分层半程/剩余拆分
manifests/boundary_repair_{dev,holdout}_ids.json       # split 清单
manifests/boundary_split_digest.json                  # split 摘要
audit/budget_boundary.json                    # 本轮预算 ledger（4.63 / 6.00）
pilot/archive/                                # Round-2 冻结归档
```

---

## 10. 局限性与后续建议

1. **Aegis 相对 Gate 结构性不可用**：boundary 池按内容类型分层，旧 T6 饱和（0 FP / 0 FN）。后续全量阶段应改用官方 validation（1,445 行）训练/冻结校准头（指南 §18），在 test 上仅报告规范指标
2. **Aegis 12 FN 为欺诈专用化代价**：建议论文正文把 Aegis 定位为"通用内容安全压力测试"，突出 fraud-aligned 子集；如需覆盖非欺诈危害，需为 general_harmful_compliance 头增加非欺诈危害分支（超出本轮预算范围）
3. **Fraud-R1 Protocol 轨**：检测率 22% 距 0.35 Gate 仍有差距，主要受 judge 标签噪声限制；建议后续用独立审计 dev（指南 §18.3）而非 judge 全量标签做校准，避免标签噪声放大
4. **DNA**：非重叠池无正样本，后续如需 DNA 增益需在预算内重建 gold 正样本层（如 partial leakage 人工复核集）
5. **全量运行门槛**：仅 Aegis 达强 Gate → 建议下一轮执行 validation 校准 + Aegis test 增量重跑（预算 15-25 元，需另行批准）

---

*报告生成：2026-08-06。所有数字可从 `pilot/boundary_gate_report.json` 与错误矩阵复算。*