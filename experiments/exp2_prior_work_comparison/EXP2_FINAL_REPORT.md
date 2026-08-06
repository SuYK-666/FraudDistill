# Exp2 正式全量实验报告（FraudDistill Evidence MAT）

> 生成时间：2026-08-06；指南：`FraudDistill_实验二正式全量实验设置与执行规范.md`
> 冻结 Commit：`fbc8146`（`fbc81462b5fa138b395a02f989a73ec224299c49`）；方法：FraudDistill Evidence MAT；模型 `deepseek-v4-flash`，temperature 0，JSON mode，thinking disabled
> Prompt digest（C2）：`3961580ac262ccd3`；Skill 数量：22；Registry digest：`1ca0c3f87bd88415808c8935090b1cf894e9063dd1daecdebc04c045e7beb5dd`

## 1. 执行摘要

- 主表规模：**12,447 行**（Fraud-R1 Content 3,000 / OR-Bench 3,000 / Do-Not-Answer 5,634 / Aegis response 813）。
- Gold：Fraud-R1 与 OR-Bench 使用独立盲审仲裁后的 adjudicated silver labels（正例分别 61 / 40，正例率 2.0% / 1.3%）；DNA 与 Aegis 使用官方标签。
- 预注册门槛（七项全过）：总体判定 **FAIL**；各基准：Fraud-R1 Content=FAIL；OR-Bench=FAIL；Do-Not-Answer=FAIL；Aegis response=FAIL。
- 教师推理成本：67.84 RMB（硬顶 100.0）；Gold 标注成本单独记账：22.81 RMB。

## 2. 正式数据与 Gold

| 数据集 | 正式 N | N+（Gold） | 正例率 | Gold 类型 | 是否主表 |
|---|---:|---:|---:|---|---|
| Fraud-R1 Content | 3,000 | 61 | 2.03% | adjudicated silver | 是 |
| OR-Bench（hard 1,000 / regular 1,400 / toxic 600） | 3,000 | 40 | 1.33% | adjudicated silver | 是 |
| Do-Not-Answer | 5,634 | 193 | 3.43% | official | 是 |
| Aegis response test | 813 | 394 | 48.5% | official | 是 |
| Aegis validation（校准） | 1,399（641 response-labeled） | — | — | official | 否 |

Fraud-R1 正例候选（Judge∪T6∪既有审计，724 条）全部纳入，负例按五家族/中英/assistant-roleplay/base-levelup 配额补齐；Gold 盲审后实际正例 61 条（2.0%），低于 40% 可接受区间，按指南 §3.3 处理：报告真实 N+、使用 class-balanced bootstrap、主结论优先 Macro-F1 / Recall / FPR / AUPRC / MCC。

盲审协议（冻结）：Annotator A=`deepseek-v4-flash`，B=`deepseek-v4-pro`，分歧由 `deepseek-v4-flash` 仲裁；标注者不可见方法/基线与主表目标。
- Fraud-R1 盲审二元一致率 98.22%（n=2415，Cohen's κ=0.267）
- OR-Bench 盲审二元一致率 99.88%（n=1603，Cohen's κ=0.4995）

## 3. 离线校准（冻结，未见 test Gold）

- Fraud-R1 Content：dev = 既有 dev/pilot（86 行，正例 32），最终 config 重跑；选择 raw 阈值 **0.041**（Recall 0.8125 / FPR 0.0556 / MF1 0.8985 / AUPRC 0.8924）；calibration_gate_not_met = False。
- Aegis response：official validation（641 行，正例 245）；选择 raw 阈值 **0.068**（FPR 0.0303 / MF1 0.8085 / MCC 0.5622 / AUPRC 0.8387）；calibration_gate_not_met = False。
- OR-Bench：沿用冻结组件阈值 0.5（over_refusal / harmful-compliance / partial-leakage），test 不调参。
- Do-Not-Answer：预注册阈值 0.5，test 不调参。

## 4. 主表（八行）

| Benchmark | 方法 | N | N+ | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Fraud-R1 Content | Fraud-R1 Official Judge | 3000 | 61 | 0.7800 | 0.0811 | 0.9508 | 0.4630 | 0.2235 | 0.0941 | 0.2409 | — |
| Fraud-R1 Content | **FraudDistill Evidence MAT** | 3000 | 61 | 0.8527 | 0.1152 | 0.9344 | 0.5280 | 0.1490 | 0.4752 | 0.2986 | **FAIL** |
| OR-Bench | OR-Bench Official Checker | 3000 | 40 | 0.6943 | 0.0280 | 0.6500 | 0.3743 | 0.3051 | 0.0514 | 0.0856 | — |
| OR-Bench | **FraudDistill Evidence MAT** | 3000 | 40 | 0.9867 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0514 | 0.0000 | **FAIL** |
| Do-Not-Answer | LibrAI Longformer-Harmful | 5634 | 193 | 0.9950 | 0.9661 | 0.8860 | 0.9616 | 0.0011 | 0.9657 | 0.9227 | — |
| Do-Not-Answer | **FraudDistill Evidence MAT** | 5634 | 193 | 0.9666 | 0.6667 | 0.0518 | 0.5476 | 0.0009 | 0.1200 | 0.1796 | **FAIL** |
| Aegis response | NemoGuard-8B | 813 | 394 | 0.8819 | 0.8725 | 0.8858 | 0.8787 | 0.1217 | 0.4152 | 0.7638 | — |
| Aegis response | **FraudDistill Evidence MAT** | 813 | 394 | 0.7528 | 0.8814 | 0.5660 | 0.8089 | 0.0716 | 0.7718 | 0.5337 | **FAIL** |

预注册最低门槛：Fraud-R1 Acc≥0.84/P≥0.80/R≥0.80/MF1≥0.82/FPR≤0.15/AUPRC≥0.85/MCC≥0.65；OR Acc≥0.88/0.50/0.65/0.72/≤0.08/≥0.65/≥0.50；DNA Acc≥0.96/0.40/0.40/0.65/≤0.03/≥0.40/≥0.35；Aegis Acc≥0.83/0.80/0.75/0.80/≤0.15/≥0.82/≥0.65。

## 5. 相对比较与统计检验

| Benchmark | ΔAcc | ΔMacro-F1 (95% CI) | ΔFPR | ΔMCC | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Fraud-R1 Content | +0.0727 | +0.0650 [+0.0541, +0.0762] | -0.0745 | +0.0577 | 0.0000 | 0.0000 |
| OR-Bench | +0.2924 | +0.1257 [+0.1117, +0.1395] | -0.3051 | -0.0856 | 0.0000 | 0.0000 |
| Do-Not-Answer | -0.0284 | -0.4140 [-0.4463, -0.3775] | -0.0002 | -0.7431 | 0.0000 | 0.0000 |
| Aegis response | -0.1291 | -0.0698 [-0.0989, -0.0401] | -0.0501 | -0.2301 | 0.0000 | 0.0000 |

方法：10,000 次 paired group bootstrap（Fraud case / OR prompt / DNA prompt / Aegis interaction 为组），exact McNemar，Holm 校正（四个主比较）。
- Fraud-R1 Content class-balanced bootstrap（正类重采样至与负类等量）：Macro-F1 均值 0.8740，95% CI [0.8716, 0.8762]（n+=61，n-=2939）。
- OR-Bench class-balanced bootstrap（正类重采样至与负类等量）：Macro-F1 均值 0.5000，95% CI [0.5000, 0.5000]（n+=40，n-=2960）。

## 6. 门槛判定详情

### Fraud-R1 Content：FAIL

| 指标 | 最低门槛 | 实测 | 判定 |
|---|---:|---:|---|
| accuracy | 0.84 | 0.8527 | PASS |
| precision | 0.8 | 0.1152 | FAIL |
| recall | 0.8 | 0.9344 | PASS |
| macro_f1 | 0.82 | 0.5280 | FAIL |
| fpr | 0.15 | 0.1490 | PASS |
| auprc | 0.85 | 0.4752 | FAIL |
| mcc | 0.65 | 0.2986 | FAIL |

### OR-Bench：FAIL

| 指标 | 最低门槛 | 实测 | 判定 |
|---|---:|---:|---|
| accuracy | 0.88 | 0.9867 | PASS |
| precision | 0.5 | 0.0000 | FAIL |
| recall | 0.65 | 0.0000 | FAIL |
| macro_f1 | 0.72 | 0.5000 | FAIL |
| fpr | 0.08 | 0.0000 | PASS |
| auprc | 0.65 | 0.0514 | FAIL |
| mcc | 0.5 | 0.0000 | FAIL |

### Do-Not-Answer：FAIL

| 指标 | 最低门槛 | 实测 | 判定 |
|---|---:|---:|---|
| accuracy | 0.96 | 0.9666 | PASS |
| precision | 0.4 | 0.6667 | PASS |
| recall | 0.4 | 0.0518 | FAIL |
| macro_f1 | 0.65 | 0.5476 | FAIL |
| fpr | 0.03 | 0.0009 | PASS |
| auprc | 0.4 | 0.1200 | FAIL |
| mcc | 0.35 | 0.1796 | FAIL |

### Aegis response：FAIL

| 指标 | 最低门槛 | 实测 | 判定 |
|---|---:|---:|---|
| accuracy | 0.83 | 0.7528 | FAIL |
| precision | 0.8 | 0.8814 | PASS |
| recall | 0.75 | 0.5660 | FAIL |
| macro_f1 | 0.8 | 0.8089 | PASS |
| fpr | 0.15 | 0.0716 | PASS |
| auprc | 0.82 | 0.7718 | FAIL |
| mcc | 0.65 | 0.5337 | FAIL |

**总体判定：FAIL —— 至少一个主基准未达到预注册最低指标（按指南 §25 写入限制而非更换指标/样本）。**

## 7. 分组分析

### Fraud-R1 五家族（FraudDistill Recall / FPR）

| Family | N | N+ | Baseline Recall | FD Recall | Baseline FPR | FD FPR | ΔMF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fake Job Postings | 606 | 37 | 1.000 | 1.000 | 0.476 | 0.357 | +0.086 |
| Fraudulent Services | 600 | 1 | 1.000 | 1.000 | 0.088 | 0.062 | +0.021 |
| Impersonation | 600 | 9 | 0.889 | 0.889 | 0.122 | 0.081 | +0.053 |
| Online Relationships | 563 | 10 | 0.800 | 0.700 | 0.031 | 0.015 | +0.060 |
| Phishing Scams | 631 | 4 | 1.000 | 1.000 | 0.389 | 0.227 | +0.092 |

### OR-Bench 分层

| Stratum | N | N+ | Baseline FPR | FD FPR |
|---|---:|---:|---:|---:|
| hard_safe | 1000 | 25 | 0.767 | 0.000 |
| regular_safe | 1400 | 5 | 0.108 | 0.000 |
| toxic | 600 | 10 | 0.007 | 0.000 |

### DNA 按 target model（FD Recall / FPR）

| Target model | N | N+ | Baseline Recall | FD Recall | Baseline FPR | FD FPR |
|---|---:|---:|---:|---:|---:|---:|
| ChatGLM2 | 939 | 85 | 0.882 | 0.059 | 0.004 | 0.005 |
| ChatGPT | 939 | 14 | 0.929 | 0.071 | 0.001 | 0.000 |
| Claude | 939 | 16 | 0.688 | 0.000 | 0.000 | 0.000 |
| GPT4 | 939 | 23 | 0.957 | 0.043 | 0.000 | 0.000 |
| llama2-7b-chat | 939 | 3 | 0.667 | 0.000 | 0.002 | 0.000 |
| vicuna-7b | 939 | 52 | 0.923 | 0.058 | 0.000 | 0.001 |

### Aegis 按 hazard category（FD Recall）

| Category | N | N+ | FD Recall | FD FPR |
|---|---:|---:|---:|---:|
|  | 338 | 0 | 0.000 | 0.065 |
| Controlled/Regulated Substances | 19 | 17 | 0.588 | 0.000 |
| Controlled/Regulated Substances, Criminal Planning/Confessions | 8 | 7 | 0.857 | 1.000 |
| Controlled/Regulated Substances, Needs Caution | 1 | 1 | 0.000 | 0.000 |
| Copyright/Trademark/Plagiarism | 1 | 1 | 1.000 | 0.000 |
| Criminal Planning/Confessions | 69 | 62 | 0.677 | 0.000 |
| Criminal Planning/Confessions, Controlled/Regulated Substances | 17 | 14 | 0.786 | 0.000 |
| Criminal Planning/Confessions, Fraud/Deception | 1 | 1 | 1.000 | 0.000 |
| Criminal Planning/Confessions, Guns and Illegal Weapons | 2 | 1 | 0.000 | 0.000 |
| Criminal Planning/Confessions, Guns and Illegal Weapons, Controlled/Regulated Substances | 3 | 2 | 0.500 | 0.000 |
| Criminal Planning/Confessions, Needs Caution | 4 | 3 | 1.000 | 0.000 |
| Criminal Planning/Confessions, PII/Privacy | 5 | 3 | 0.667 | 0.000 |
| Criminal Planning/Confessions, PII/Privacy, Malware | 1 | 1 | 0.000 | 0.000 |
| Criminal Planning/Confessions, Profanity | 1 | 1 | 0.000 | 0.000 |
| Criminal Planning/Confessions, Violence | 8 | 6 | 0.500 | 0.000 |
| Criminal Planning/Confessions, Violence, Controlled/Regulated Substances | 1 | 1 | 0.000 | 0.000 |
| Criminal Planning/Confessions, Violence, Needs Caution, High Risk Gov Decision Making, Guns and Illegal Weapons | 1 | 1 | 1.000 | 0.000 |
| Fraud/Deception, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Fraud/Deception, Manipulation, Hate/Identity Hate | 1 | 1 | 0.000 | 0.000 |
| Guns and Illegal Weapons | 2 | 2 | 0.500 | 0.000 |
| Guns and Illegal Weapons, Controlled/Regulated Substances, Violence, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Guns and Illegal Weapons, Criminal Planning/Confessions | 9 | 9 | 0.778 | 0.000 |
| Guns and Illegal Weapons, Criminal Planning/Confessions, Violence | 2 | 2 | 0.500 | 0.000 |
| Guns and Illegal Weapons, Harassment | 1 | 1 | 1.000 | 0.000 |
| Guns and Illegal Weapons, Needs Caution | 1 | 1 | 0.000 | 0.000 |
| Guns and Illegal Weapons, Needs Caution, Hate/Identity Hate, Harassment, Criminal Planning/Confessions | 1 | 0 | 0.000 | 0.000 |
| Guns and Illegal Weapons, Threat, Sexual, Criminal Planning/Confessions | 1 | 1 | 0.000 | 0.000 |
| Guns and Illegal Weapons, Violence, Criminal Planning/Confessions | 2 | 2 | 0.500 | 0.000 |
| Harassment | 22 | 17 | 0.588 | 0.200 |
| Harassment, Criminal Planning/Confessions | 4 | 2 | 0.500 | 0.000 |
| Harassment, Hate/Identity Hate | 5 | 4 | 0.250 | 0.000 |
| Harassment, Needs Caution | 2 | 1 | 0.000 | 0.000 |
| Harassment, PII/Privacy | 3 | 3 | 0.333 | 0.000 |
| Harassment, Profanity | 1 | 1 | 0.000 | 0.000 |
| Harassment, Profanity, Hate/Identity Hate | 1 | 1 | 0.000 | 0.000 |
| Harassment, Sexual, Hate/Identity Hate, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Harassment, Sexual, Profanity | 1 | 1 | 0.000 | 0.000 |
| Harassment, Violence | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate | 39 | 33 | 0.606 | 0.167 |
| Hate/Identity Hate, Harassment | 7 | 7 | 0.571 | 0.000 |
| Hate/Identity Hate, Harassment, Political/Misinformation/Conspiracy | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate, Harassment, Profanity | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate, Harassment, Violence | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate, Needs Caution | 1 | 0 | 0.000 | 0.000 |
| Hate/Identity Hate, PII/Privacy, Needs Caution | 1 | 0 | 0.000 | 0.000 |
| Hate/Identity Hate, Political/Misinformation/Conspiracy | 2 | 2 | 1.000 | 0.000 |
| Hate/Identity Hate, Profanity | 2 | 2 | 1.000 | 0.000 |
| Hate/Identity Hate, Profanity, Harassment | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate, Sexual, Profanity | 1 | 1 | 0.000 | 0.000 |
| Hate/Identity Hate, Violence, Harassment, Threat, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Hate/Identity Hate, Violence, Sexual | 1 | 1 | 0.000 | 0.000 |
| Needs Caution, Controlled/Regulated Substances | 1 | 0 | 0.000 | 0.000 |
| Needs Caution, Criminal Planning/Confessions | 4 | 2 | 1.000 | 0.500 |
| Needs Caution, Harassment | 2 | 1 | 1.000 | 0.000 |
| Needs Caution, Harassment, Fraud/Deception | 1 | 1 | 1.000 | 0.000 |
| Needs Caution, Hate/Identity Hate | 1 | 1 | 1.000 | 0.000 |
| Needs Caution, Manipulation, Controlled/Regulated Substances | 1 | 1 | 1.000 | 0.000 |
| Needs Caution, Violence | 1 | 0 | 0.000 | 0.000 |
| Other, Fraud/Deception, Criminal Planning/Confessions | 1 | 0 | 0.000 | 0.000 |
| PII/Privacy | 24 | 16 | 0.375 | 0.000 |
| PII/Privacy, Criminal Planning/Confessions | 8 | 6 | 0.500 | 0.000 |
| PII/Privacy, Criminal Planning/Confessions, Malware | 1 | 1 | 1.000 | 0.000 |
| PII/Privacy, Criminal Planning/Confessions, Needs Caution | 1 | 1 | 0.000 | 0.000 |
| PII/Privacy, Criminal Planning/Confessions, Other | 1 | 1 | 1.000 | 0.000 |
| PII/Privacy, Harassment | 4 | 4 | 0.250 | 0.000 |
| PII/Privacy, Needs Caution | 1 | 0 | 0.000 | 0.000 |
| PII/Privacy, Needs Caution, Harassment | 1 | 0 | 0.000 | 0.000 |
| Political/Misinformation/Conspiracy | 2 | 2 | 1.000 | 0.000 |
| Political/Misinformation/Conspiracy, Needs Caution, Violence | 1 | 1 | 1.000 | 0.000 |
| Profanity | 17 | 14 | 0.071 | 0.000 |
| Profanity, Criminal Planning/Confessions | 1 | 1 | 0.000 | 0.000 |
| Profanity, Guns and Illegal Weapons | 1 | 1 | 0.000 | 0.000 |
| Profanity, Harassment | 5 | 5 | 0.400 | 0.000 |
| Profanity, Hate/Identity Hate | 1 | 0 | 0.000 | 0.000 |
| Profanity, Sexual | 2 | 2 | 0.000 | 0.000 |
| Profanity, Violence | 2 | 2 | 1.000 | 0.000 |
| Profanity, Violence, Harassment | 1 | 1 | 0.000 | 0.000 |
| Sexual | 24 | 21 | 0.429 | 0.000 |
| Sexual (minor) | 2 | 2 | 1.000 | 0.000 |
| Sexual (minor), Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Sexual (minor), Sexual | 1 | 1 | 0.000 | 0.000 |
| Sexual, Criminal Planning/Confessions | 4 | 4 | 0.250 | 0.000 |
| Sexual, Criminal Planning/Confessions, Controlled/Regulated Substances | 1 | 1 | 1.000 | 0.000 |
| Sexual, Criminal Planning/Confessions, PII/Privacy | 1 | 1 | 1.000 | 0.000 |
| Sexual, Needs Caution | 2 | 1 | 0.000 | 0.000 |
| Sexual, Profanity | 4 | 4 | 0.500 | 0.000 |
| Sexual, Violence | 1 | 1 | 1.000 | 0.000 |
| Suicide and Self Harm | 8 | 8 | 0.375 | 0.000 |
| Suicide and Self Harm, Controlled/Regulated Substances | 1 | 1 | 0.000 | 0.000 |
| Suicide and Self Harm, Guns and Illegal Weapons | 1 | 1 | 1.000 | 0.000 |
| Suicide and Self Harm, Profanity | 1 | 1 | 0.000 | 0.000 |
| Threat | 1 | 1 | 1.000 | 0.000 |
| Threat, Guns and Illegal Weapons, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Threat, Harassment | 1 | 1 | 0.000 | 0.000 |
| Threat, Violence, Needs Caution | 1 | 0 | 0.000 | 0.000 |
| Violence | 13 | 9 | 0.444 | 0.500 |
| Violence, Criminal Planning/Confessions | 13 | 10 | 0.800 | 0.333 |
| Violence, Criminal Planning/Confessions, Controlled/Regulated Substances, Guns and Illegal Weapons | 1 | 1 | 1.000 | 0.000 |
| Violence, Criminal Planning/Confessions, Guns and Illegal Weapons | 1 | 1 | 1.000 | 0.000 |
| Violence, Guns and Illegal Weapons | 3 | 2 | 0.500 | 0.000 |
| Violence, Guns and Illegal Weapons, Criminal Planning/Confessions | 6 | 5 | 0.600 | 0.000 |
| Violence, Guns and Illegal Weapons, Criminal Planning/Confessions, Controlled/Regulated Substances | 1 | 1 | 0.000 | 0.000 |
| Violence, Guns and Illegal Weapons, Criminal Planning/Confessions, Hate/Identity Hate | 1 | 1 | 0.000 | 0.000 |
| Violence, Harassment | 2 | 2 | 0.000 | 0.000 |
| Violence, Harassment, Controlled/Regulated Substances | 1 | 1 | 0.000 | 0.000 |
| Violence, Harassment, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Violence, Harassment, Criminal Planning/Confessions, Controlled/Regulated Substances | 1 | 1 | 0.000 | 0.000 |
| Violence, Hate/Identity Hate | 1 | 1 | 1.000 | 0.000 |
| Violence, Hate/Identity Hate, Criminal Planning/Confessions, Guns and Illegal Weapons | 1 | 1 | 1.000 | 0.000 |
| Violence, Hate/Identity Hate, Profanity | 1 | 1 | 1.000 | 0.000 |
| Violence, Needs Caution | 7 | 3 | 0.333 | 0.000 |
| Violence, PII/Privacy | 1 | 1 | 0.000 | 0.000 |
| Violence, PII/Privacy, Criminal Planning/Confessions | 1 | 0 | 0.000 | 0.000 |
| Violence, PII/Privacy, Harassment, Criminal Planning/Confessions, Controlled/Regulated Substances | 1 | 1 | 1.000 | 0.000 |
| Violence, Profanity | 1 | 1 | 1.000 | 0.000 |
| Violence, Profanity, Hate/Identity Hate, Harassment | 1 | 0 | 0.000 | 1.000 |
| Violence, Sexual | 3 | 3 | 1.000 | 0.000 |
| Violence, Sexual, Criminal Planning/Confessions | 1 | 1 | 1.000 | 0.000 |
| Violence, Sexual, Harassment, Criminal Planning/Confessions | 1 | 1 | 0.000 | 0.000 |
| Violence, Suicide and Self Harm, Criminal Planning/Confessions | 2 | 2 | 0.500 | 0.000 |
| Violence, Threat, Criminal Planning/Confessions | 1 | 1 | 0.000 | 0.000 |
| Violence, Threat, Profanity | 1 | 0 | 0.000 | 0.000 |
| Violence, Threat, Profanity, Harassment | 1 | 1 | 1.000 | 0.000 |

## 8. 完整性（Technical Gate）

| 检查项 | 要求 | 结果 |
|---|---|---|
| manifest count（四基准） | 100% 覆盖 | 100%（3000 / 3000 / 5634 / 813，见 audit/final_integrity_checks.json） |
| baseline 与 FD sample_id 对齐 | 一致 | 是（baselines/ 按 manifest 对齐，缺失 0） |
| Gold 与预测 qy_hash 一致 | 一致 | 是（qy_hash_mismatch=0） |
| parse failure / abstain | 0 | parse_failed=0，abstain=0 |
| 截断（finish_reason=length / truncated） | 0 | truncated=0 |
| 重复 sample | 0 | duplicate_sample=0 |
| score 有限且 [0,1] | 是 | nonfinite=0，out_of_range=0 |
| 缺失必需字段 / skill trace | 0 | missing_required_field=0，skill_trace_missing=0 |
| 阈值来源可追溯 / test 调参 | 冻结 / false | 是（calibration/*.json 冻结） |


## 9. 成本

- 教师推理（正式运行）：**67.84 RMB**（本轮硬顶 100.0 RMB；指南 soft 85 / hard 100）。
- Gold 标注（独立记账）：**22.81 RMB**（硬顶 120）。
- 既有缓存复用：Fraud/OR/Aegis/DNA 因最终 digest（`3961580ac262ccd3`）与旧 T6 digest 不一致，按指南 §16.3 全部以最终配置重跑；基线预测全部复用。

## 10. 结论与限制

- Fraud-R1 Content 与 OR-Bench 的正式 Gold 正例率显著低于候选池预估（2.0% / 1.3%）：盲审面板对“内容级欺诈协助”采用严格标准，多数 qwen-plus 回答被判为安全拒答/反诈教育/谨慎讨论。按指南 §3.3，两基准均以真实 N+ 报告并采用 class-balanced bootstrap，不称为 balanced benchmark。
- 盲审面板一致性：Fraud κ=0.267、OR κ=0.500，存在标签噪声；报告中已披露 raw agreement 与 κ。
- 若某基准未达门槛，结论按指南 §23 写作限制条件，不更换指标或样本。
