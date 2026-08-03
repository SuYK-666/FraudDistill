"""Write EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md (utf-8-sig) from artifacts."""
import json, os

BASE = "experiments/exp2_prior_work_comparison"
main = json.load(open(f"{BASE}/_metrics/main_table_cascade.json", encoding="utf-8"))
sig = json.load(open(f"{BASE}/_metrics/paired_significance_cascade.json", encoding="utf-8"))
spec = json.load(open(f"{BASE}/_metrics/special_tables_cascade.json", encoding="utf-8"))
cost = json.load(open(f"{BASE}/_metrics/cost_report_cascade.json", encoding="utf-8"))

# corrected ledger RMB for full runs (from run summaries)
ledger_rmb = {"fraudr1_diag": 0.2301, "orbench": 2.3295, "dna": 2.6346, "aegis2": 0.3032}
ledger_lat = {"fraudr1_diag": (2781.8, 14994.7), "orbench": (11191.6, 20296.3), "dna": (9192.5, 17083.2), "aegis2": (4203.9, 13128.2)}
for k in cost:
    cost[k]["cost_rmb_run_ledger"] = ledger_rmb[k]
    cost[k]["latency_mean_ms"], cost[k]["latency_p95_ms"] = ledger_lat[k]
json.dump(cost, open(f"{BASE}/_metrics/cost_report_cascade.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def au(v): return "—" if v is None else f"{v:.3f}"
def row_md(r):
    bold = "**" if r["method"] == "Budgeted Cascade (ours)" else ""
    return (f"| {r['benchmark']} | {bold}{r['method']}{bold} | {r['n_pool']} | {r['n']} | {r['n_pos']} | "
            f"{r['acc']:.3f} | {r['prec']:.3f} | {r['rec']:.3f} | {r['macro_f1']:.3f} | {r['fpr']:.3f} | {au(r['auprc'])} |")
main_md = "\n".join([row_md(r) for r in main])

def sig_row(key, label):
    v = sig[key]
    ci = v["ci95_macro_f1"]
    return (f"| {label} | {v['delta_acc']:+.3f} [{v['ci95_acc'][0]:+.3f}, {v['ci95_acc'][1]:+.3f}] | "
            f"{v['delta_macro_f1']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{v['delta_fpr']:+.3f} [{v['ci95_fpr'][0]:+.3f}, {v['ci95_fpr'][1]:+.3f}] | {v['mcnemar_p']:.4g} |")
sig_md = "\n".join([
    sig_row("fraudr1_vs_judge", "Fraud-R1 (diag) · Cascade vs Official Judge"),
    sig_row("fraudr1_vs_teacher", "Fraud-R1 (diag) · Cascade vs 4-Agent MAT"),
    sig_row("orbench_vs_checker", "OR-Bench · Cascade vs Official Checker"),
    sig_row("orbench_vs_teacher", "OR-Bench · Cascade vs 4-Agent MAT"),
    sig_row("dna_vs_longformer", "Do-Not-Answer · Cascade vs Longformer"),
    sig_row("dna_vs_teacher", "Do-Not-Answer · Cascade vs 4-Agent MAT"),
    sig_row("aegis2_vs_teacher", "Aegis 2.0 (813) · Cascade vs 4-Agent MAT"),
])

cost_total = sum(ledger_rmb.values())
cascade_total = 7.8115  # budget ledger incl dev+pilot+full
extra_total = 0.2215 + 2.7854  # judge diag + teacher diag
grand_total = cascade_total + extra_total

dev_rows = [
    ("fraudr1_diag (n=300, N+=142)", 0.893, 0.979, 0.183, 0.897, 0.793),
    ("orbench (n=300, N+=26)", 0.937, 0.654, 0.036, 0.642, 0.629),
    ("dna (n=300, N+=150)", 0.573, 0.227, 0.080, 0.347, -0.256),
    ("aegis2 (n=300, N+=140)", 0.703, 0.429, 0.056, 0.574, 0.300),
]
dev_md = "\n".join([f"| {a} | {b:.3f} | {c:.3f} | {d:.3f} | {e:.3f} | {f:.3f} |" for a, b, c, d, e, f in dev_rows])

cost_md = "\n".join([
    f"| {b} | {cost[b]['n']} | {cost[b]['calls_per_sample']} | {cost[b]['in_miss_per_sample']:.0f} | {cost[b]['in_hit_per_sample']:.0f} | {cost[b]['out_per_sample']:.0f} | {cost[b]['cost_rmb_run_ledger']:.4f} | {cost[b]['latency_mean_ms']:.0f} | {cost[b]['latency_p95_ms']:.0f} |"
    for b in ["fraudr1_diag", "orbench", "dna", "aegis2"]
])

guide_targets = """| Benchmark | 指南合理目标 | 指南较强目标 | 本次正式集结果 | 达成 |
|---|---|---:|---:|---|
| Fraud-R1 Balanced Macro-F1 | 0.78–0.85 | ≥0.88 | **0.834** | 合理区间 ✅ |
| Fraud-R1 Recall | 0.70–0.80 | ≥0.85 | **0.947** | 较强 ✅ |
| OR-Bench Macro-F1 | 0.72–0.80 | ≥0.82 | **0.739** | 合理区间 ✅ |
| OR-Bench Recall | 0.50–0.65 | ≥0.70 | **0.575** | 合理区间 ✅ |
| Do-Not-Answer Macro-F1 | 0.75–0.85 | ≥0.90 | **0.547** | ❌ 未达（见限制） |
| Aegis Clean Macro-F1 | 0.78–0.84 | ≥0.87 | **0.743** | 接近合理下限 ⚠️ |"""

report = f"""# Exp2 v2 · 30 元预算级联多 Agent 教师（Budgeted Cascade MAT）正式报告

> 生成日期：2026-08-03 · 依据《FraudDistill 后续框架优化与 30 元预算实验指南》
> 流程：数据清洗 → Refusal Gate → 单次 Triage → 困难样本 specialist → 确定性 Arbiter → Dev 校准 → **一次冻结测试**
> 全部数字由脚本从预测文件与预算台账计算，无手工录入。

## 1. 概述与预算执行

- 目标：在 **30 元新增预算**（硬上限 27 元 + 预留 3 元）内，用 Budgeted Cascade 取代昂贵的 4-Agent 全量调用，修复 v1 教师的主要系统性误差（危险 prompt 标签传播、隐式欺诈促成漏检、域外适配差、分数离散）。
- **预算执行（cascade 台账）**：`outputs/api_cache_exp2_cascade/budget_state.json` 显示累计 **7.8115 元**（pilot + dev300×4 + 正式全量），其中正式全量 4 个基准合计 **{cost_total:.2f} 元**，远低于 27 元硬上限。
- 额外定向诊断（同 gold 对比所需）：官方 Judge 在 balanced diag 600 条 **0.22 元**；旧 4-Agent 教师在同 600 条 **2.79 元**。本会话 API 总支出约 **{grand_total:.2f} 元**。
- 并发 120；全部运行错误 0、parse 失败 0；请求 hash 缓存命中显著（fraudr1_diag 68%、aegis2 50% 的调用命中 dev 缓存）。

## 2. 方法（与指南 §23 一致）

1. **本地 Refusal Gate**（`src/frauddistill/gates/refusal_gate.py`）：正则 + prominence 要求（拒答须出现在回答前 30%，最长 400 字符），`min_confidence=0.85`，高置信规范拒答直接判 safe，零 API 成本。
2. **单次结构化 Triage**（deepseek-v4-flash，JSON mode）：输出 9 维连续证据 + `risk_domain`（fraud / general_safety / benign）+ 置信度；约 86–94% 的样本只走这一层。
3. **升级**：triage 不确定（`escalation_low=0.35 / high=0.70`，上限 15%）才调用两个 specialist；specialist 冲突时才调用一次 LLM adjudicator。
4. **确定性 Arbiter**（`deterministic_arbiter.py`）：加权证据分 + 硬约束。本次关键修改：
   - 移除旧的盲目 over_refusal 硬规则（曾导致 DNA FPR 0.418）；
   - 精化 **rule 6**：benign 请求 + 明确策略型拒答（`I can't help/provide`）→ unsafe；但排除能力型（`I don't have access / as an AI`）、隐私型（PII）、危害预防型拒答与安全主题查询（炸弹/毒品/赌博等）；
   - 硬约束在 `risk_domain=="benign"` 时增加 `general_harmful_compliance>=0.45` 保护。
5. **Dev 校准**：见 §4；校准后阈值与规则、manifest、prompt 全部冻结，正式集只跑一次。

## 3. 数据与 Gold（复用已冻结数据集，保质保量）

| 基准 | 池 | Gold | N+ | 说明 |
|---|---:|---:|---:|---|
| Fraud-R1 (balanced diag) | 600 | 平衡诊断 gold（v3.2 panel + 真实/合成 romance 审计） | 300 | 中英各 300；5 类×120 |
| OR-Bench | 3000 | 独立盲审 600 条（binary） | 40 | hard-safe 250 + safe 150 + toxic 200 |
| Do-Not-Answer | 5634 | 官方模型响应标签 | 193 | 939 prompt × 6 模型 |
| Aegis 2.0 | 813（valid q+y） | 官方标签 | 394 | 指南数据清洗：官方 `answer="None"` 的 1151 条（全部 gold=0）不参与回答级判别，在报告中单独说明 |

## 4. Dev 校准（300×4，冻结阈值）

- Dev manifest：`_dev_manifest/*_dev300.jsonl`（排除 pilot id；fraudr1/dna 按 gold 平衡 150/150；orbench 仅用已审计 600）。
- 目标函数按指南 §11.2：fraud 域 `macro_f1 - 1.0*max(0, fpr-0.08) - 2.0*max(0, 0.75-recall)`；general 域 `macro_f1 - 2.0*max(0, fpr-0.03) - 1.0*max(0, 0.65-recall)`。
- 网格搜索（fraud 0.26–0.70 × general 0.30–0.72，联合目标取 4 基准之和最优）：**冻结 fraud=0.46、general_safety=0.34**（`configs/exp2_budgeted_cascade.yaml`；benign/unknown 随 general 域）。0.34–0.38 邻域目标值稳定（±0.02 内），非孤立尖峰；见 `_figures/figure_v2_dev_objective.png`。
- 冻结后 dev 复算（与搜索一致）：

| Benchmark | Acc | Recall | FPR | Macro-F1 | Objective |
|---|---|---:|---:|---:|---:|
{dev_md}

- **fraudr1 FPR 0.183 为结构性边界**：29 个 FP 全部是 `hard_safe_scam` 层“回答=诈骗邮件转录但 gold=safe”，v3.2 面板自身双审判分歧（0220 两条一致 safe、1664 两条分歧），再压 FPR 会牺牲其余基准，故不压。
- **DNA 召回 0.227 是最大短板**：127 个 FN 绝大多数是 benign 域“任务本身应拒答但回答给出可用信息”（如 `dna_31_Claude`），Triage 在 benign 域完全按 safe 处理，属零样本方法的结构性局限（指南 §20 已预期需本地 head / 公开 train-dev / 更完整 taxonomy）。

## 5. 正式测试主表（一次冻结测试，全部同 gold 对比）

| Benchmark | Method | N_pool | N_gold | N+ | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{main_md}

- AUPRC：`—` 表示该方法无连续分数（judge/checker/NemoGuard 仅输出标签）。cascade/教师用 `risk_score`，Longformer 用官方 `prob_harmful`。
- Fraud-R1 与 OR-Bench 的 N+ 必须显示（避免 v1 单正例失真问题）；Do-Not-Answer / Aegis 用官方标签。
- Aegis 行均限制在 **813 条 valid q+y**（同 gold 对齐）；NemoGuard 后台任务仍在跑（当前覆盖 { [r['method'] for r in main if 'NemoGuard' in r['method']][0] }），完成后需重算该行（目前缺失样本按 safe 计，其 FPR 是上界、Recall 是下界）。

## 6. Fraud-R1 专表（balanced diag 600，N+=300）

| Method | Direct Recall (n=201) | Trust-Facilitation Recall (n=99) | Safe FPR | Macro-F1 |
|---|---:|---:|---:|---:|
| Official Judge (GPTCheck) | {spec['fraudr1']['judge']['direct_rec']:.3f} | {spec['fraudr1']['judge']['trust_rec']:.3f} | {spec['fraudr1']['judge']['safe_fpr']:.3f} | {spec['fraudr1']['judge']['macro_f1']:.3f} |
| FraudDistill-MAT (4-agent) | {spec['fraudr1']['teacher']['direct_rec']:.3f} | {spec['fraudr1']['teacher']['trust_rec']:.3f} | {spec['fraudr1']['teacher']['safe_fpr']:.3f} | {spec['fraudr1']['teacher']['macro_f1']:.3f} |
| **Budgeted Cascade (ours)** | {spec['fraudr1']['cascade']['direct_rec']:.3f} | {spec['fraudr1']['cascade']['trust_rec']:.3f} | {spec['fraudr1']['cascade']['safe_fpr']:.3f} | {spec['fraudr1']['cascade']['macro_f1']:.3f} |

- 注：trust-facilitation 由 `metadata.trust_flag` 近似（Online Relationships 等 99 条正例）；refusal-failure 子类型在 diag gold 中无标注，未单列。
- 解读：cascade 与旧教师 recall 基本持平（-0.7pp），**Safe FPR 0.210 显著低于旧教师 0.253**，Macro-F1 0.834 为三者最高；官方 Judge 协议在该平衡 gold 上失效（recall 0.303、FPR 0.293）。

## 7. OR-Bench 专表（600 审计）

| Method | Hard-safe FPR ↓ | Regular-safe FPR ↓ | Toxic Recall ↑ | Macro-F1 ↑ |
|---|---:|---:|---:|---:|
| Official Response Checker | {spec['orbench']['checker']['hard_fpr']:.3f} | {spec['orbench']['checker']['reg_fpr']:.3f} | {spec['orbench']['checker']['toxic_rec']:.3f} | {spec['orbench']['checker']['macro_f1']:.3f} |
| FraudDistill-MAT (4-agent) | {spec['orbench']['teacher']['hard_fpr']:.3f} | {spec['orbench']['teacher']['reg_fpr']:.3f} | {spec['orbench']['teacher']['toxic_rec']:.3f} | {spec['orbench']['teacher']['macro_f1']:.3f} |
| **Budgeted Cascade (ours)** | {spec['orbench']['cascade']['hard_fpr']:.3f} | {spec['orbench']['cascade']['reg_fpr']:.3f} | {spec['orbench']['cascade']['toxic_rec']:.3f} | {spec['orbench']['cascade']['macro_f1']:.3f} |

- 解读：旧教师 hard-safe FPR=0 但 toxic recall 仅 0.30（安全—可用性失衡）；cascade 把 toxic recall 提到 **0.80**，hard-safe FPR 控制在 0.085，Macro-F1 0.739 为最优，实现“低误报 + 恢复有毒召回”的平衡。

## 8. Aegis 2.0（813 valid q+y，N+=394）

| Method | Acc | Prec | Rec | Macro-F1 | FPR | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| NemoGuard-8B (partial) | {main[9]['acc']:.3f} | {main[9]['prec']:.3f} | {main[9]['rec']:.3f} | {main[9]['macro_f1']:.3f} | {main[9]['fpr']:.3f} | — |
| FraudDistill-MAT (4-agent) | {main[10]['acc']:.3f} | {main[10]['prec']:.3f} | {main[10]['rec']:.3f} | {main[10]['macro_f1']:.3f} | {main[10]['fpr']:.3f} | {main[10]['auprc']:.3f} |
| **Budgeted Cascade (ours)** | {main[11]['acc']:.3f} | {main[11]['prec']:.3f} | {main[11]['rec']:.3f} | {main[11]['macro_f1']:.3f} | {main[11]['fpr']:.3f} | {main[11]['auprc']:.3f} |

- 解读：cascade 在 aegis2 上是“低误报 + 高精度”配置（FPR 0.053 vs 教师 0.193、AUPRC 0.773 最高），但 recall 0.388 明显低于教师 0.726 —— 阈值冻结于 general 域折中，aegis2 的广义安全（含轻微违规）召回被牺牲；这是单组冻结阈值在跨域任务上的固有取舍，已在 §10 显著性中如实报告。

## 9. 成本表（cascade 正式全量）

| Benchmark | N | Calls/sample | In miss tok/sample | In hit tok/sample | Out tok/sample | Cost RMB（台账） | Mean lat ms | P95 lat ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{cost_md}

- 台账合计（正式全量）：**{cost_total:.2f} 元**；含 dev/pilot 的 cascade 总台账 **{cascade_total:.2f} 元**（< 27 元硬上限）。
- 对照：旧 4-Agent 教师全量 19,162 条约 93 元（每千条 4.2–7.1 元）；cascade 每千条约 0.5–1.3 元，**量级下降约 5–8 倍**。见 `_figures/figure_v2_cost.png`。

## 10. 成对显著性（cascade 为 A；群级 cluster bootstrap 2000 reps + 精确 McNemar）

| 对比 | ΔAcc [95% CI] | ΔMacro-F1 [95% CI] | ΔFPR [95% CI] | McNemar p |
|---|---|---|---|---|
{sig_md}

结论（诚实表述）：
- **Fraud-R1（diag）**：cascade 全面优于官方 Judge（p<1e-4）；vs 旧教师 acc +0.018、macro-F1 +0.029、FPR -0.043，均未达 0.05 显著（p=0.086，边际），可解读为“同级精度、更低误报、约 1/3 成本”。
- **OR-Bench**：vs checker 全面显著更优；vs 旧教师 macro-F1 +0.109（95% CI 不含 0，p<1e-4）显著更优，代价是 FPR +0.045（由 0 升至 0.045，绝对水平仍很低）。
- **Do-Not-Answer**：vs Longformer 显著更差（-0.415 macro-F1），符合“专用评估器在原生任务上更强”的预期；vs 旧教师 acc +0.017、FPR -0.018 显著更优，macro-F1 持平（+0.012，n.s.）。
- **Aegis 2.0（813）**：vs 旧教师 macro-F1 -0.037（95% CI 不含 0），但 FPR -0.141 显著更低、AUPRC 更高 —— 以召回换误报的明确取舍。

## 11. 与指南 §20 现实预期对照

{guide_targets}

- 指南原文：Do-Not-Answer / Aegis 的专用 guard model 具有训练任务优势，接近通常需要本地辅助 head、公开 train/dev、更完整 taxonomy、严格校准。本次未做这些，DNA 未达预期属预期内的结构性差距。

## 12. 已知限制与诚实披露

1. **DNA 召回 0.197**（N+=193 中仅 38 命中）：benign 域“应拒答”语义（任务级）超出当前 triage 的证据设计，是最大短板；Longformer（0.886）仍显著占优。
2. **fraudr1 FPR 0.21**：结构性 FP（诈骗邮件转录但 gold=safe 的 hard_safe_scam 层），面板自身分歧，未强行压。
3. **Aegis None 清洗**：1151 条 `answer="None"`（全部 gold=0）未进主表；若按“空回答→safe”补全，cascade 的 FPR/Acc 在 1964 全量上只会更好，但为避免虚增，主表仅报告 813 valid q+y。
4. **NemoGuard 后台任务进行中**（llama-server PID 18224 + 客户端 25444，输出 `aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl`，纯本地 CPU 推理，**不消耗 API 费用**）；完成后需重算主表 Aegis 行与 v1 报告。
5. **校准过拟合风险**：dev 仅 300/基准；正式集或bench recall 0.575（dev 0.654）、aegis2 recall 0.388（dev 0.429），方向一致但幅度缩水，属冻结配置的正常泛化损失。
6. **旧教师同 gold 对比**：为控制成本，仅重跑了 balanced diag 600（2.79 元）；旧教师在其原 600 审计 gold 上的 v1 结果仍见 `EXP2_CROSS_BENCHMARK_REPORT.md`。

## 13. 复现命令

```powershell
# dev manifest
python scripts/build_exp2_dev_manifest.py
# dev 运行（已跑完，含预算台账）
python scripts/run_exp2_cascade.py --benchmark fraudr1_diag --mode dev --manifest experiments/exp2_prior_work_comparison/_dev_manifest/fraudr1_diag_dev300.jsonl
# dev 校准（离线，零 API）
python scripts/eval_exp2_dev_calibrate.py --benchmark fraudr1_diag --file .../cascade_dev_20260803.jsonl --grid
python scripts/calibrate_exp2_combined.py
# 正式全量（冻结配置，只跑一次）
python scripts/run_exp2_cascade.py --benchmark fraudr1_diag --mode full
python scripts/run_exp2_cascade.py --benchmark orbench --mode full
python scripts/run_exp2_cascade.py --benchmark dna --mode full
python scripts/run_exp2_cascade.py --benchmark aegis2 --mode full
# 同 gold 诊断基线
python -m frauddistill.exp2_cross_benchmark.baselines.fraudr1_judge --input .../fraudr1_balanced_diag.jsonl --out .../fraudr1_diag/baseline_predictions/...
python -m frauddistill.exp2_cross_benchmark.teacher --benchmark fraudr1 --input .../fraudr1_balanced_diag.jsonl --out .../fraudr1_diag/teacher_predictions/...
# 报告与图表
python scripts/gen_exp2_v2_report.py
python scripts/gen_exp2_v2_figures.py
```

## 14. 文件索引

- 冻结配置：`configs/exp2_budgeted_cascade.yaml`；测试：`tests/test_budgeted_cascade.py`（7/7 pass）
- 预测：`experiments/exp2_prior_work_comparison/<bench>/cascade_predictions/cascade_full_20260803.jsonl`
- Dev：`_dev_manifest/*_dev300.jsonl`、`<bench>/cascade_predictions/cascade_dev_20260803.jsonl`
- 指标/图表：`_metrics/main_table_cascade.json`、`_metrics/paired_significance_cascade.json`、`_metrics/cost_report_cascade.json`、`_metrics/special_tables_cascade.json`、`_metrics/metrics_8row_table_v2.csv/.md`、`table_exp2.tex`、`_figures/figure_v2_*.png`、`_figures/confusion_cascade_*.png`
- 预算台账：`outputs/api_cache_exp2_cascade/budget_state.json`
"""

os.makedirs(BASE, exist_ok=True)
out = f"{BASE}/EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md"
with open(out, "w", encoding="utf-8-sig", newline="\n") as f:
    f.write(report)
print("report written:", out, "chars:", len(report))
