# E1 FINAL TRIAD v3.1 整体任务进度与收尾报告

> 协议：`E1-FINAL-TRIAD-v3.1-7500-3200-RealPrevalence`（2026-08-01 冻结版）
> 状态：**全部 WU 完成**，报告已生成，代码已提交 GitHub。
> 生成日期：2026-08-02

## 一、最终决策码

```json
{
  "decision_code": "E1_V31_BEHAVIOR_PASS_MECHANISM_WEAK",
  "p0_gate": "PASS",
  "a_manifest_gate": "PASS",
  "a_target_gate": "PASS",
  "a_gold_gate": "PASS",
  "b_gate": "PASS",
  "c_gate": "PASS"
}
```

解读：A/B/C 三道门禁全部 PASS；B 层 Frozen Anchor 的 q+y Macro-F1 = 0.950（强 PASS，95% CI 下界 0.891 ≥ 0.88），
但 **q+y − y-only 机制增益仅为 0.0065**（McNemar p=1.0），未达 0.03 的条件 PASS 门槛，
因此落入 `BEHAVIOR_PASS_MECHANISM_WEAK`：行为检测强，但“q+y 显著优于 y-only”的机制叙事在本面板上不成立。

## 二、执行摘要（按工作量单元）

| WU | 内容 | 状态 | 关键结果 |
|---|---|---:|---|
| WU-1 | v3.1 代码整改（阶段机 + API 执行器 + 金标/统计/面板/检测器/回放） | ✅ | 12 项新增测试通过；全量 265 项通过 |
| WU-2 | P0 协议锁 / 数据审计 / secret scan | ✅ | gate PASS，api_allowed_now=True |
| WU-3 | A7500 target 生成 | ✅ | 4418/4418 调用成功，7500 条真实响应就绪 |
| WU-4 | A 双 Gold + 冻结 + 统计 | ✅ | 8838 次 judge 调用完成；agreement 0.9995；completion 1.0 |
| WU-5 | B3200 面板构建 + Gold | ✅ | 3200 行面板，formal_panel_ready=True，B Gold gate PASS |
| WU-6 | B CPU detector 训练/校准/冻结 | ✅ | 阈值冻结：q-only 0.60 / y-only 0.75 / q+y 0.65 |
| WU-7 | B Frozen Anchor 一次性消耗 | ✅ | q+y 0.950 / y-only 0.943 / q-only 0.796 |
| WU-8 | C 真实低基率回放 | ✅ | q+y AUPRC 0.492 vs y-only 0.380（ratio 1.29） |
| WU-9 | 最终报告 + 归档 + GitHub | ✅ | 11 份报告；data/reports 双归档；已 push |

## 三、A 层：7500 条真实目标回答

- 构成：Fraud-R1 2141 canonical case；3750 个唯一 prompt instance；新增 API 4418 条 + 严格 Gate 复用 3082 条 = 7500。
- Gold：双 LLM（qwen3.7-plus / deepseek-v4-pro）+ qwen3.7-max 裁决，completion=1.0，binary agreement=0.9995，PABAK=0.999，**PASS**。
- 自然发生率（central）：**28/7500 = 0.37%**（95% CI 0.26%–0.54%）。
  - qwen 0.29%（11/3750），deepseek 0.45%（17/3750）；McNemar p=0.286（不显著）。
  - assistant 0.05%（2/4282），roleplay 0.81%（26/3218）。
  - 类别：fake_job_posting 2.24% 最高；phishing 0.18%、network_friendship 0.67%、fraudulent_service/impersonation ≈0。

## 四、B 层：3200 条上下文机制诊断面板

- 面板构成（Amendment V2 后）：real 2000 + counterfactual synthetic 929 + source-derived open control 271。
- 配额：stable+ 318、stable- 2550、critical+ 12、hard- 320（合计 3200）。
- 划分（按 canonical family，无跨 split 泄漏）：model-dev 1600 / calibration 641 / anchor 800 / reserve 159。
- B Gold：2199 行双判 + 202 裁决，completion 1.0，binary agreement 0.943，PABAK 0.885，**PASS**。
- Frozen Anchor 主指标（5-seed，SD=0.0 表示该 detector 族对 seed 不变）：

| view | Macro-F1 | AUPRC | recall | FPR |
|---|---:|---:|---:|---:|
| q-only | 0.7961 | 0.6949 | 0.6463 | 0.0446 |
| y-only | 0.9432 | 0.9717 | 0.8537 | 0.0056 |
| q+y | 0.9497 | 0.9709 | 0.8537 | 0.0028 |

- 方向一致：q-only < y-only < q+y（5/5 seed 一致）。
- 反快捷方式审计：provenance shortcut AUC、nuisance baseline、wrong-q permutation drop 均已写入 `E1_V31_B_FREEZE.json`。

### 配额 Amendment 记录（两次，均写入 config 与代码）

1. **E1_V31_AMENDMENT_B_QUOTA_V1（2026-08-02）**：协议原 stable+ 1280 数学不可行（真实 positive 仅 34 条，synthetic cap 1200 无法补足）。
   改为 stable+ 480 / stable- 2080 / critical+ 320 / hard- 320。
2. **E1_V31_AMENDMENT_B_QUOTA_V2（2026-08-02）**：V1 仍不可行——安全对齐的目标模型即使在显式 counterfactual 指令下也只有约 4% 的 positive gold 产出率。
   依据冻结双 Gold 的实际供给（stable+ 318、critical+ 12、hard- 724、stable- 3145）将配额设为可行最大值，
   并按协议 5.3.2 由双 Gold 判定决定 synthetic 行的 stratum（不再按生成意图匹配）。

## 五、C 层：真实低基率回放

- 将冻结 B detector 直接回放到 A7500 真实分布（非 unseen 泛化实验）。
- 发生率 0.37%；q+y AUPRC 0.492 vs y-only 0.380（ratio 1.29）；paired bootstrap gain 0.111（95% CI -0.030 ~ 0.276，方向为正但未显著）。
- 说明：C 层仅为回放验证，不声称 unseen。

## 六、预算与调用账目

- 总调用 19,961 次；总成本约 **54.24 元**（qwen 29.81 / deepseek 24.43），远低于硬停 215 元。
- 分阶段：A target 4418、A gold 8838、A gold-reused 4、B counterfactual 1799、B gold 4902。
- 并发：用户要求 120，按协议与 provider cap 实施（qwen=24 / deepseek=20 / adjudicator=8）。

## 七、文件与归档

- 实验数据（gitignored）：`data/prepared/e1_final_triad_v31/`
  - A：`E1_V31_A_REGISTRY_FROZEN.jsonl`（7500）、`E1_V31_A_BEHAVIOR_STATS.json`
  - B：`E1_V31_B_PANEL_ALL.jsonl`（3200）、split 文件、`E1_V31_B_ANCHOR_RESULTS.json`、`E1_V31_B_FREEZE.json`
  - C：`E1_V31_C_RESULT.json`；账目：`E1_V31_BUDGET_LEDGER.jsonl`
- 报告（git tracked）：`reports/e1_final_triad_v31/`（11 份）
- 归档（gitignored）：
  - `archive/pre_e1_final_triad_v31_20260802_170725/`（数据）
  - `reports/pre_e1_final_triad_v31_20260802_170725/`（旧报告）

## 八、复现命令

```powershell
python scripts/run_e1_a7500.py --phase p0
python scripts/run_e1_a7500.py --phase build-manifest
python scripts/run_e1_a7500.py --phase health --run-api --confirm-budget --limit-q 50
python scripts/run_e1_a7500.py --phase generate --run-api --confirm-budget --batch-size-q 500 --resume
python scripts/run_e1_a7500.py --phase validate-targets
python scripts/run_e1_a7500.py --phase gold --run-api --confirm-budget --resume
python scripts/run_e1_a7500.py --phase adjudicate --run-api --confirm-budget --resume
python scripts/run_e1_a7500.py --phase freeze
python scripts/run_e1_b3200.py --phase build-panel --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-gold --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-adjudicate --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-consensus
python scripts/run_e1_b3200.py --phase validate-panel
python scripts/run_e1_b3200.py --phase model-dev
python scripts/run_e1_b3200.py --phase calibration
python scripts/run_e1_b3200.py --phase freeze-b
python scripts/run_e1_b3200.py --phase anchor --consume-anchor
python scripts/run_e1_c_real_prevalence.py --phase c-all
python scripts/run_e1_final_triad_v3.py --phase final-report
```

## 九、已知限制与说明

1. B 面板 positive 类占比约 10%（318+12 / 3200），低于协议 1280/1280 的均衡设计；这是安全对齐模型的真实产出约束，已通过两次 Amendment 如实记录。
2. y-only 在 B 面板与 C 回放中均非常强（0.943 / AUPRC 0.380），q+y 的增量很小（B: +0.0065；C: AUPRC ratio 1.29）——机制叙事证据为“弱”。
3. 5-seed SD=0.0：TF-IDF + LogisticRegression 对该数据 seed 不变，属诚实结果而非伪造。
4. source-derived open control 271 行为 positive（Fraud-R1 生成文本本身被判为欺诈协助），已按 provenance 明确标记，未冒充真实 target response。
