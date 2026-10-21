# Table 9 Reproduction: Reference-Audit Simulation

> **Scientific label:** this table reproduces the historical audit-budget simulation by replacing audited rows with E6 Silver/reference binary labels. It is an oracle upper-bound analysis and is **not** an external-API audit result. The real DeepSeek v2 result is kept separately.

| 运行配置 | Macro-F1 | Recall | FPR | ΔF1 vs A2 | 修正错误 | 引入错误 | 净纠错 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 校准后自动判定 A2 | 0.7266 | 0.7292 | 0.2760 | 0.0000 | 0 | 0 | 0 |
| 参考审计5%，19条 | 0.7526 | 0.7604 | 0.2552 | +0.0260 | 10 | 0 | +10 |
| 参考审计10%，38条 | **0.7760** | **0.7865** | **0.2344** | +0.0495 | 19 | 0 | +19 |
| 参考审计15%，57条 | 0.7891 | 0.7969 | 0.2188 | +0.0625 | 24 | 0 | +24 |
| 参考审计20%，76条 | **0.8046** | **0.8229** | **0.2135** | **+0.0781** | 30 | 0 | +30 |

Historical K=77 reproduction for the old 20% row: **F1 0.8072**, Recall 0.8281, FPR 0.2135.

Historical K=77 bootstrap vs A2: F1 delta +0.0810, 95% CI [0.0537, 0.1094], significant.

## 10,000 family bootstrap vs A2

| Audit budget | F1 delta | 95% CI | Significant |
|---|---:|---|---|
| 5% / 19 | +0.0261 | [0.0122, 0.0423] | yes |
| 10% / 38 | +0.0496 | [0.0284, 0.0726] | yes |
| 15% / 57 | +0.0628 | [0.0394, 0.0882] | yes |
| 20% / 76 | +0.0783 | [0.0515, 0.1062] | yes |

Adjacent-budget increments:

| Comparison | F1 delta | 95% CI | Significant |
|---|---:|---|---|
| 10% vs 5% | +0.0235 | [0.0084, 0.0407] | yes |
| 15% vs 10% | +0.0132 | [0.0027, 0.0255] | yes |
| 20% vs 15% | +0.0155 | [0.0050, 0.0285] | yes |

## Real external audit counterpart

The frozen real DeepSeek v2 audit result is:

| Policy | F1 | Recall | FPR |
|---|---:|---:|---:|
| auto A2 | 0.7266 | 0.7292 | 0.2760 |
| real audit 5% | 0.7474 | 0.7552 | 0.2604 |
| real audit 10% | 0.7630 | 0.7656 | 0.2396 |
| real audit 15% | 0.7682 | 0.7604 | 0.2240 |
| real audit 20% | 0.7812 | 0.7812 | 0.2188 |

The reference simulation should be reported as an oracle analysis or upper bound, not as the external DeepSeek audit result.
