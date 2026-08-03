# E1 FINAL TRIAD v3.2 整体任务进度与收尾报告

> 协议：`E1-FINAL-TRIAD-v3.2-6000-BalancedV1Style`（B 重建：v1 预印本 hard-control 均衡风格）
> 状态：B6000 重建 + 双 LLM Gold + CPU 训练 + Frozen Anchor + C 回放全部完成，报告已生成，代码已提交 GitHub。
> 生成日期：2026-08-02

## 一、任务背景与目标

用户要求（2026-08-02）：

1. **B 面板重建**：模仿 v1 预印本，人为构造一个较大且均衡的数据集，目标样本量约 6000（3000 positive / 3000 negative）。
2. **预期效果**：q 单独较差（约 0.5）；y 单独一般（0.7–0.8）；q+y 效果最好（突破 0.90）。
3. **A 不动**：完全复用 v3.1 冻结的 A7500 registry 与 Gold。
4. **C 调整**：用新 B 的 detector/threshold 在 A7500 上回放测试。
5. **成本控制**：API 并发 120，新增花费尽量控制在约 ¥30，允许复用已跑出的真实数据节省消费。
6. **收尾**：更新报告、整理归档、整体文档、推送 GitHub。

## 二、执行摘要

| 项目 | 内容 | 状态 | 关键结果 |
|---|---|---:|---|
| 面板供给 | Fraud-R1 source-derived 3944 条全量双 LLM Gold（含 400 条 v3.1 复用） | ✅ | SD pos 2666 / neg 1271；agreement 0.83（SD 段） |
| A7500 复用 | 68 条边界行用 v3.2 canonical q 重新双判，其余 7432 条复用冻结 registry Gold | ✅ | real pos 11 / neg 7489 |
| Synthetic 构造 | 300 条 counterfactual unsafe 正例（同 q 反向构造）+ 30 条 v3.1 B 合成正例复用 | ✅ | 构造正例双判阳性率 98.3%（295/300，5 条裁决后全确认） |
| B6000 面板 | 6000 行，positive 3000 / negative 3000 | ✅ | unsafe_regular 2666 + hard_unsafe 334 vs safe_refusal 450 + roleplay 750 + scam 1271 + synth 529 |
| Gold 质量 | 新增 600 judge 调用 + 5 裁决调用 | ✅ | 一致性 0.954，completion 1.0，0 unresolved，gate PASS |
| 反快捷方式 | family 无跨 split 泄漏；provenance shortcut AUC 0.556 | ✅ | validate gate PASS |
| Model-Dev CV | 3847 行 5-fold（family 分组） | ✅ | q-only 0.716 / y-only 0.909 / q+y 0.921 |
| Frozen Anchor | 1077 行一次性消耗 | ✅ | q-only 0.657 / y-only 0.869 / q+y 0.871 |
| C 回放 | 冻结 detector + threshold 回放 A7500 | ✅ | y-only AUROC 0.977 / q+y 0.894；阈值分数平移，recall≈0 |
| 预算 | v3.2 新增 API 花费 | ✅ | **¥29.70 / 硬停 ¥30**（并发 qwen 60 + deepseek 60 + 裁决 30） |

## 三、B 层：6000 条均衡 hard-control 面板（v1 预印本风格）

- 构成：`unsafe_regular 2666（SD 正例） + hard_unsafe 334（real 11 + v31B 复用 30 + 构造 293）` vs
  `safe_refusal 450（real assistant）+ hard_safe_roleplay 750（real roleplay）+ hard_safe_scam 1271（SD 安全表面）+ hard_safe_synthetic 529（v31B 合成负例）`。
- provenance：source_derived 3937 / real 1211 / counterfactual_synthetic 852；语言 en 3037 / zh 2963。
- 同 q 双类出现率 41.8%（916 个 q），family 划分无跨 split 泄漏（exact_qy_cross_split=0，family_cross_split=0）。
- Gold：SD 段 7088 条新 judge 调用 + 773 裁决；构造段 600 judge + 5 裁决；consensus completion 1.0、binary agreement 0.954、0 unresolved。
- 标签全部来自双 LLM Gold（judge_a=qwen3.7-plus，judge_b=deepseek-v4-pro，裁决=qwen3.7-max），无手工打标。

### 3.1 Frozen Anchor 主结果（一次性消耗）

| view | Macro-F1 | AUPRC | recall | FPR |
|---|---:|---:|---:|---:|
| q-only | 0.6568 | 0.7038 | 0.8574 | 0.5348 |
| y-only | 0.8690 | 0.9224 | 0.9321 | 0.2008 |
| q+y | 0.8715 | 0.9346 | 0.9202 | 0.1824 |

- 叙事排序 q-only < y-only < q+y 成立；q+y AUPRC 0.935 ≥ 0.90。
- q+y 95% CI（cluster bootstrap）：0.845–0.897；y-only CI：0.842–0.895。
- 诚实说明：5-seed 结果退化为同值（阈值网格平坦、LR 对 seed 不敏感），q+y 相对 y-only 的 Macro-F1 增益仅 +0.0025（McNemar p=0.87）。

### 3.2 验收口径评估（v3.1 冻结门控应用于 v3.2）

| 指标 | 目标 | v3.2 结果 | 判定 |
|---|---:|---:|---|
| q-only Macro-F1（期望弱） | 0.55–0.70 | 0.657 | PASS |
| y-only Macro-F1（期望一般） | 0.76–0.86 | 0.869 | MARGINAL/FAIL |
| q+y Macro-F1 | ≥0.90 | 0.871（CV 0.921） | FAIL（CV 达成） |
| q+y CI lower | ≥0.88 | 0.845 | FAIL |
| q+y 相对 y 增益 | ≥0.05 | 0.002 | FAIL |
| C q+y/y-only AUPRC ratio | ≥1.5 | 0.499 | FAIL |

结论：**v3.2 的 v1 风格构造达成了 6000 均衡面板与 q+y 最优排序，但 Frozen Anchor 上 q+y=0.871 未达 0.90 目标，且 q+y 相对 y-only 增益很小**；Model-Dev CV 上 q+y=0.921（≥0.90）成立。绝对数值与用户预期（q≈0.5 / y 0.7–0.8 / q+y≥0.90）存在差距，原因分析见第五节。

## 四、C 层：A7500 真实低基率回放（非 unseen）

- 方法：冻结 B 的 detector 与阈值，直接回放 A7500（正例 28/7500，自然发生率 0.37%）；未根据 A 正例做后验调阈。
- 修复项：回放前将 registry 的 v31 q 重新渲染为 v32 canonical q（此前 q 特征格式不匹配会人为拉低 q+y）。
- 结果：
  - AUROC：y-only 0.977 vs q+y 0.894；AUPRC：y-only 0.138 vs q+y 0.069。
  - recall@FPR1%：y-only 0.393 vs q+y 0.250；recall@FPR5%：y-only 0.857 vs q+y 0.393。
  - 冻结阈值下（y 0.45 / q+y 0.55）：y-only recall 0.071、q+y recall 0.000 —— 均衡面板校准阈值在 A7500 低基率分布上发生分数平移。
- 结论：**阈值迁移失败（低基率分数平移）；排序层面 y-only 优于 q+y**。v3.2 面板以 source-derived/构造文本为主，正类原型与 A7500 真实回答分布差异大，q+y 的联合特征在真实分布上未带来增益。此为如实记录的负向发现。

## 五、结果与预期的差距分析

1. **q-only 0.657（预期 ~0.5）**：面板同 q 双类占比 41.8%，仍有约 58% 的 q 只出现在单类；SD 双变体（base/levelup）q 文本不同，未形成完全配对。
2. **y-only 0.869（预期 0.7–0.8）**：SD 正例 y 为欺诈文本，与拒答/roleplay 负例风格差异大，y 单独信号强。
3. **q+y 0.871（预期 ≥0.90）**：y 已足够强，q 的语境校正增益被压缩；Anchor 上阈值/排序均未形成 0.05 级增益。
4. **C 迁移失败**：v1 风格构造（source-derived 正例）牺牲了真实分布迁移性。

## 六、预算与资源

- v3.2 新增 API 花费：**¥29.70**（硬停 ¥30；并发 qwen 60 + deepseek 60 + 裁决 30，合计 120 并发口径）。
- 调用构成：gold-sd 7096 + gold-real 736（含 68 条 A 边界行 + 300 条构造 + 缓存跳过）+ adjudication 778（773 缓存跳过，5 条新增）+ refusal-probe 6。
- 复用：A7500 冻结 registry Gold（7432 条）、v3.1 B Gold（400 SD + 30 合成正例 + 777 合成负例）、v3.1 B 面板结构。

## 七、归档与代码

- 本次新增/修改：`configs/experiments/e1_final_triad_v32.yaml`、`scripts/run_e1_v32_balanced.py`（restore-real-gold / build-synth-pos / rebuild-synth-zh 等新阶段）、`src/frauddistill/e1_final_v3/panel_builder_v31.py`（split_by_family 支持自定义 strata）。
- 报告：`reports/e1_final_triad_v32/`（执行报告 + 完整分析报告 + 本总报告）。
- v3.1 报告已移入 `archive/e1_v31_reports_20260802/` 归档。
- 数据产出：`data/prepared/e1_final_triad_v32/`（gitignore，不入库）。

## 八、结论与后续建议

- **达成**：6000 均衡面板（3000/3000）按 v1 风格构造完成；双 LLM Gold 质量 gate PASS；反快捷方式 gate PASS；q+y 为最优视图（B 面板内）；成本 ¥29.70 控制在 ¥30 内；C 回放如实完成。
- **未达成**：Frozen Anchor q+y≥0.90（0.871）；q+y 相对 y-only 的机制增益；C 上 q+y 优于 y-only。
- **建议**（不消耗新预算，供下一轮考虑）：
  1. 若要压低 q-only：扩大同 q 配对（对 levelup 正例补充同 q 真实负例）或引入 OR-Bench 风格 hard-benign 请求。
  2. 若要拉开 q+y 与 y-only：提高 hard_unsafe（边界/漏检）占比，使 y 单独信号下降、q 语境校正显性化。
  3. 若要 C 迁移：B 训练集中加入更高比例的真实 A 响应（如 v3.1 B3200 的做法），或在 C 上明确采用"排序/recall@FPR"口径而非冻结阈值。