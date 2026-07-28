# FraudDistill E1 CPU CCF-A v6 任务收尾报告

生成时间：2026-07-28

## 1. 本轮执行结论

本轮已按 `FraudDistill_E1_CPU_CCF-A_冻结方案_v6.md` 完成代码整改、旧结果归档、G0 数据可行性审计和 G1 小规模 Pilot Gate。结论为：

```text
E1_G1_PILOT_STOP
```

因此本轮没有解封 formal test，也没有进行 5-seed full formal run。原因不是资源不足，也不是 G0 数据数量不足，而是当前轻量 CPU relation 模型没有证明 `q+y` 相对 `y-only` 的稳定关系增益。

## 2. 归档与目录

- 旧输出与旧 prepared 数据已归档到：`archive/pre_v6_20260728_205158/`
- v6 manifest 输出：`data/prepared/e1_relation_gate_v6/`
- v6 pilot 输出：`outputs/e1_relation_gate_v6/pilot/`
- 本收尾报告：`reports/E1_CPU_CCF-A_v6_任务收尾报告_中文.md`

`data/`、`outputs/`、`archive/` 仍由 `.gitignore` 排除，不会把大数据、模型、预测文件或缓存推到 GitHub。

## 3. 代码整改内容

本轮新增/修改的关键点：

- 新增 `configs/experiments/e1_relation_gate_v6.yaml`，冻结 v6 的 G0/G1 规模、seeds、模型和 gate。
- 新增 `scripts/build_e1_relation_manifests.py`，生成 R1/R2/R3、G0 census、label provenance、component audit、relation funnel 和 license lock。
- 新增 `scripts/run_e1_relation_pilot_v6.py`，执行 3-seed pilot、q-shuffle、cluster bootstrap、LOSO source、资源统计和中文报告。
- 新增 `src/frauddistill/exp1_ccfa/relation_manifest.py`，实现 v6 的 response-level gold 数据构造和 component/group split。
- 修改 `public_gold.py`：删除 Aegis prompt-label fallback，空 response 直接排除。
- 修改 `fraud_taxonomy.py`：metadata/category 优先，文本关键词只作为更谨慎的 fraud-domain 辅助，不再把 broad privacy/cyber 直接升格。
- 修改 `embedding_cache.py`：加入 per-text cache key 和逐文本缓存路径。
- 修改 `pairlite_cpu.py`：R2 hashed cross-token 改为基于 TF-IDF idf 权重选择 top terms。
- 扩展 `relation_features.py`：加入中英文 refusal、defense、credential、money transfer、impersonation、job scam、romance scam、actionability 等关系特征。
- 新增 `tests/test_e1_relation_v6.py`，覆盖 v6 fallback、same-q group split、split leakage 和 per-text cache key。

## 4. G0 数据审计结果

G0 已通过：

| 检查项 | 结果 |
|---|---:|
| PKU-SafeRLHF rows | 164,196 |
| Aegis response-level non-empty rows | 813 |
| BeaverTails rows | 336,984 |
| R1 same-q pairs | 4,585 |
| R2 y-hard groups | 1,500 |
| R3 natural rows | 3,000 |
| prompt-label fallback | 0 |
| empty response | 0 |
| cross-split component leakage | 0 |
| pilot train/model_dev/calibration/pilot_test | 8,000 / 1,000 / 1,000 / 2,400 |

G0 的主要修复点是 split 策略：R1/R2 必须按 group/component 切分，组内 safe/unsafe 双回答保留在同一 split，而不是全局只保留一行。

## 5. G1 Pilot 主要结果

3 个 seed 的 E1-Score：

| Model | 20260724 | 20260725 | 20260726 | 均值 |
|---|---:|---:|---:|---:|
| q-only | 0.6336 | 0.6349 | 0.6336 | 0.6340 |
| y-only | 0.8064 | 0.8065 | 0.8069 | 0.8066 |
| additive q+y | 0.7956 | 0.7944 | 0.7952 | 0.7951 |
| relation q+y | 0.7160 | 0.7179 | 0.7158 | 0.7166 |

关键统计：

- `relation q+y - y-only` 的 bootstrap delta 约为 `-0.089`，三个 seed 全为负。
- q-shuffle 后 relation 平均下降约 `0.092`，说明模型确实读取了 q-y 关系特征，但该关系特征没有带来正确增益。
- q-only 在 R1 的 AUROC 为 0.5，same-q 泄漏控制有效；但总体 q-only AUROC 约 0.693，R2/R3 仍存在 prompt/domain 信号。
- y-only 处于文档要求的 0.70-0.85 区间，但 relation q+y 未达到 0.84，也未超过 y-only。

## 6. Gate 判定

通过项：

- q-only < y-only
- R1 q-only AUROC 在 [0.47, 0.53]
- y-only E1-Score 在 [0.70, 0.85]
- CPU-only、Peak RSS、wall time、artifact size 均通过

失败项：

- overall q-only AUROC > 0.58
- relation q+y E1-Score < 0.84
- relation q+y - y-only < 0.05，且方向为负
- 3/3 seed 方向不满足 q+y > y-only
- q-shuffle 后不是回到 y-only ±0.03，而是明显低于 y-only

## 7. 分析

本轮结果说明，公开 response-level safety gold 在当前构造下仍主要由回答文本本身决定。`y-only` 已经能捕捉拒答、行动步骤、凭证、转账等主要信号；加入 q 后，当前 sparse relation hash 和手工交互特征引入了噪声，尤其在 R1 上把 F1 从约 0.706 降到约 0.62-0.63。

R1 的 q-only 控制是成功的：同一问题下 safe/unsafe 双回答使 q-only AUROC 回到 0.5。但 R2/R3 的 prompt/domain 信号仍偏强，导致总体 q-only AUROC 超线。这意味着后续如果继续推进，需要优先重新设计 R2/R3 的 prompt-family 平衡，而不是扩大训练行数。

模型层面，`additive q+y` 比 relation hash 更接近 y-only，但仍低于 y-only，说明简单拼接或当前 cross-token hash 都不足以形成 v6 要求的正向关系证据。继续 full formal run 只会把失败结论放大，不符合文档“Pilot 未通过不得解封 formal test”的要求。

## 8. 后续建议

下一轮应只改一个因素，建议优先改数据定义：

- 对 R2/R3 增加 q-only nuisance matching，使总体 q-only AUROC 回到 ≤0.58。
- 保持 response-level official label，不引入 prompt-label fallback 或项目自标 test label。
- 在不读取 formal test 的前提下，再跑一次 v6r1 pilot。
- 若数据修复后 relation 仍低于 y-only，再考虑替换 relation head，例如低维 dense interaction + 小 MLP，而不是继续调阈值。

本轮不建议进入实验 2 或 full formal。
