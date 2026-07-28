# FraudDistill E1 v6r1 G0r STOP 任务收尾报告

生成时间：2026-07-28

## 1. 本轮结论

本轮已按 `FraudDistill_E1_v6r1_全量前最后修订与严格准入计划.md` 完成 v6r1 代码修订、旧 v6 产物归档、G0r 数据与实现 Gate 重跑。当前结论为：

```text
E1_V6R1_STOP
```

原因是 G0r 数据 Gate 未通过，因此没有进入 2k smoke、one-shot Pilot 或 Full Formal。该停止符合文档要求：R2 可用 groups 少于 3,500 或 R2 balance 不通过时，不允许继续解封后续测试。

## 2. 归档与输出

- 旧 v6 产物归档：`archive/pre_v6r1_20260728_215017/`
- v6r1 manifest 与 G0r 审计：`data/prepared/e1_relation_gate_v6r1/`
- v6r1 G0r 报告与决策：`outputs/e1_relation_gate_v6r1/g0/`
- 本收尾报告：`reports/E1_v6r1_G0r_STOP_任务收尾报告_中文.md`

`data/`、`outputs/`、`archive/` 继续由 `.gitignore` 排除，不提交大数据、预测、模型或缓存。

## 3. 代码修订摘要

本轮新增或修改：

- 新增 `configs/experiments/e1_relation_gate_v6r1.yaml`，冻结 v6r1 G0r/smoke/pilot/formal 的规模、seeds、bootstrap 和 Gate。
- 新增 `src/frauddistill/exp1_ccfa/residual_relation_cpu.py`，实现 `y-only logit + bounded relation residual` 的 CPU-only M5 模型，`\lambda=0` 时严格退化为 y-only。
- 扩展 `src/frauddistill/exp1_ccfa/relation_manifest.py`，加入 `row_uid` 去重、q-only fraud family、v6r1 R2 nuisance matching、R2 balance audit、manifest fingerprint 和 split components。
- 新增 `scripts/run_e1_relation_v6r1.py`，支持 `g0/smoke/pilot/formal` stage，formal 默认受 Pilot PASS 锁保护。
- 新增 `tests/test_e1_relation_v6r1.py`，覆盖 response-derived family、row_uid、比较器唯一性、分层 q-shuffle、E1 bootstrap 等关键失败模式。
- 更新 `scripts/build_e1_relation_manifests.py`，支持 `--protocol v6r1`。

## 4. G0r 数据结果

| 项目 | 结果 |
|---|---:|
| PKU-SafeRLHF rows | 164,196 |
| Aegis response-level non-empty rows | 813 |
| BeaverTails rows | 336,984 |
| R1 groups | 4,585 |
| R2 groups | 2,178 |
| R3 rows | 9,000 |
| prompt-label fallback | 0 |
| empty q/y | 0 |
| same row_uid duplicate | 0 |

R1 与 R3 数量满足要求，但 R2 仅形成 2,178 个有效 groups，低于文档最低准入线 3,500。

## 5. R2 Balance 结果

| 指标 | 结果 | Gate |
|---|---:|---|
| q nuisance SMD | 0.0028 | 通过 |
| y nuisance SMD | 0.0062 | 通过 |
| log answer length SMD | 0.0068 | 通过 |
| refusal gap | 0.0487 | 通过 |
| independent q-only AUROC | 0.5013 | 通过 |
| independent y-only AUROC | 0.5017 | 通过 |
| source pair types | 6 | 通过 |
| largest source rate | 0.7293 | 未通过 |
| R2 groups | 2,178 | 未通过 |

这说明匹配质量在 q/y/length/refusal 控制上是有效的，但公开数据在严格匹配后来源集中度过高，且数量不足以支撑 v6r1 的 Full 前准入。

## 6. 为什么没有继续 smoke/Pilot

文档明确要求：

- R2 可用 groups ≥ 4,600；若 3,500-4,599 才允许降级 formal R2；
- 若少于 3,500 groups，直接 STOP；
- R2 最大单一来源占比 ≤ 0.40；
- 不得降低匹配质量换数量。

当前 R2 groups=2,178，且最大来源占比=0.7293，因此继续运行 smoke/Pilot 会违反本轮冻结协议，也会产生不可用于论文叙事的结果。

## 7. 后续建议

下一步不要继续在当前公开 safety gold panel 上调阈值或扩大训练。若仍要推进 E1，需要改变前提之一：

- 引入新的、已有公开 response-level 标签的 fraud-conversation 数据；
- 或降低论文主张，只报告当前公开数据不足以支持强关系增益；
- 或把 R2 作为独立数据可行性问题重做，但必须产生新的 protocol version，不能复用 v6r1 的 one-shot Pilot 名义。
