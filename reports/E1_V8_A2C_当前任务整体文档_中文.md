# FraudDistill E1 V8 A2C 当前任务整体文档

生成时间：2026-07-30  
协议：`E1-V8-A2C-OFFICIAL-DELTA-STAGE-PRESSURE-v1.0`  
当前代码提交：`6ac60ffcad1ffb3af0ee1de6b3815dce522332b1`

## 1. 本轮已完成的整理与归档

已将上一轮可见结果归档到：

`archive/pre_e1_v8_20260730_115108`

归档内容包括：

| 原位置 | 归档后位置 |
|---|---|
| `outputs/e1_dual_v7` | `archive/pre_e1_v8_20260730_115108/outputs_e1_dual_v7` |
| `data/prepared/e1_dual_v7` | `archive/pre_e1_v8_20260730_115108/data_prepared_e1_dual_v7` |
| `reports` 中的 V7 报告 | `archive/pre_e1_v8_20260730_115108/reports_v7` |

原始 Fraud-R1 数据、运行缓存和归档目录均未提交到 GitHub，符合 `.gitignore` 和 V8 数据再分发限制。

## 2. 本轮代码整改

本轮新增独立 V8 执行链，没有继续修改 V7 主脚本：

| 文件 | 作用 |
|---|---|
| `configs/experiments/e1_v8_a2c.yaml` | V8 协议配置、模型、并发、Gate、路径 |
| `scripts/run_e1_v8_a2c.py` | P0、A-Delta、C-ISO、C-ADAPT、Probe、Decision、报告统一 CLI |
| `src/frauddistill/e1_v8/fraudr1_stage_loader.py` | Fraud-R1 四阶段 canonical loader 与 split |
| `src/frauddistill/e1_v8/official_prompt_renderer.py` | 官方 Role-play、C-ISO、V7 parity renderer |
| `src/frauddistill/e1_v8/consensus.py` | O/X 双视图 strict consensus、kappa、confusion |
| `src/frauddistill/e1_v8/capacity_projection.py` | Wilson lower bound 与容量投影 |
| `src/frauddistill/e1_v8/diagnostic_probe.py` | Probe panel 构造与 q-only 结构上限 |
| `tests/test_e1_v8_a2c.py` | V8 loader、renderer、consensus、probe 单测 |

过程中发现并修复了一个重要缓存问题：旧 fingerprint 只依赖 prompt/参数，极少数重复 prompt 会让不同 response 映射被覆盖。修复后 target fingerprint 显式包含 `response_id`、`canonical_id`、`stage_id`，并且标签阶段按 `response_id` 读取最新响应映射，避免重复 prompt 丢样本。

## 3. 测试与 P0 Gate

单元测试：

| 测试 | 结果 |
|---|---|
| `python -m pytest tests/test_e1_v8_a2c.py -q` | 6 passed |
| `python -m py_compile scripts/run_e1_v8_a2c.py` | PASS |

P0 审计结果：`P0_PASS`

| 检查项 | 结果 |
|---|---:|
| canonical case | 2,141 |
| 每 case 四阶段 | PASS |
| stage id 0/1/2/3 | PASS |
| language | en/zh only |
| category | 5 类，无 unknown |
| data_type | message/email/job posting |
| Pilot split | 200 |
| Model-Dev split | 400 |
| Frozen Anchor split | 1,541 |
| split overlap | 0 |
| prompt fixture | PASS |
| secret scan | PASS |
| dataset terms gate | PASS |

数据分布：

| 项目 | 分布 |
|---|---|
| language | en 1,071；zh 1,070 |
| category | fake job posting 300；fraudulent service 600；impersonation 600；network friendship 169；phishing 472 |
| data_type | job posting 300；message 1,369；email 472 |

## 4. API 参数探测

第一次参数探测：`MODEL_PARAM_PROBE_PASS`

| 角色 | 模型 | 状态 |
|---|---|---|
| Qwen Target | `qwen3.7-plus-2026-05-26` | ok |
| Qwen Labeler | `qwen3.7-max-2026-06-08` | ok |
| DeepSeek Target | `deepseek-v4-flash` | ok |
| DeepSeek Labeler | `deepseek-v4-pro` | ok |

在 C-ISO 标注进行中，Qwen API 后续开始返回 `Arrearage` 400 错误。复测结果：

| 角色 | 模型 | 状态 |
|---|---|---|
| Qwen Target | `qwen3.7-plus-2026-05-26` | STOP：`Arrearage` |
| Qwen Labeler | `qwen3.7-max-2026-06-08` | STOP：`Arrearage` |
| DeepSeek Target | `deepseek-v4-flash` | ok |
| DeepSeek Labeler | `deepseek-v4-pro` | ok |

因此本轮不是代码卡住，而是 Qwen 账户/额度状态在运行过程中失效。根据 V8 协议，双标签器缺一不可，不能用 DeepSeek 单标签器伪造 strict consensus。

## 5. A-Delta 已完成结果

A-Delta 目标回答：`A_GENERATE_PASS`

| 指标 | 值 |
|---|---:|
| expected | 1,600 |
| observed | 1,600 |
| completion rate | 100.0% |
| valid response rate | 100.0% |
| finish reason rate | 100.0% |
| provenance rate | 100.0% |
| truncation rate | 2.625% |

A-Delta 标签：`A_LABEL_PASS`

| 视图 | state agreement | binary agreement | kappa | uncertain |
|---|---:|---:|---:|---:|
| O official-y | 92.44% | 96.06% | 0.815 | 7.56% |
| X contextual-qy | 93.38% | 95.19% | 0.746 | 6.62% |

A-Delta 分析：`A_ANALYZE_GREEN`

| arm/model | n | O failure | X failure | O continue | X assist |
|---|---:|---:|---:|---:|---:|
| A0 parity / DeepSeek | 400 | 20 | 32 | 98 | 32 |
| A1 official / DeepSeek | 400 | 27 | 52 | 65 | 52 |
| A0 parity / Qwen | 400 | 13 | 17 | 10 | 17 |
| A1 official / Qwen | 400 | 10 | 14 | 22 | 14 |

关键结论：

- DeepSeek A1 official treatment 的 X contextual failure 达到 52/400 = 13.0%，明显高于 V7 的约 5%历史水平，也高于同期 A0 parity 的 8.0%。
- DeepSeek A1 在中英文均有风险事件：en 32、zh 20。
- DeepSeek A1 覆盖 5/5 风险类别。
- A1 global mixed canonical 为 26，达到 A-GREEN 中 `≥20` 的要求。
- Round-1 保守 Wilson 投影约 520，其中 DeepSeek 约 430，Qwen 约 90。
- A 阶段整体达成 V8 的 A-GREEN，说明官方 Role-play bundle 对自然风险容量有实质提升。

需要在论文叙事中如实说明：A1 提升是 official-alignment bundle effect，不能拆成“某一个 prompt 字段单独导致”。

## 6. C-ISO 当前完成度

C-ISO stage 1–3 目标回答：`C_ISO_GENERATE_PASS`

| 指标 | 值 |
|---|---:|
| expected | 2,400 |
| observed | 2,400 |
| completion rate | 100.0% |
| valid response rate | 100.0% |
| finish reason rate | 100.0% |
| provenance rate | 100.0% |
| truncation rate | 12.625% |

当前目标回答总量：

| track | latest response_id | 状态 |
|---|---:|---|
| A-Delta | 1,600 | 完成 |
| C-ISO stage 1–3 | 2,400 | 完成 |
| 合计 | 4,000 | 完成 |

C-ISO 标签阶段因 Qwen 失效中止：

| 标签统计 | 数量 |
|---|---:|
| raw label rows | 7,752 |
| ok labels | 7,589 |
| error labels | 163 |
| labeler_d ok | 3,874 |
| labeler_q ok | 3,715 |
| labeler_q error | 163 |
| error type | `BadRequestError / Arrearage` |

当前已落盘数据全部保留在：

`data/prepared/e1_v8_a2c`

## 7. 当前 STOP 原因

当前状态应判为：`ENGINEERING_PROVIDER_STOP`

原因不是实验设计失败，也不是代码执行链失败，而是 Qwen provider 在 C-ISO 标注阶段中途返回欠费/账户不可用错误。V8 要求 O/X 双视图、Qwen/DeepSeek 双标签器 strict consensus；在 Qwen labeler 不可用时继续分析 C 容量、Probe 和最终 Full GO 都不合规。

不能做的替代操作：

- 不能只用 DeepSeek labeler 生成 strict consensus；
- 不能把已失败的 Qwen label 调用按 safe 或 uncertain 人工补齐；
- 不能把 C-ISO 目标回答直接交给 Probe；
- 不能为了得到更好结果跳过 Qwen 或替换为未预注册模型。

可恢复方式：

1. 恢复 `QWEN_API_KEY` 对 `qwen3.7-plus-2026-05-26` 和 `qwen3.7-max-2026-06-08` 的可用额度。
2. 重新运行：
   `python scripts/run_e1_v8_a2c.py --phase model-param-probe`
3. 若探测 PASS，继续运行：
   `python scripts/run_e1_v8_a2c.py --phase c-iso-label`
4. 脚本会根据 fingerprint 自动跳过已成功标签，只补齐缺失/失败对应的新调用。
5. 之后继续：
   `c-adaptive`、`c-analyze`、`probe-build`、`probe-run`、`decide`、`report`。

## 8. 当前可用结论

本轮最重要的阶段性收获是 A-Delta 已经达成 GREEN，且解决了 V7 中“官方 prompt 不对齐导致风险率偏低”的核心疑问。DeepSeek 在官方 Role-play 下的 X contextual failure 达到 13.0%，相对同期 A0 的 8.0% 有明显提升；Qwen 仍更安全，A1 X failure 为 3.5%，可以作为强安全对照。

C-ISO 目标回答层面已经完整可用，说明四阶段自然压力的目标回答生产链路完成；但 C 的标签和容量 Gate 尚未完成，当前不能宣称 C-GREEN、Probe-GO 或 eligible-for-full。

## 9. GitHub 同步状态

已提交代码基线：

| commit | 说明 |
|---|---|
| `ee3dd07` | Implement E1 V8 A2C pilot protocol |
| `991d1f9` | Fix E1 V8 fingerprint shadowing |
| `6ac60ff` | Fix E1 V8 cache response mapping |

本报告提交后将作为当前任务收尾文档同步到 GitHub。数据、outputs、archive 不提交，但本地完整保留。
