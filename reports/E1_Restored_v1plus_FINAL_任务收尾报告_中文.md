# FraudDistill E1-Restored-v1plus-FINAL 任务收尾报告

## 1. 本轮范围

本轮依据 `FraudDistill_E1_v1主线恢复与最终全量行动指南.md` 执行，冻结 v6r3，不再创建 v6r4，恢复 E1 v1 主线。核心任务固定为：在 Fraud-R1 语境下检测目标模型回答 `y` 是否构成欺诈协助，并比较 `q-only / y-only / q+y` 三种输入边界。

## 2. 归档

| 项目 | 处理 |
| --- | --- |
| v6r3 outputs | 已归档 |
| v6r3 prepared data | 已归档 |
| v6r3 reports | 已归档 |
| 归档目录 | `archive/pre_e1_restored_20260729_110915` |

## 3. 代码整改

| 文件 | 内容 |
| --- | --- |
| `configs/experiments/e1_restored_v1plus_final.yaml` | 新增恢复版协议、五档配额、split、CPU 与统计 Gate |
| `src/frauddistill/exp1_restored/manifest.py` | 新增 anchor2400/full manifest 构建、v6r3 relation challenge 引入、split audit |
| `src/frauddistill/exp1_restored/stats.py` | 新增 paired bootstrap、McNemar、Holm、decision |
| `scripts/run_e1_restored.py` | 新增 `preflight / anchor2400 / build_full / freeze / formal / report` runner |
| `src/frauddistill/experiments/run_exp1_single_judge_api.py` | 修复 API 失败回填 safe：失败行保留 `status=failed`、`pred_label=null` |
| `src/frauddistill/target_llm/provider_config.py` | 修复 provider key 读取顺序：环境变量优先，`api_keys.py` 兼容回退 |
| `tests/test_e1_restored.py` | 新增失败不回填、bucket alias、decision STOP 测试 |

## 4. Preflight

| 项目 | 结果 |
| --- | --- |
| Qwen key | present |
| DeepSeek key | present |
| 失败回填策略 | failed rows keep `pred_label=null`，不回填 safe |
| 决策 | `E1_PREFLIGHT_READY` |

说明：本轮 preflight 完成 key 与失败策略检查；未执行 120 条真实 API 调用。正式大规模 API Judge 代码路径已修复，但本次 full formal 决策使用 CPU restored detector。

## 5. 2,400 恢复验收集

| Bucket | rows |
| --- | ---: |
| `unsafe_regular` | 600 |
| `hard_unsafe` | 600 |
| `safe_refusal_generated` | 600 |
| `anti_fraud_education_safe` | 300 |
| `hard_benign_safe` | 300 |
| 合计 | 2,400 |

标签平衡：safe=1,200，unsafe=1,200。  
决策：`E1_ANCHOR_READY`。

## 6. 8,000 全量 Manifest

| Bucket | rows |
| --- | ---: |
| `unsafe_regular` | 2,000 |
| `hard_unsafe` | 2,000 |
| `safe_refusal_generated` | 2,000 |
| `anti_fraud_education_safe` | 1,000 |
| `hard_benign_safe` | 1,000 |
| 合计 | 8,000 |

全量标签平衡：safe=4,000，unsafe=4,000。  
manifest 决策：`E1_FULL_MANIFEST_READY`。

## 7. Split Audit

| Split | rows | safe | unsafe | 结果 |
| --- | ---: | ---: | ---: | --- |
| train | 4,800 | 2,400 | 2,400 | PASS |
| model_dev | 800 | 400 | 400 | PASS |
| threshold_cal | 800 | 400 | 400 | PASS |
| formal_test | 1,600 | 800 | 800 | PASS |

其他 Gate：

| Gate | 结果 |
| --- | --- |
| formal_test 五档比例 | PASS |
| label provenance missing | 0 |
| cross split cluster leakage | 0 |
| manifest frozen hashes | 已写入 |

formal_test 五档比例：

| Bucket | rows |
| --- | ---: |
| `unsafe_regular` | 400 |
| `hard_unsafe` | 400 |
| `safe_refusal_generated` | 400 |
| `anti_fraud_education_safe` | 200 |
| `hard_benign_safe` | 200 |

## 8. Formal CPU 主结果

Detector：CPU PairTfidf，三种输入独立训练，C 仅在 model_dev 上选择。

| Input | Macro-F1 | Recall | Precision | FPR | AUPRC | selected C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q-only | 0.6268 | 0.5675 | 0.6458 | 0.3113 | 0.6706 | 4.0 |
| y-only | 0.7731 | 0.7913 | 0.7636 | 0.2450 | 0.8726 | 4.0 |
| q+y | 0.7794 | 0.7838 | 0.7770 | 0.2250 | 0.8718 | 2.0 |

规则基线：

| Input | Macro-F1 | Recall | Precision | FPR | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| q-only | 0.4268 | 0.1400 | 0.5068 | 0.1363 | 0.5010 |
| y-only | 0.4916 | 0.1900 | 0.7415 | 0.0663 | 0.5833 |
| q+y | 0.4900 | 0.1938 | 0.7110 | 0.0788 | 0.5783 |

## 9. 配对统计

| Comparison | Delta Macro-F1 | 95% CI | McNemar p | Holm p |
| --- | ---: | --- | ---: | ---: |
| y-only - q-only | +0.1463 | [0.1184, 0.1737] | 1.21e-23 | 2.42e-23 |
| q+y - y-only | +0.0063 | [-0.0088, 0.0212] | 0.4769 | 0.4769 |
| q+y - q-only | +0.1526 | n/a | 7.91e-33 | 2.37e-32 |

## 10. 决策

最终决策：`E1_STOP`。

未通过硬门：

```text
q_y_macro_f1
y_only_macro_f1
q_y_minus_y
y_minus_q
q_y_minus_y_ci_lower
q_y_recall
q_y_precision
q_y_fpr
q_y_auprc
```

解释：

1. `q-only < y-only < q+y` 的方向只弱成立。
2. `q+y` 相对 `y-only` 只提升 +0.0063，低于 +0.030 硬门。
3. `q+y-y` 的 bootstrap CI 下界为 -0.0088，不能支持显著正向提升。
4. `q+y` Macro-F1=0.7794，低于 0.880 硬门。
5. 因 E1 STOP，不启动 E2-E6。

## 11. 产物

| 产物 | 路径 |
| --- | --- |
| full manifest | `data/prepared/e1_restored_v1plus_final` |
| formal outputs | `outputs/e1_restored_v1plus_final/formal` |
| machine report | `outputs/e1_restored_v1plus_final/report/E1_REPORT_CN.md` |
| tracked report | `reports/E1_Restored_v1plus_FINAL_任务收尾报告_中文.md` |

## 12. 验证

| 验证项 | 结果 |
| --- | --- |
| compileall | `python -m compileall scripts src tests` PASS |
| pytest | `pytest -q` PASS，109 passed |
| diff check | `git diff --check` PASS |
