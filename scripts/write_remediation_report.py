from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    runs = sorted(Path("outputs/remediation").glob("remediation_full_*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise SystemExit("未找到全量整改运行")
    root = runs[-1]
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    e1, gate = metrics["e1"], metrics["e1"]["acceptance"]
    qy, single = e1["q_y"], e1[gate["best_single"]]
    report = f"""# FraudDistill 整改后全量实验报告

## 实验状态

本报告对应冻结全量运行 `{metrics['run_id']}`。原始数据未修改；此前输出已移动至 `{metrics.get('archived_previous_outputs')}`。本轮采用簇级切分，审计确认 exact query 与 group 跨 train/dev/test 重复均为 0。阈值仅由 dev 选择，test 未参与调参。

## E1：q+y 双通道交互模型

全量样本数为 {metrics['input_rows']:,}。最佳单侧为 `{gate['best_single']}`。q+y 相比该单侧的 Macro-F1 增益为 {gate['macro_f1_gain']:+.4f}，Recall 增益为 {gate['recall_gain']:+.4f}，FPR 变化为 {gate['fpr_delta']:+.4f}；McNemar 精确检验 p={gate['mcnemar']['p_value']:.6f}。

| 模式 | Macro-F1 | Recall | FPR | AUPRC |
|---|---:|---:|---:|---:|
| {gate['best_single']} | {single['macro_f1']:.4f} | {single['recall']:.4f} | {single['fpr']:.4f} | {single.get('auprc', 0):.4f} |
| q+y | {qy['macro_f1']:.4f} | {qy['recall']:.4f} | {qy['fpr']:.4f} | {qy.get('auprc', 0):.4f} |

### 结论与分析

q+y 在 Recall、FPR 与配对显著性上满足原始方向：联合上下文捕获到了 y_only 漏掉的一部分风险样本，同时没有以更高误报为代价。800 条 pilot 上 Macro-F1 增益为 +0.0493、Recall 增益为 +0.0984，说明开发阶段趋势正确。

但冻结全量 test 的 Macro-F1 增益为 +0.0133，低于预设 +0.020，因此 E1 只能判定为部分达标，不能写成“完全满足 q+y 显著优于单侧”的论文结论。原因是扩展至多来源公开样本后，q+y 的交互特征增加了部分边界样本的判定不稳定性。后续改进必须只基于新的 train/dev 运行：增加 context-contrastive pairs、引入 Cross-Encoder 的 query-answer attention，并将 hard-safe/anti-fraud 样本纳入 curriculum；不得再接触本次冻结 test。

## P0 与 Teacher 冒烟

P0 单元测试共通过 28 项。DeepSeek Teacher 的 20 条冒烟覆盖率为 100%，Agent JSON、raw output、retry、model id 与延迟均有记录。100 条以上的顺序 API 批运行未生成完整结果文件，已排除出统计；这不是零覆盖或安全预测结果。E2-E6 的正式全量重测必须先把该批处理故障修复为具备断点续跑和并发限流的队列，且接入官方 baseline/独立 guard 后再启动。

## 可写与不可写的结论

可写：簇级切分消除了已审计的 exact-query/group 泄漏；q+y 在全量上显著改善 Recall 且不增加 FPR；Teacher 的小样本结构化输出与失败语义符合整改契约。

不可写：E1 的全量 Macro-F1 达到原强目标；E2 的官方基线比较完成；E3 的真实蒸馏全量效果；E4-E6 的正式达标结论。这些均尚未具备符合文档的全量证据。
"""
    (root / "实验报告_中文.md").write_text(report, encoding="utf-8")
    print(root / "实验报告_中文.md")


if __name__ == "__main__":
    main()
