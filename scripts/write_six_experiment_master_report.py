from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUN = "high_standard_full"

SPECS = [
    ("数据标签审计", "audit_label_integrity", "README.md"),
    ("实验1：输入消融", "exp1_input_ablation", "exp1_final_report.md"),
    ("实验2：现有工作对比", "exp2_prior_work_comparison", "exp2_final_report.md"),
    ("实验3：Agent 与蒸馏", "exp3_agent_distillation_ablation", "exp3_final_report.md"),
    ("实验4：未见泛化", "exp4_unseen", "exp4_final_report.md"),
    ("实验5：概率校准", "exp5_calibration", "exp5_final_report.md"),
    ("实验6：多 API", "exp6_multi_api", "exp6_final_report.md"),
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    master = OUT / "SIX_EXPERIMENTS_MASTER_REPORT_中文.md"
    manifest = OUT / "SIX_EXPERIMENTS_ARTIFACT_MANIFEST.tsv"
    artifact_rows = []
    sections = []
    for title, directory, report_name in SPECS:
        run = OUT / directory / RUN
        report = run / report_name
        if not report.exists():
            raise FileNotFoundError(report)
        content = report.read_text(encoding="utf-8").strip()
        sections.append(f"## {title}\n\n运行目录：`{run.relative_to(ROOT)}`\n\n{content}\n")
        for path in sorted(item for item in run.rglob("*") if item.is_file()):
            artifact_rows.append((str(path.relative_to(ROOT)), path.stat().st_size, digest(path)))
    overview = """# FraudDistill 六实验全量总报告

## 运行说明

本报告汇总同一轮 `high_standard_full` 的六组正式输出。每组的原始预测、指标、模型、审计、配置、失败记录和逐实验报告均保留在原运行目录；本总报告不重采样、不覆盖任何原始结果。附带的产物清单记录每个文件的路径、字节数与 SHA-256，用于后续完整性核验。

## 总结结论

- E1：q+y 相比 y-only 的 Macro-F1 从 0.8114 提升至 0.8375，Recall 从 0.7529 提升至 0.8664；FPR 为 0.0999，较 y-only 增加 0.0117，仍满足预注册的最大增量 0.020。该结论来自冻结 test，阈值只在 dev 上按联合约束选择。
- E2：结果仍是 proxy 对照，缺少官方 baseline/checkpoint；不得作为“优于现有工作”的论文主表。
- E3：Full Distill 的 Macro-F1 为 0.8105，较 Gold 的 0.8010 增加 0.0095，接近但未达到 +0.010 门槛；Teacher/Agent 的 unsafe Recall 仍不足，不能作为强教师结论。
- E4：fake job 强，impersonation Recall 偏低，phishing FPR 偏高；类别与 source/language 覆盖尚不完整。
- E5：Platt 将 ECE 从 0.1427 降至 0.0157，概率校准可写；FPR<=0.05 操作点的 Recall 为 0.5918，不能写低 FPR 高 Recall 强结论。
- E6：仅四个模型且没有独立 guard consensus，结果是 detector-dependent observation，不是可发表的安全排名。

## 原始产物保留

完整产物清单：`outputs/SIX_EXPERIMENTS_ARTIFACT_MANIFEST.tsv`。清单覆盖本轮每一个配置、预测、表格、模型、审计与报告文件。原始数据仍位于本地 `data/`，按照 `.gitignore` 不提交到 GitHub。
"""
    master.write_text(overview + "\n".join(sections) + f"\n生成时间：{datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    manifest.write_text("path\tbytes\tsha256\n" + "\n".join(f"{path}\t{size}\t{sha}" for path, size, sha in artifact_rows) + "\n", encoding="utf-8")
    print(master)
    print(manifest)


if __name__ == "__main__":
    main()
