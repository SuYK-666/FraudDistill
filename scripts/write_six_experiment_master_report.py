from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOC_RESULTS = ROOT / "docs" / "results"
DOC_REPRO = ROOT / "docs" / "reproduction"
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
    overview = f"""# FraudDistill 六实验全量总报告

## 运行说明

本报告汇总同一轮 `high_standard_full` 的六组正式输出。每组的原始预测、指标、模型、审计、配置、失败记录和逐实验报告均保留在原运行目录；本总报告不重采样、不覆盖任何原始结果。附带的产物清单记录每个文件的路径、字节数与 SHA-256，用于后续完整性核验。

## 总结结论

{summary_bullets()}

## 原始产物保留

完整产物清单：`outputs/SIX_EXPERIMENTS_ARTIFACT_MANIFEST.tsv`。清单覆盖本轮每一个配置、预测、表格、模型、审计与报告文件。原始数据仍位于本地 `data/`，按照 `.gitignore` 不提交到 GitHub。
"""
    master.write_text(overview + "\n".join(sections) + f"\n生成时间：{datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    manifest.write_text("path\tbytes\tsha256\n" + "\n".join(f"{path}\t{size}\t{sha}" for path, size, sha in artifact_rows) + "\n", encoding="utf-8")
    DOC_RESULTS.mkdir(parents=True, exist_ok=True)
    DOC_REPRO.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master, DOC_RESULTS / master.name)
    shutil.copy2(manifest, DOC_RESULTS / manifest.name)
    (DOC_REPRO / "REPRODUCE_SIX_EXPERIMENTS.md").write_text(reproduction_doc(), encoding="utf-8")
    print(master)
    print(manifest)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary_bullets() -> str:
    lines = []
    e1 = read_json(OUT / "exp1_input_ablation" / RUN / "metrics.json")
    if e1 and "q+y" in e1 and "y_only" in e1:
        gain = e1["q+y"]["macro_f1"] - e1["y_only"]["macro_f1"]
        lines.append(f"- E1：q+y Macro-F1={e1['q+y']['macro_f1']:.4f}，y-only={e1['y_only']['macro_f1']:.4f}，增益={gain:.4f}；AUPRC={e1['q+y']['auprc_unsafe']:.4f}，阈值仅在 dev 上选择。")
    e2 = read_json(OUT / "exp2_prior_work_comparison" / RUN / "metrics.json")
    if e2 is not None:
        lines.append(f"- E2：共输出 {len(e2)} 行 proxy/coverage 结果；仍需官方 evaluator/checkpoint 才能作为论文主表。")
    e3 = read_json(OUT / "exp3_agent_distillation_ablation" / RUN / "metrics.json")
    if e3 and e3.get("student"):
        first, last = e3["student"][0], e3["student"][-1]
        lines.append(f"- E3：Student 从 {first['Variant']} Macro-F1={first['Macro-F1']} 到 {last['Variant']} Macro-F1={last['Macro-F1']}；新增 nested、leave-one-out、组件压力三类表。")
    e4 = read_json(OUT / "exp4_unseen" / RUN / "worst_group_metrics.json")
    if e4:
        lines.append(f"- E4：最弱 held-out 项为 {e4.get('Held-out', 'unknown')}，Macro-F1={e4.get('Macro-F1', '')}，保留类别覆盖限制。")
    e5 = read_json(OUT / "exp5_calibration" / RUN / "metrics.json")
    if e5:
        lines.append("- E5：输出 raw、Platt 与多档 FPR-UCB 操作点，报告 calibration split/test split 的样本量和 UCB。")
    e6 = read_json(OUT / "exp6_multi_api" / RUN / "metrics.json")
    if e6:
        n_models = len(e6.get("student_vs_prompt_reference", []))
        lines.append(f"- E6：覆盖 {n_models} 个已有目标模型 generations；行为指标 FAR/RFR/CRR/ORR 使用独立字段计算，仍标注为 detector-dependent。")
    return "\n".join(lines) if lines else "- 本轮指标文件尚未生成。"


def reproduction_doc() -> str:
    return """# Reproduce Six Experiments

本仓库的统一重跑入口：

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/run_high_standard_rerun.py all --bootstrap 500
python scripts/write_six_experiment_master_report.py
pytest -q
```

运行顺序固定为 smoke、pilot、full；smoke/pilot 会自动归档，全量正式产物保留在 `outputs/*/high_standard_full/`。全量重跑前脚本会把已有 `high_standard_full` 归档到 `archive/pre_high_standard_full_rerun_*`。

注意：`data/`、`outputs/`、`archive/`、模型文件和 `api_keys.py` 不提交到 GitHub。报告副本位于 `docs/results/`，原始实验产物仍以本地 `outputs/` 为准。
"""


if __name__ == "__main__":
    main()
