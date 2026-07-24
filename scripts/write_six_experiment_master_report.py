from __future__ import annotations

import argparse
import hashlib
import csv
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
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN)
    args = parser.parse_args()
    RUN = args.run_id
    suffix = "" if RUN == "high_standard_full" else "GATE_" if "gate" in RUN else "SMALL_"
    master = OUT / f"SIX_EXPERIMENTS_{suffix}MASTER_REPORT_中文.md"
    manifest = OUT / f"SIX_EXPERIMENTS_{suffix}ARTIFACT_MANIFEST.tsv"
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
    report_scope = "全量正式" if RUN == "high_standard_full" else "中等规模门控" if "gate" in RUN else "小规模重跑"
    overview = f"""# FraudDistill 六实验{report_scope}总报告

## 运行说明

本报告汇总同一轮 `{RUN}` 的六组输出。每组的原始预测、指标、模型、审计、配置、失败记录和逐实验报告均保留在原运行目录；本总报告不重采样、不覆盖任何原始结果。附带的产物清单记录每个文件的路径、字节数与 SHA-256，用于后续完整性核验。

## 总结结论

{summary_bullets()}

## 原始产物保留

完整产物清单：`{manifest.relative_to(ROOT)}`。清单覆盖本轮每一个配置、预测、表格、模型、审计与报告文件。原始数据仍位于本地 `data/`，按照 `.gitignore` 不提交到 GitHub。
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_bullets() -> str:
    lines = []
    e1 = read_json(OUT / "exp1_input_ablation" / RUN / "metrics.json")
    if e1 and "q+y" in e1 and "y_only" in e1:
        gain = e1["q+y"]["macro_f1"] - e1["y_only"]["macro_f1"]
        lines.append(f"- E1：q+y Macro-F1={e1['q+y']['macro_f1']:.4f}，y-only={e1['y_only']['macro_f1']:.4f}，增益={gain:.4f}；AUPRC={e1['q+y']['auprc_unsafe']:.4f}，阈值仅在 dev 上选择。")
    e1_context = read_csv_rows(OUT / "exp1_input_ablation" / RUN / "tables" / "context_critical_table.csv")
    if e1_context:
        qy = next((row for row in e1_context if row.get("Input") == "q+y"), None)
        yonly = next((row for row in e1_context if row.get("Input") == "y_only"), None)
        if qy and yonly:
            lines.append(f"- E1 Track B：Context-Critical paired N={qy.get('N')}，q+y Macro-F1={qy.get('Macro-F1')}，y-only Macro-F1={yonly.get('Macro-F1')}，Pair consistency 见逐实验表；该轨道为 procedural weak benchmark。")
    e2 = read_json(OUT / "exp2_prior_work_comparison" / RUN / "metrics.json")
    if e2 is not None:
        lines.append(f"- E2：共输出 {len(e2)} 行 proxy/coverage 结果；仍需官方 evaluator/checkpoint 才能作为论文主表。")
    e3 = read_json(OUT / "exp3_agent_distillation_ablation" / RUN / "metrics.json")
    if e3 and e3.get("student"):
        first, last = e3["student"][0], e3["student"][-1]
        lines.append(f"- E3：Student 从 {first['Variant']} Macro-F1={first['Macro-F1']} 到 {last['Variant']} Macro-F1={last['Macro-F1']}；新增 nested、leave-one-out、组件压力三类表。")
    e3_stress = read_csv_rows(OUT / "exp3_agent_distillation_ablation" / RUN / "tables" / "stress_agent_ablation.csv")
    if e3_stress:
        full = next((row for row in e3_stress if row.get("Variant") == "Full learned"), e3_stress[-1])
        single = next((row for row in e3_stress if row.get("Variant") == "Single Judge"), e3_stress[0])
        lines.append(f"- E3 Stress：Full learned Macro-F1={full.get('Macro-F1')}，Single Judge={single.get('Macro-F1')}，用于组件不可替代性压力验证；标签为 procedural weak stress。")
    e4 = read_json(OUT / "exp4_unseen" / RUN / "worst_group_metrics.json")
    if e4:
        lines.append(f"- E4：最弱 held-out 项为 {e4.get('Held-out', 'unknown')}，Macro-F1={e4.get('Macro-F1', '')}，保留类别覆盖限制。")
    e4_loco = read_csv_rows(OUT / "exp4_unseen" / RUN / "tables" / "procedural_loco5.csv")
    if e4_loco:
        lines.append(f"- E4 扩展：procedural five-category LOCO 覆盖 {len(e4_loco)} 类，每类 N={e4_loco[0].get('N')}；source/language holdout 仍显示真实跨源迁移不足。")
    e5 = read_json(OUT / "exp5_calibration" / RUN / "metrics.json")
    if e5:
        lines.append("- E5：输出 raw、Platt 与多档 FPR-UCB 操作点，报告 calibration split/test split 的样本量和 UCB。")
    e6 = read_json(OUT / "exp6_multi_api" / RUN / "metrics.json")
    if e6:
        e6_main = e6.get("student_vs_pair_silver", e6.get("student_vs_prompt_reference", []))
        n_models = len(e6_main)
        lines.append(f"- E6：覆盖 {n_models} 个已有目标模型 generations；新回复不继承 prompt gold，主表改为 student_vs_pair_silver；仍需开放 guard 共识替换 deterministic proxy。")
        lomo = e6.get("leave_one_model_out", [])
        if lomo:
            macros = [float(row["Macro-F1"]) for row in lomo if row.get("Macro-F1") not in {"", None}]
            if macros:
                lines.append(f"- E6 LOMO：已有 {len(lomo)} 个 held-out model family，Macro-F1 范围 {min(macros):.4f}-{max(macros):.4f}；仍未达到 12 模型 CCF-A 目标。")
    return "\n".join(lines) if lines else "- 本轮指标文件尚未生成。"


def reproduction_doc() -> str:
    return f"""# Reproduce Six Experiments

本仓库的统一重跑入口：

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/run_high_standard_rerun.py small --bootstrap 300 --small-limit 720 --api-provider qwen --api-probe-limit 6
python scripts/write_six_experiment_master_report.py --run-id ccfa_small_qwen
git status --short
python scripts/run_high_standard_rerun.py gate --bootstrap 500 --api-provider qwen --api-probe-limit 12
python scripts/write_six_experiment_master_report.py --run-id ccfa_medium_gate
python scripts/run_high_standard_rerun.py all --bootstrap 500
python scripts/write_six_experiment_master_report.py
pytest -q
```

本轮小规模审查产物保留在 `outputs/*/ccfa_small_qwen/`；E1-E5 使用 qwen API teacher probe 写入各实验 `raw_outputs/qwen_teacher_probe.jsonl`，实验指标仍保留 gold/silver/proxy 的标签边界。

复盘后的中等规模门控产物保留在 `outputs/*/ccfa_medium_gate/`。Gate 流程要求从 clean committed SHA 启动，记录 `git_status_porcelain.txt`、split manifest、split hash、配置 hash，并修复 group split、E5 calibration split、E3 teacher-free inference 和 E6 generation label schema。

运行顺序固定为 smoke、pilot、full；smoke/pilot 会自动归档，全量正式产物保留在 `outputs/*/high_standard_full/`。全量重跑前脚本会把已有 `high_standard_full` 归档到 `archive/pre_high_standard_full_rerun_*`。

注意：`data/`、`outputs/`、`archive/`、模型文件和 `api_keys.py` 不提交到 GitHub。报告副本位于 `docs/results/`，原始实验产物仍以本地 `outputs/` 为准。
"""


if __name__ == "__main__":
    main()
