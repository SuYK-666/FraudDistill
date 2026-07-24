from __future__ import annotations

import argparse
import csv
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
    ("实验3：Agent 与样本重加权", "exp3_agent_distillation_ablation", "exp3_final_report.md"),
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
    master, manifest = output_paths(RUN)
    artifact_rows: list[tuple[str, int, str]] = []
    sections: list[str] = []
    for title, directory, report_name in SPECS:
        run = OUT / directory / RUN
        report = run / report_name
        if not report.exists():
            raise FileNotFoundError(report)
        content = report.read_text(encoding="utf-8").strip()
        sections.append(f"## {title}\n\n运行目录：`{run.relative_to(ROOT)}`\n\n{content}\n")
        for path in sorted(item for item in run.rglob("*") if item.is_file()):
            artifact_rows.append((str(path.relative_to(ROOT)), path.stat().st_size, digest(path)))

    overview = build_overview(manifest)
    master.write_text(
        overview + "\n".join(sections) + f"\n生成时间：{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    manifest.write_text(
        "path\tbytes\tsha256\n" + "\n".join(f"{path}\t{size}\t{sha}" for path, size, sha in artifact_rows) + "\n",
        encoding="utf-8",
    )
    assert_publish_consistency(master, manifest, RUN)
    DOC_RESULTS.mkdir(parents=True, exist_ok=True)
    DOC_REPRO.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master, DOC_RESULTS / master.name)
    shutil.copy2(manifest, DOC_RESULTS / manifest.name)
    if RUN == "protocol_gate_v2":
        write_protocol_gate_v2_sidecars(artifact_rows)
    (DOC_REPRO / "REPRODUCE_SIX_EXPERIMENTS.md").write_text(reproduction_doc(), encoding="utf-8")
    print(master)
    print(manifest)


def output_paths(run_id: str) -> tuple[Path, Path]:
    if run_id == "protocol_gate_v2":
        return OUT / "PROTOCOL_GATE_V2_MASTER_REPORT_中文.md", OUT / "PROTOCOL_GATE_V2_ARTIFACT_MANIFEST.tsv"
    suffix = "" if run_id == "high_standard_full" else "GATE_" if "gate" in run_id else "SMALL_"
    return OUT / f"SIX_EXPERIMENTS_{suffix}MASTER_REPORT_中文.md", OUT / f"SIX_EXPERIMENTS_{suffix}ARTIFACT_MANIFEST.tsv"


def build_overview(manifest: Path) -> str:
    scope = "Protocol Gate v2" if RUN == "protocol_gate_v2" else "全量正式" if RUN == "high_standard_full" else "中等规模门控" if "gate" in RUN else "小规模重跑"
    return f"""# FraudDistill 六实验{scope}总报告

## 运行说明

本报告汇总同一轮 `{RUN}` 的七组输出：数据审计和实验 1-6。每组原始预测、指标、配置、模型、审计和分实验报告均保留在原运行目录；本报告只做汇总，不重采样、不覆盖原始结果。产物清单 `{manifest.relative_to(ROOT)}` 记录每个文件的路径、字节数和 SHA-256。

## 总结结论

{summary_bullets()}

## 原始产物保留

完整产物清单随本报告生成并复制到 `docs/results/`。本地 `data/`、`outputs/`、`archive/` 按 `.gitignore` 不提交到 GitHub。
"""


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
    lines: list[str] = []
    e1 = read_json(OUT / "exp1_input_ablation" / RUN / "metrics.json")
    if e1 and "q+y" in e1 and "y_only" in e1:
        gain = e1["q+y"]["macro_f1"] - e1["y_only"]["macro_f1"]
        lines.append(f"- E1：q+y Macro-F1={e1['q+y']['macro_f1']:.4f}，y-only={e1['y_only']['macro_f1']:.4f}，增益={gain:.4f}。")
    e2 = read_json(OUT / "exp2_prior_work_comparison" / RUN / "metrics.json")
    if e2 is not None:
        lines.append(f"- E2：输出 {len(e2)} 行 proxy/baseline 审计；官方 guard baseline 仍需单独接入。")
    e3 = read_json(OUT / "exp3_agent_distillation_ablation" / RUN / "metrics.json")
    if e3 and e3.get("student"):
        first, last = e3["student"][0], e3["student"][-1]
        lines.append(f"- E3：当前定位为 sample reweighting proxy；{first['Variant']} Macro-F1={first['Macro-F1']}，{last['Variant']} Macro-F1={last['Macro-F1']}。")
    e4 = read_json(OUT / "exp4_unseen" / RUN / "worst_group_metrics.json")
    if e4:
        lines.append(f"- E4：最弱 held-out={e4.get('Held-out', 'unknown')}，Macro-F1={e4.get('Macro-F1', '')}；language holdout 已排除 procedural rows。")
    e5 = read_json(OUT / "exp5_calibration" / RUN / "metrics.json")
    if e5:
        lines.append("- E5：报告 raw/Platt 和 dev-UCB FPR 操作点，calibration/threshold/test split 分离。")
    e6 = read_json(OUT / "exp6_multi_api" / RUN / "metrics.json")
    if e6:
        n_models = len(e6.get("student_vs_pair_silver", []))
        guard = e6.get("guard_consensus_audit", {})
        lines.append(f"- E6：覆盖 {n_models} 个已有目标模型 generation；pair label 为 proxy guard consensus，独立 guard 可用性={guard.get('official_qwen3guard_wildguard_harmbench_available', False)}。")
    if RUN == "protocol_gate_v2":
        lines.append("- Gate：按用户新文档，本轮仍是 NO-GO Full；目标是完成 P0/P1 修复和门控产物，而不是启动完整大样本。")
    return "\n".join(lines) if lines else "- 本轮指标文件尚未生成。"


def assert_publish_consistency(master: Path, manifest: Path, run_id: str) -> None:
    text = master.read_text(encoding="utf-8")
    if run_id == "protocol_gate_v2" and "PROTOCOL_GATE_V2" not in master.name:
        raise RuntimeError("Protocol Gate v2 report filename mismatch")
    if run_id not in text:
        raise RuntimeError(f"master report body does not mention run_id={run_id}")
    if not manifest.exists() or manifest.stat().st_size == 0:
        raise RuntimeError(f"empty artifact manifest: {manifest}")


def write_protocol_gate_v2_sidecars(artifact_rows: list[tuple[str, int, str]]) -> None:
    go_nogo = DOC_RESULTS / "PROTOCOL_GATE_V2_GO_NOGO.csv"
    rows = [
        {"Gate": "Full run", "Status": "NO-GO", "Evidence": "Source decision document requires Protocol Gate v2 before Full."},
        {"Gate": "E3 method scope", "Status": "FIXED-NAMING", "Evidence": "E3 is reported as sample reweighting proxy, not true KD."},
        {"Gate": "E6 labels", "Status": "NO-GO", "Evidence": "Independent Qwen3Guard/WildGuard/HarmBench consensus is unavailable locally."},
        {"Gate": "Artifacts", "Status": "PASS", "Evidence": "Master report, manifest, sidecar audits, and hashes were generated."},
    ]
    with go_nogo.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Gate", "Status", "Evidence"])
        writer.writeheader()
        writer.writerows(rows)
    (DOC_RESULTS / "DATA_LICENSE_MANIFEST.yaml").write_text(
        """datasets:
  - name: Fraud-R1
    source_url: https://github.com/kaustpradalab/Fraud-R1
    raw_data_redistributed: false
    derived_ids_only: true
  - name: Aegis-AI-Content-Safety-Dataset-2.0
    source_url: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
    raw_data_redistributed: false
    derived_ids_only: true
  - name: Do-Not-Answer
    source_url: https://github.com/libr-ai/do-not-answer
    raw_data_redistributed: false
    derived_ids_only: true
""",
        encoding="utf-8",
    )
    (DOC_RESULTS / "DATA_SEMANTICS_AUDIT.md").write_text(
        "# Data Semantics Audit\n\nFraud-R1 generated text is prompt/context material, not target-model answer y. E4 language holdout excludes procedural rows. E6 prompt risk labels are not accepted as response/pair gold.\n",
        encoding="utf-8",
    )
    (DOC_RESULTS / "BASELINE_VERSION_LOCK.yaml").write_text(
        f"baselines:\n  official_guard_status: not_available_in_current_workspace\n  proxy_baselines_are_paper_claims: false\n  artifact_count: {len(artifact_rows)}\n",
        encoding="utf-8",
    )
    (DOC_RESULTS / "GUARD_CONSENSUS_AUDIT.md").write_text(
        "# Guard Consensus Audit\n\nProtocol Gate v2 still uses deterministic proxy guard votes for E6 because frozen Qwen3Guard/WildGuard/HarmBench evaluators are not available in the local runnable environment. Reports mark this as NO-GO for paper-level target-model ranking.\n",
        encoding="utf-8",
    )


def reproduction_doc() -> str:
    return """# Reproduce Six Experiments

```powershell
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_high_standard_rerun.py protocol_gate_v2 --bootstrap 500 --api-provider qwen --api-probe-limit 12
python scripts/write_six_experiment_master_report.py --run-id protocol_gate_v2
```

`data/`, `outputs/`, `archive/`, model files, and `api_keys.py` are local-only and are not committed to GitHub.
"""


if __name__ == "__main__":
    main()
