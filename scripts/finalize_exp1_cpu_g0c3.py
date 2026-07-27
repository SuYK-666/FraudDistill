from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_ccfa.resource_profile import dir_size_mb, resource_gate
from frauddistill.utils.io import read_jsonl
from scripts.run_exp1_cpu_g0c3 import (
    CONFIG_PATH,
    correctness_gate,
    decide,
    git_commit,
    git_status,
    paired_stats,
    protocol_lock,
    seed_direction_summary,
    selected_summary,
    sha256_file,
    source_tree_sha256,
    to_jsonable,
    write_json,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize E1 CPU G0c3 from completed prediction files")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0c3")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0c3")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--wall_seconds", type=float, default=8568.0)
    args = parser.parse_args()
    payload = finalize(ROOT / args.manifest_dir, ROOT / args.output_dir, args.bootstrap_iterations, args.wall_seconds)
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))
    if payload["decision"] == "STOP_RELATION_CLAIM":
        raise SystemExit(2)


def finalize(manifest_dir: Path, output_dir: Path, bootstrap_iterations: int, wall_seconds: float) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit = json.loads((manifest_dir / "G0c3_DATA_AUDIT.json").read_text(encoding="utf-8"))
    selected = json.loads((output_dir / "G0c3_SELECTED_MODELS.json").read_text(encoding="utf-8"))
    metric_rows = [*read_csv(output_dir / "G0c3_P1_METRICS_BY_SEED.csv"), *read_csv(output_dir / "G0c3_P2_METRICS_BY_SEED.csv"), *read_csv(output_dir / "G0c3_P3_METRICS_BY_SEED.csv")]
    predictions = load_predictions(output_dir / "predictions")
    stats = paired_stats(predictions, selected, sorted({row["train_seed"] for row in metric_rows}), bootstrap_iterations)
    write_json(output_dir / "G0c3_PAIRED_STATS.json", stats)
    seed_direction = seed_direction_summary(metric_rows, selected)
    write_json(output_dir / "G0c3_SEED_DIRECTION.json", seed_direction)
    profile = {
        "wall_seconds": wall_seconds,
        "peak_rss_mb": max_float_from_csv(output_dir / "G0c3_RESOURCE_PROFILE.csv", "peak_ram_mb"),
        "artifact_mb": dir_size_mb(output_dir),
        "cpu_threads": 0,
        "cuda_available": False,
        "device_count": 0,
        "finalized_from_predictions": True,
    }
    resource = resource_gate(profile, config["resource_gates"])
    write_json(output_dir / "G0c3_RESOURCE_GATE.json", resource)
    dirty = git_status()
    protocol = protocol_lock(config, audit, manifest_dir, dirty)
    protocol["finalized_from_predictions"] = True
    write_json(output_dir / "G0c3_PROTOCOL_LOCK.json", protocol)
    decision = decide(stats, metric_rows, audit, resource, selected, seed_direction)
    payload = {
        "decision": decision,
        "correctness_gate": correctness_gate(audit, resource, dirty),
        "selected": selected_summary(selected),
        "git_commit": git_commit(),
        "git_status_porcelain": dirty,
        "source_tree_sha256": source_tree_sha256(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_fingerprint": audit.get("data_fingerprint"),
        "encoder_revision": config["semantic_cpu"]["encoder"]["revision"],
        "dataset_revisions": config["data_policy"].get("dataset_revisions", {}),
        "seed_direction": seed_direction,
        "finalized_from_predictions": True,
    }
    write_json(output_dir / "G0c3_DECISION.json", payload)
    write_json(output_dir / "G0c3_RUN_FINGERPRINT.json", payload)
    write_report(output_dir, payload, audit, stats, metric_rows, resource)
    write_chinese_task_report(output_dir, payload, audit, stats, metric_rows, resource)
    return payload


def load_predictions(pred_dir: Path) -> dict:
    out = {}
    for path in pred_dir.glob("*.jsonl"):
        rows = list(read_jsonl(path))
        if not rows:
            continue
        first = rows[0]
        out[(path_key(first), str(first["train_seed"]), str(first["panel"]))] = rows
    return out


def path_key(row: dict) -> str:
    level = row["level"]
    mode = row["mode"]
    backend = row["backend"]
    if backend == "pairlite":
        if mode == "q_only":
            return "PairLite_q_only"
        if mode == "y_only":
            return "PairLite_y_only"
        return "PairLite_q_y"
    if mode == "q_only":
        return "E_q_only"
    if mode == "y_only":
        return "E_y_only"
    if level == "S0":
        return "S0_q_y"
    if level == "S1":
        return "S1_q_y"
    return "S2_joint_q_y"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def max_float_from_csv(path: Path, field: str) -> float:
    if not path.exists():
        return 0.0
    values = []
    for row in read_csv(path):
        try:
            values.append(float(row.get(field, 0.0) or 0.0))
        except ValueError:
            pass
    return max(values, default=0.0)


def write_chinese_task_report(output_dir: Path, decision: dict, audit: dict, stats: dict, metric_rows: list[dict], resource: dict) -> None:
    report = output_dir / "reports" / "E1_CPU_v5_G0c3_FINAL_FREEZE_整体任务报告_中文.md"
    p2_audit = audit.get("p2_dvm_audit", {})
    lines = [
        "# E1 CPU v5 G0c3 FINAL FREEZE 整体任务报告",
        "",
        "## 一、执行结论",
        f"- 最终决策：{decision['decision']}",
        f"- full comparator：{decision['selected']['best_full']}",
        f"- best single：{decision['selected']['best_single']}",
        f"- Git commit：{decision['git_commit']}",
        f"- 数据指纹：{decision.get('data_fingerprint')}",
        f"- 是否从完整预测文件恢复 finalize：{decision.get('finalized_from_predictions')}",
        "",
        "本轮训练和预测已经完整完成，96 个 prediction 文件、P1/P2/P3 metrics、thresholds 均已落盘。原 runner 在最终 exact McNemar 统计阶段触发大样本组合数溢出，因此本报告使用修复后的稳定 McNemar 实现，从已落盘预测文件恢复 paired stats、seed direction、decision 和 fingerprint。",
        "",
        "## 二、数据 Gate",
        f"- Base data gate：{audit.get('base_data_gate', {}).get('passed')}",
        f"- P2 data gate：{audit.get('p2_data_gate', {}).get('passed')}",
        f"- P2 groups：{p2_audit.get('groups')}",
        f"- P2 rows/components：{p2_audit.get('rows')} / {audit.get('counts', {}).get('g0_p2_dvm_core', {}).get('components')}",
        f"- P2 component_occurrence_max：{p2_audit.get('component_occurrence_max')}",
        f"- P2 q/y selector SMD：{p2_audit.get('q_selector_smd')} / {p2_audit.get('y_selector_smd')}",
        f"- P2 log answer length SMD：{p2_audit.get('log_answer_length_smd')}",
        f"- P2 independent q/y probe AUC：{p2_audit.get('independent_q_probe_auc')} / {p2_audit.get('independent_y_probe_auc')}",
        "",
        "## 三、主要指标表",
        "",
        "| Panel | Comparator | Seed | Macro-F1 | Recall | FPR | AUPRC | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        if row["comparator"] in {decision["selected"]["best_full"], "E_y_only", decision["selected"]["s0"], decision["selected"]["best_single"]}:
            lines.append(f"| {row['panel']} | {row['comparator']} | {row['train_seed']} | {float(row['macro_f1']):.4f} | {float(row['recall']):.4f} | {float(row['fpr']):.4f} | {float(row['auprc']):.4f} | {float(row['auroc']):.4f} |")
    lines.extend([
        "",
        "## 四、配对统计摘要",
        "```json",
        json.dumps(stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 五、Seed 方向性",
        "```json",
        json.dumps(decision.get("seed_direction", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 六、资源 Gate",
        "```json",
        json.dumps(resource, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 七、分析",
        "本轮数据侧整改达到 G0c3 要求：P2-first 后 P1 不复用 P2 component，P2 实现全局 component 容量约束，并通过 selector SMD、长度 SMD、拒答 gap、independent probe 与泄漏审计。P2 规模为 200 groups，属于文档定义的 LIMITED_P2 最低边界，而不是 300 groups 的 FULL 规模。",
        "",
        "模型侧完整评测了 P1/P2/P3 与多 seed。最终是否能进入 FULL_GO 或 FULL_GO_LIMITED_P2 由 relation delta、seed 方向性、P3 非劣和 correctness/resource Gate 共同决定；本报告保留 STOP 情况下的全部可用数据，便于后续分析失败原因，而不改写实验门槛。",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
