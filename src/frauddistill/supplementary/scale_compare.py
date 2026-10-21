"""Build a complete local comparison for all completed scale runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import REPO, binary_label, binary_metrics, read_json, read_jsonl, row_key, write_json

RESULT_ROOT = REPO / "experiments" / "lighteval_supplementary" / "results"
SCALE_ROOT = RESULT_ROOT / "scale"
OUT_ROOT = RESULT_ROOT / "scale_complete"
RAW_ROOT = REPO / "experiments" / "lighteval_supplementary" / "prepared" / "raw"
MODELS = [
    ("0.5B", "qwen25_0_5b"),
    ("1.5B", "qwen25_1_5b"),
    ("3B", "qwen25_3b"),
    ("7B", "qwen25_7b"),
    ("0.6B (Qwen3)", "qwen3_0_6b"),
    ("1.7B (Qwen3)", "qwen3_1_7b"),
    ("4B (Qwen3)", "qwen3_4b"),
    ("0.8B (Qwen3.5)", "qwen35_0_8b"),
    ("2B (Qwen3.5)", "qwen35_2b"),
    ("4B (Qwen3.5)", "qwen35_4b"),
]
OBSERVED_STEPS = {"qwen25_0_5b": 140, "qwen25_1_5b": 296, "qwen25_3b": 280, "qwen25_7b": 304}
NOMINAL_TOTAL_STEPS = 296


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _predictions(model_id: str, name: str) -> dict[str, dict]:
    return {item["id"]: item for item in read_jsonl(SCALE_ROOT / model_id / f"predictions_{name}.jsonl")}


def _gold(name: str) -> dict[str, int]:
    return {row_key(item): binary_label(item) for item in read_jsonl(RAW_ROOT / f"{name}.jsonl")}


def _source_threshold_metrics(model_id: str, name: str, threshold: float) -> dict:
    gold = _gold(name)
    preds = _predictions(model_id, name)
    keys = list(gold)
    return binary_metrics([gold[key] for key in keys],
                          [preds[key]["risk_score"] for key in keys], threshold=threshold)


def _load() -> list[dict]:
    result = []
    for label, model_id in MODELS:
        summary = read_json(SCALE_ROOT / model_id / "summary.json")
        record = read_json(SCALE_ROOT / model_id / "checkpoint_record.json")
        item = {"label": label, "model_id": model_id, "summary": summary, "record": record}
        steps = int(record.get("optimizer_steps_observed") or OBSERVED_STEPS.get(model_id, 304))
        item["observed_optimizer_steps"] = steps
        item["actual_train_input_tokens_estimate"] = round(
            summary["cost"]["training"]["input_tokens"] * steps / NOMINAL_TOTAL_STEPS
        )
        result.append(item)
    return result


def build() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = _load()
    all_json = {}
    for item in items:
        model_id = item["model_id"]
        selected = float(item["summary"]["threshold_selection"]["threshold"])
        source_metrics = {
            "core_test": _source_threshold_metrics(model_id, "core_test", 0.5622),
            "relation_test": _source_threshold_metrics(model_id, "relation_test", 0.5622),
            "hard_safe_test": _source_threshold_metrics(model_id, "hard_safe_test", 0.5622),
        }
        all_json[model_id] = {**item, "source_threshold": 0.5622,
                              "selected_threshold": selected,
                              "source_threshold_metrics": source_metrics}
    write_json(OUT_ROOT / "scale_complete.json", all_json)

    quality = [
        "| Model | Checkpoint | Threshold | Core-dev MF1 | Core MF1 | Core Precision | Core Recall | Core FPR | E6 MF1 | E6 Precision | E6 Recall | E6 FPR | Hard-safe FPR | E6 AUROC | E6 AUPRC | 4-class MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        m = item["summary"]["metrics"]
        core, e6, hs = m["core_test"], m["relation_test_online"], m["hard_safe_test"]
        quality.append(
            f"| {item['label']} | {item['record']['checkpoint_step']} | {item['summary']['threshold_selection']['threshold']:.6f} | "
            f"{m['core_dev']['macro_f1']:.4f} | {core['macro_f1']:.4f} | {core['precision']:.4f} | {core['recall']:.4f} | {core['fpr']:.4f} | "
            f"{e6['macro_f1']:.4f} | {e6['precision']:.4f} | {e6['recall']:.4f} | {e6['fpr']:.4f} | {hs['fpr']:.4f} | "
            f"{e6['auroc']:.4f} | {e6['auprc']:.4f} | {core['four_class']['macro_f1']:.4f} |"
        )
    (OUT_ROOT / "table_quality.md").write_text("\n".join(quality) + "\n", encoding="utf-8")

    confusion = [
        "| Model | Set | TP | FP | FN | TN | Accuracy | Balanced Acc | MCC | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        m = item["summary"]["metrics"]
        for set_label, key in (("Core test", "core_test"), ("E6 online", "relation_test_online"), ("Hard-safe", "hard_safe_test")):
            x = m[key]
            confusion.append(
                f"| {item['label']} | {set_label} | {x['tp']} | {x['fp']} | {x['fn']} | {x['tn']} | "
                f"{x['accuracy']:.4f} | {fmt(x['balanced_accuracy'])} | {fmt(x['mcc'])} | {fmt(x['auroc'])} | {fmt(x['auprc'])} |"
            )
    (OUT_ROOT / "table_confusions.md").write_text("\n".join(confusion) + "\n", encoding="utf-8")

    thresholds = [
        "| Model | Source T | Selected T | Cal Feasible | Cal Relation MF1 | Cal Recall | Cal HS FPR | Selected E6 MF1 | Source-T E6 MF1 | Selected Core MF1 | Source-T Core MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        rec = item["record"]["threshold_selection"]
        cal = rec.get("calibration") or {}
        selected = item["summary"]["metrics"]
        source = all_json[item["model_id"]]["source_threshold_metrics"]
        thresholds.append(
            f"| {item['label']} | 0.5622 | {rec['threshold']:.6f} | {rec['feasible']} | {fmt(cal.get('relation_macro_f1'))} | "
            f"{fmt(cal.get('relation_recall'))} | {fmt(cal.get('hard_safe_fpr'))} | {selected['relation_test_online']['macro_f1']:.4f} | "
            f"{source['relation_test']['macro_f1']:.4f} | {selected['core_test']['macro_f1']:.4f} | {source['core_test']['macro_f1']:.4f} |"
        )
    (OUT_ROOT / "table_thresholds.md").write_text("\n".join(thresholds) + "\n", encoding="utf-8")

    costs = [
        "| Model | Optimizer Steps | Train s | Train h | Train Peak VRAM GB | Processed Train Tokens (est.) | Eval s | Eval Peak VRAM GB | Eval Input Tokens | P50 ms | P95 ms | Throughput Rows/s | Sec/1k Rows | Total GPU h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_set = [
        "| Model | Set | Rows | Seconds | Rows/s | Input Tokens | Output Tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        cost = item["summary"]["cost"]
        train, ev = cost["training"], cost["evaluation"]
        serve = ev["serving"]
        costs.append(
            f"| {item['label']} | {item['observed_optimizer_steps']} | {train['wall_seconds']} | {train['wall_hours']:.3f} | {fmt(train['peak_vram_gb'])} | {item['actual_train_input_tokens_estimate']} | "
            f"{ev['wall_seconds']} | {fmt(ev['peak_vram_gb'])} | {ev['total_input_tokens']} | {fmt(serve['latency_p50_ms'], 2)} | "
            f"{fmt(serve['latency_p95_ms'], 2)} | {fmt(serve['rows_per_s'], 2)} | {fmt(serve['seconds_per_1000_rows'], 2)} | {cost['total_gpu_hours']:.4f} |"
        )
        for name, row in ev["per_set"].items():
            per_set.append(
                f"| {item['label']} | {name} | {row['rows']} | {row['seconds']} | {row['rows_per_s']} | {row['input_tokens']} | {row['output_tokens']} |"
            )
    costs.extend([
        "",
        "> Train s includes model reload, resume and CUDA-retry overhead; endpoint checkpoint saving is excluded. Qwen2.5-7B used 3 attempts/2 CUDA retries. Qwen3.5-4B used 9 attempts/8 retries and official evaluation batch=2. Qwen3.5-2B/4B use the training-best proxy checkpoint because full candidate selection was unavailable. Qwen3.5 latency/throughput use unoptimized reference kernels."
    ])
    (OUT_ROOT / "table_costs.md").write_text("\n".join(costs) + "\n", encoding="utf-8")
    (OUT_ROOT / "table_per_set_cost.md").write_text("\n".join(per_set) + "\n", encoding="utf-8")

    selections = [
        "| Model | Checkpoint Step | Checkpoint SHA256 | Candidates | Selected Full-dev MF1 | Recall | FPR | AUPRC | MCC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        selection = item["summary"].get("selection") or {}
        full = (selection.get("selected") or {}).get("full_metrics") or {}
        selections.append(
            f"| {item['label']} | {item['record']['checkpoint_step']} | `{item['record']['checkpoint_sha256']}` | "
            f"{selection.get('candidate_count')} | {fmt(full.get('macro_f1'))} | {fmt(full.get('recall'))} | {fmt(full.get('fpr'))} | "
            f"{fmt(full.get('auprc'))} | {fmt(full.get('mcc'))} |"
        )
    (OUT_ROOT / "table_selection.md").write_text("\n".join(selections) + "\n", encoding="utf-8")

    # Flat CSV for downstream plotting and paper-table assembly.
    fields = ["model_id", "model_label", "checkpoint_step", "checkpoint_sha256", "threshold"]
    for prefix in ("core_dev", "core_test", "relation_test_online", "hard_safe_test"):
        for key in ("macro_f1", "precision", "recall", "fpr", "accuracy", "balanced_accuracy", "mcc", "auroc", "auprc", "tp", "fp", "fn", "tn"):
            fields.append(f"{prefix}_{key}")
    fields += ["four_class_macro_f1", "observed_optimizer_steps", "train_wall_seconds", "train_peak_vram_gb", "train_tokens",
               "eval_wall_seconds", "eval_peak_vram_gb", "eval_input_tokens", "latency_p50_ms",
               "latency_p95_ms", "throughput_rows_s", "seconds_per_1000_rows", "total_gpu_hours"]
    with (OUT_ROOT / "scale_complete_flat.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            record = item["record"]
            m, cost = item["summary"]["metrics"], item["summary"]["cost"]
            flat = {"model_id": item["model_id"], "model_label": item["label"],
                    "checkpoint_step": record["checkpoint_step"], "checkpoint_sha256": record["checkpoint_sha256"],
                    "threshold": item["summary"]["threshold_selection"]["threshold"],
                    "four_class_macro_f1": m["core_test"]["four_class"]["macro_f1"],
                    "observed_optimizer_steps": item["observed_optimizer_steps"],
                    "train_wall_seconds": cost["training"]["wall_seconds"],
                    "train_peak_vram_gb": cost["training"]["peak_vram_gb"],
                    "train_tokens": item["actual_train_input_tokens_estimate"],
                    "eval_wall_seconds": cost["evaluation"]["wall_seconds"],
                    "eval_peak_vram_gb": cost["evaluation"]["peak_vram_gb"],
                    "eval_input_tokens": cost["evaluation"]["total_input_tokens"],
                    "latency_p50_ms": cost["evaluation"]["serving"]["latency_p50_ms"],
                    "latency_p95_ms": cost["evaluation"]["serving"]["latency_p95_ms"],
                    "throughput_rows_s": cost["evaluation"]["serving"]["rows_per_s"],
                    "seconds_per_1000_rows": cost["evaluation"]["serving"]["seconds_per_1000_rows"],
                    "total_gpu_hours": cost["total_gpu_hours"]}
            for prefix in ("core_dev", "core_test", "relation_test_online", "hard_safe_test"):
                for key in ("macro_f1", "precision", "recall", "fpr", "accuracy", "balanced_accuracy", "mcc", "auroc", "auprc", "tp", "fp", "fn", "tn"):
                    flat[f"{prefix}_{key}"] = m[prefix].get(key)
            writer.writerow(flat)

    return {"output": str(OUT_ROOT), "models": [item["model_id"] for item in items]}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = build()
    print(json.dumps({"status": "READY", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
