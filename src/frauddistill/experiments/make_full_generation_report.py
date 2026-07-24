from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from frauddistill.utils.io import read_jsonl


def make_generation_report(generation_dir: str | Path, manifest_file: str | Path, output_dir: str | Path, prompt_file: str | Path | None = None) -> dict[str, Any]:
    gen_dir = Path(generation_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    success_file = gen_dir / "generations_success.jsonl"
    failure_file = gen_dir / "generation_failures.jsonl"
    all_file = gen_dir / "generations.jsonl"
    rows = list(read_jsonl(success_file)) if success_file.exists() else []
    failures = list(read_jsonl(failure_file)) if failure_file.exists() else []
    prompt_by_id = {row["id"]: row for row in read_jsonl(prompt_file)} if prompt_file else {}
    rows = [_enrich_generation(row, prompt_by_id) for row in rows]
    failures = [_enrich_generation(row, prompt_by_id) for row in failures]
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))

    model_rows = _model_summary(rows, failures)
    source_rows = _group_summary(rows, "source")
    prior_rows = _metadata_or_field_summary(rows, "source_prior")
    category_rows = _metadata_or_field_summary(rows, "fraud_category")

    _write_csv(out / "generation_by_model.csv", model_rows)
    _write_csv(out / "generation_by_source.csv", source_rows)
    _write_csv(out / "generation_by_prior.csv", prior_rows)
    _write_csv(out / "generation_by_category.csv", category_rows)

    report = {
        "generation_dir": str(gen_dir),
        "all_file": str(all_file),
        "success_file": str(success_file),
        "failure_file": str(failure_file),
        "prompt_file": str(prompt_file) if prompt_file else None,
        "expected_rows": manifest.get("totals", {}).get("prompt_rows", 0) * 2,
        "success_rows": len(rows),
        "failure_rows": len(failures),
        "model_summary": model_rows,
        "source_summary": source_rows,
        "prior_summary": prior_rows,
        "category_summary": category_rows,
    }
    (out / "GENERATION_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "GENERATION_REPORT.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _model_summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fail_by_model = Counter(str(row.get("target_model", "unknown")) for row in failures)
    for row in rows:
        by_model[str(row.get("target_model", "unknown"))].append(row)
    records = []
    for model, sub in sorted(by_model.items()):
        records.append(
            {
                "target_model": model,
                "success_rows": len(sub),
                "failure_rows": fail_by_model.get(model, 0),
                "avg_latency_seconds": _avg(float(row.get("latency_seconds", 0.0)) for row in sub),
                "avg_answer_chars": _avg(len(str(row.get("target_model_answer", ""))) for row in sub),
                "empty_answers": sum(1 for row in sub if not row.get("target_model_answer")),
            }
        )
    for model, count in sorted(fail_by_model.items()):
        if model not in by_model:
            records.append({"target_model": model, "success_rows": 0, "failure_rows": count, "avg_latency_seconds": 0.0, "avg_answer_chars": 0.0, "empty_answers": 0})
    return records


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return [_summary_record(key, value, sub) for value, sub in sorted(grouped.items())]


def _metadata_or_field_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, (row.get("metadata") or {}).get(key, "unknown"))
        grouped[str(value)].append(row)
    return [_summary_record(key, value, sub) for value, sub in sorted(grouped.items())]


def _summary_record(key: str, value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: value,
        "rows": len(rows),
        "qwen_rows": sum(1 for row in rows if row.get("target_model") == "qwen-plus"),
        "deepseek_rows": sum(1 for row in rows if row.get("target_model") == "deepseek-chat"),
        "avg_latency_seconds": _avg(float(row.get("latency_seconds", 0.0)) for row in rows),
        "avg_answer_chars": _avg(len(str(row.get("target_model_answer", ""))) for row in rows),
    }


def _enrich_generation(row: dict[str, Any], prompt_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_prompt_id = str((row.get("metadata") or {}).get("source_prompt_id", ""))
    prompt = prompt_by_id.get(source_prompt_id, {})
    enriched = dict(row)
    for key in ["source_prior", "fraud_category", "fraud_subcategory", "prompt_setting"]:
        if enriched.get(key) in {None, "", "unknown"}:
            enriched[key] = prompt.get(key, (prompt.get("metadata") or {}).get(key, "unknown"))
    return enriched


def _avg(values) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Full Target Generation Report",
        "",
        f"Generation directory: `{report['generation_dir']}`",
        "",
        "## Completeness",
        "",
        f"- Expected rows: {report['expected_rows']}",
        f"- Successful rows: {report['success_rows']}",
        f"- Failed rows: {report['failure_rows']}",
        "",
        "## Model Summary",
        "",
        "| Model | Success | Failure | Avg Latency | Avg Answer Chars |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["model_summary"]:
        lines.append(
            f"| `{row['target_model']}` | {row['success_rows']} | {row['failure_rows']} | "
            f"{row['avg_latency_seconds']:.4f} | {row['avg_answer_chars']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Data And Method",
            "",
            "The generation step uses the complete prepared prompt pool. Each prompt is sent once to Qwen and once to DeepSeek with identical generation parameters. Outputs are persisted as JSONL with model id, provider, latency, timestamp, response hash, and API error fields.",
            "",
            "## Expectedness Check",
            "",
            "The run is considered structurally valid if successful rows plus failed rows match the expected row count, each model has comparable coverage, and failed/empty responses are isolated in `generation_failures.jsonl` for reruns.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", required=True)
    parser.add_argument("--manifest_file", default="data/prepared/full/MANIFEST.json")
    parser.add_argument("--prompt_file", default="data/prepared/full/prompts/all_target_prompts.jsonl")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = make_generation_report(args.generation_dir, args.manifest_file, args.output_dir, args.prompt_file)
    print(json.dumps({"output_dir": args.output_dir, "success_rows": report["success_rows"], "failure_rows": report["failure_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
