from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from frauddistill.utils.io import read_jsonl


def make_target_safety_report(generations_file: str | Path, silver_file: str | Path, output_dir: str | Path, prompt_file: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generations = list(read_jsonl(generations_file))
    silver_by_id = {row["id"]: row["auto_label"] for row in read_jsonl(silver_file)}
    prompt_by_id = {row["id"]: row for row in read_jsonl(prompt_file)} if prompt_file else {}
    joined = [_enrich_generation(row, prompt_by_id, silver_by_id) for row in generations]

    model_rows = _summary(joined, ["target_model"])
    source_rows = _summary(joined, ["target_model", "source"])
    prior_rows = _summary(joined, ["target_model", "source_prior"])
    category_rows = _summary(joined, ["target_model", "fraud_category"])
    confidence_rows = _confidence_summary(joined)

    _write_csv(out / "target_safety_by_model.csv", model_rows)
    _write_csv(out / "target_safety_by_source.csv", source_rows)
    _write_csv(out / "target_safety_by_prior.csv", prior_rows)
    _write_csv(out / "target_safety_by_category.csv", category_rows)
    _write_csv(out / "target_safety_confidence.csv", confidence_rows)

    report = {
        "generations_file": str(generations_file),
        "silver_file": str(silver_file),
        "prompt_file": str(prompt_file) if prompt_file else None,
        "rows": len(joined),
        "model_summary": model_rows,
        "source_summary": source_rows,
        "prior_summary": prior_rows,
        "category_summary": category_rows,
        "confidence_summary": confidence_rows,
    }
    (out / "TARGET_SAFETY_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "TARGET_SAFETY_REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_value(row, key) for key in keys)].append(row)
    records = []
    for values, sub in sorted(grouped.items()):
        labels = [str((row.get("auto_label") or {}).get("binary", "missing")) for row in sub]
        risk_scores = [float((row.get("auto_label") or {}).get("risk_score", 0.5)) for row in sub if (row.get("auto_label") or {}).get("risk_score") is not None]
        record = {key: value for key, value in zip(keys, values, strict=True)}
        n = len(sub)
        unsafe = labels.count("unsafe")
        safe = labels.count("safe")
        ambiguous = labels.count("ambiguous")
        missing = labels.count("missing")
        record.update(
            {
                "n": n,
                "unsafe": unsafe,
                "safe": safe,
                "ambiguous": ambiguous,
                "missing_label": missing,
                "unsafe_rate": round(unsafe / n, 6) if n else 0.0,
                "safe_rate": round(safe / n, 6) if n else 0.0,
                "ambiguous_rate": round(ambiguous / n, 6) if n else 0.0,
                "avg_risk_score": round(sum(risk_scores) / len(risk_scores), 6) if risk_scores else 0.0,
                "avg_latency_seconds": round(sum(float(row.get("latency_seconds", 0.0)) for row in sub) / n, 6) if n else 0.0,
                "avg_answer_chars": round(sum(len(str(row.get("target_model_answer", ""))) for row in sub) / n, 2) if n else 0.0,
            }
        )
        records.append(record)
    return records


def _confidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(str(row.get("target_model", "unknown")), str((row.get("auto_label") or {}).get("confidence", "missing")))] += 1
    return [{"target_model": model, "confidence": confidence, "n": n} for (model, confidence), n in sorted(grouped.items())]


def _enrich_generation(row: dict[str, Any], prompt_by_id: dict[str, dict[str, Any]], silver_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_prompt_id = str((row.get("metadata") or {}).get("source_prompt_id", ""))
    prompt = prompt_by_id.get(source_prompt_id, {})
    enriched = dict(row)
    for key in ["source_prior", "fraud_category", "fraud_subcategory", "prompt_setting"]:
        if enriched.get(key) in {None, "", "unknown"}:
            enriched[key] = prompt.get(key, (prompt.get("metadata") or {}).get(key, "unknown"))
    enriched["auto_label"] = silver_by_id.get(row["id"], {})
    return enriched


def _value(row: dict[str, Any], key: str) -> str:
    if key in row:
        return str(row.get(key, "unknown"))
    return str((row.get("metadata") or {}).get(key, "unknown"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full Target LLM Safety Report",
        "",
        "This report summarizes automatic silver-label evaluation of target LLM outputs. It does not reproduce high-risk prompt or answer text.",
        "",
        f"Rows: {report['rows']}",
        "",
        "## Main Table: By Target Model",
        "",
        "| Target Model | N | Unsafe | Safe | Ambiguous | Unsafe Rate | Avg Risk | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["model_summary"]:
        lines.append(
            f"| `{row['target_model']}` | {row['n']} | {row['unsafe']} | {row['safe']} | {row['ambiguous']} | "
            f"{row['unsafe_rate']:.4f} | {row['avg_risk_score']:.4f} | {row['avg_latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Each target model response is paired with its original prompt and passed through the automatic label pipeline. The current local pipeline includes refusal rules and the offline FraudDistill teacher. API judge and guard votes should be added before freezing final paper claims.",
            "",
            "## Expectedness Checks",
            "",
            "- Fraud-inducement prompts should have lower unsafe rates for safer target models.",
            "- Benign hard-safe and anti-fraud education prompts should have low unsafe rates; high rates indicate detector over-sensitivity.",
            "- A high ambiguous rate indicates insufficient agreement among automatic labelers and should be excluded from main paper metrics or resolved with additional judge/guard votes.",
            "",
            "## Raw Tables",
            "",
            "- `target_safety_by_model.csv`",
            "- `target_safety_by_source.csv`",
            "- `target_safety_by_prior.csv`",
            "- `target_safety_by_category.csv`",
            "- `target_safety_confidence.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--silver_file", required=True)
    parser.add_argument("--prompt_file", default="data/prepared/full/prompts/all_target_prompts.jsonl")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = make_target_safety_report(args.generations_file, args.silver_file, args.output_dir, args.prompt_file)
    print(json.dumps({"output_dir": args.output_dir, "rows": report["rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
