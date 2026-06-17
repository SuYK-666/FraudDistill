from __future__ import annotations

from collections import Counter


def teacher_quality_report(samples: list[dict], signals: list[dict]) -> dict:
    gold_by_id = {row["id"]: row["gold_label"] for row in samples}
    counts = Counter()
    scores = []
    for signal in signals:
        sample_id = signal["id"]
        if sample_id not in gold_by_id:
            counts["missing_gold"] += 1
            continue
        counts["total"] += 1
        counts["agree"] += int(signal["teacher_label"] == gold_by_id[sample_id])
        scores.append(float(signal["teacher_score"]))
        counts["empty_span_unsafe"] += int(signal["teacher_label"] == "unsafe" and not signal.get("teacher_spans"))
    total = counts["total"] or 1
    return {
        "total": counts["total"],
        "agreement": counts["agree"] / total,
        "conflict_rate": 1.0 - counts["agree"] / total,
        "avg_teacher_score": sum(scores) / len(scores) if scores else 0.0,
        "empty_span_rate_unsafe": counts["empty_span_unsafe"] / total,
        "missing_gold": counts["missing_gold"],
    }
