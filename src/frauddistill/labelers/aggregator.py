from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from frauddistill.utils.io import read_jsonl, write_jsonl


def aggregate_votes(vote_rows: list[dict[str, Any]], *, unsafe_threshold: float = 0.60, safe_threshold: float = 0.40) -> dict[str, Any]:
    if not vote_rows:
        raise ValueError("vote_rows must not be empty")
    sample_id = str(vote_rows[0]["id"])
    scores = [_score(row) for row in vote_rows]
    labels = ["unsafe" if score >= 0.5 else "safe" for score in scores]
    risk_score = mean(scores)
    unsafe_votes = labels.count("unsafe")
    safe_votes = labels.count("safe")
    if risk_score >= unsafe_threshold:
        binary = "unsafe"
    elif risk_score <= safe_threshold:
        binary = "safe"
    else:
        binary = "ambiguous"
    confidence = _confidence(binary, risk_score, unsafe_votes, safe_votes, len(vote_rows))
    return {
        "id": sample_id,
        "auto_label": {
            "binary": binary,
            "risk_type": _risk_type(vote_rows, binary),
            "risk_score": risk_score,
            "confidence": confidence,
            "teacher_votes": {str(row.get("labeler", row.get("teacher_name", "unknown"))): row.get("label", row.get("teacher_label")) for row in vote_rows},
            "vote_count": len(vote_rows),
            "unsafe_votes": unsafe_votes,
            "safe_votes": safe_votes,
        },
    }


def aggregate_vote_file(input_file: str | Path, output_file: str | Path) -> int:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(input_file):
        grouped.setdefault(str(row["id"]), []).append(row)
    return write_jsonl(output_file, (aggregate_votes(rows) for rows in grouped.values()))


def _score(row: dict[str, Any]) -> float:
    for key in ("score", "teacher_score", "pred_score", "risk_score"):
        if key in row:
            try:
                return max(0.0, min(1.0, float(row[key])))
            except (TypeError, ValueError):
                pass
    label = str(row.get("label", row.get("teacher_label", row.get("pred_label", "safe"))))
    return 0.75 if label == "unsafe" else 0.25


def _confidence(binary: str, risk_score: float, unsafe_votes: int, safe_votes: int, vote_count: int) -> str:
    agreement = max(unsafe_votes, safe_votes) / max(vote_count, 1)
    margin = abs(risk_score - 0.5)
    if binary == "ambiguous":
        return "ambiguous"
    if vote_count >= 3 and agreement >= 0.75 and margin >= 0.25:
        return "silver_high"
    if agreement >= 0.60 and margin >= 0.10:
        return "silver_medium"
    return "ambiguous"


def _risk_type(vote_rows: list[dict[str, Any]], binary: str) -> str:
    if binary == "safe":
        return "safe"
    counts: dict[str, int] = {}
    for row in vote_rows:
        risk_type = str(row.get("risk_type", row.get("teacher_type", row.get("pred_type", "fraud_assistance"))))
        if risk_type in {"safe", "none"}:
            continue
        counts[risk_type] = counts.get(risk_type, 0) + 1
    if not counts:
        return "fraud_assistance"
    return max(counts.items(), key=lambda item: item[1])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    count = aggregate_vote_file(args.input_file, args.output_file)
    print(f"wrote {count} silver labels to {args.output_file}")


if __name__ == "__main__":
    main()
