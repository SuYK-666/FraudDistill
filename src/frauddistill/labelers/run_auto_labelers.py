from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from tqdm import tqdm

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.labelers.refusal_rules import detect_refusal
from frauddistill.utils.io import read_jsonl, write_jsonl


def run_auto_labelers(input_file: str | Path, output_file: str | Path, *, limit: int | None = None) -> int:
    rows = list(read_jsonl(input_file))
    if limit is not None:
        rows = rows[:limit]
    teacher = MultiAgentTeacher(client=None)
    votes: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="running automatic labelers"):
        q = str(row.get("user_query", ""))
        y = str(row.get("target_model_answer", ""))
        refusal = detect_refusal(q, y)
        votes.append({"id": row["id"], **refusal})
        signal = teacher.run(row)
        votes.append(
            {
                "id": row["id"],
                "labeler": "frauddistill_teacher_offline",
                "label": signal["teacher_label"],
                "score": signal["teacher_score"],
                "risk_type": signal["teacher_type"],
                "rationale": signal.get("teacher_rationale", ""),
            }
        )
    return write_jsonl(output_file, votes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = run_auto_labelers(args.input_file, args.output_file, limit=args.limit)
    print(f"wrote {count} labeler votes to {args.output_file}")


if __name__ == "__main__":
    main()
