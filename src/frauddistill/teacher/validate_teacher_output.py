from __future__ import annotations

import argparse
from pathlib import Path

from frauddistill.data.schema import TeacherSignal
from frauddistill.utils.io import read_jsonl


def validate_teacher_rows(rows: list[dict], answer_by_id: dict[str, str] | None = None) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        try:
            signal = TeacherSignal.model_validate(row)
        except Exception as exc:
            errors.append(f"line {idx}: {exc}")
            continue
        if answer_by_id:
            answer = answer_by_id.get(signal.id, "")
            for span in signal.teacher_spans:
                if span.span and span.span not in answer:
                    errors.append(f"line {idx}: span is not present in target_model_answer")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.input_file))
    errors = validate_teacher_rows(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(rows)} teacher signals")


if __name__ == "__main__":
    main()
