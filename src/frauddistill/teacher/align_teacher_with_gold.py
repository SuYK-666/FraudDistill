from __future__ import annotations

import argparse

from frauddistill.data.schema import FraudDistillSample, TeacherSignal
from frauddistill.utils.io import read_jsonl, write_jsonl


def align_rows(samples: list[dict], signals: list[dict]) -> list[dict]:
    signal_by_id = {TeacherSignal.model_validate(row).id: TeacherSignal.model_validate(row) for row in signals}
    output = []
    for row in samples:
        sample = FraudDistillSample.model_validate(row)
        item = sample.model_dump(mode="json")
        signal = signal_by_id.get(sample.id)
        if signal:
            teacher = signal.model_dump(mode="json")
            item.update({key: value for key, value in teacher.items() if key != "id"})
            item["teacher_gold_conflict"] = signal.teacher_label.value != sample.gold_label.value
            item["teacher_loss_weight"] = 0.1 if item["teacher_gold_conflict"] else 0.3
        else:
            item["teacher_gold_conflict"] = None
            item["teacher_loss_weight"] = 0.0
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", required=True)
    parser.add_argument("--teacher_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    rows = align_rows(list(read_jsonl(args.samples_file)), list(read_jsonl(args.teacher_file)))
    print(f"wrote {write_jsonl(args.output_file, rows)} rows")


if __name__ == "__main__":
    main()
