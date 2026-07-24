from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.data.build_context_pairs import build_context_pairs
from frauddistill.data.group_split import grouped_train_dev_test_split, write_split_audit
from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.paired_compare import mcnemar_exact
from frauddistill.eval.threshold_selection import select_fpr_constrained_threshold
from frauddistill.student.pair_tfidf import PairTfidfDetector
from frauddistill.target_llm.openai_client import OpenAIJsonClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key
from frauddistill.utils.io import read_jsonl, write_jsonl


INPUTS = ["data/prepared/full/evaluation_qy/fraudr1_all_categories_qy.jsonl", "data/prepared/full/evaluation_qy/aegis_qy.jsonl", "data/prepared/full/evaluation_qy/do_not_answer_qy.jsonl"]


def rows_from_inputs(limit: int | None) -> list[dict]:
    rows, seen = [], set()
    for file_name in INPUTS:
        path = Path(file_name)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("gold_label") not in {"safe", "unsafe"}:
                continue
            identity = (str(row.get("source")), str(row.get("id")))
            if identity in seen:
                continue
            seen.add(identity)
            row = dict(row)
            row.setdefault("reference_type", "official_gold")
            row.setdefault("reference_source", row.get("source"))
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def archive_previous() -> Path:
    source = Path("outputs")
    target = Path("archive") / f"pre_remediation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=False)
    for directory in source.glob("*"):
        if directory.name == "HIGH_STANDARD_RERUN_SUMMARY.md" or not directory.is_dir():
            continue
        shutil.move(str(directory), str(target / directory.name))
    return target


def run(stage: str, limit: int | None, provider: str | None, archive: bool) -> dict:
    if archive:
        archived = str(archive_previous())
    else:
        archived = None
    run_id = f"remediation_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = Path("outputs") / "remediation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows = rows_from_inputs(limit)
    if len(rows) < 20:
        raise RuntimeError("insufficient public labelled rows")
    splits = grouped_train_dev_test_split(rows, seed=20260723)
    for name, values in splits.items():
        write_jsonl(output / f"{name}.jsonl", values)
    audit = write_split_audit(splits, output / "split_audit.json")
    context_pairs = build_context_pairs(splits["train"])
    write_jsonl(output / "context_pairs.jsonl", context_pairs)

    detector = PairTfidfDetector(max_features=80000, C=1.2).fit(splits["train"], [row["gold_label"] for row in splits["train"]])
    results, prediction_rows = {}, []
    for mode in ("q_only", "y_only", "q_y"):
        dev_scores = detector.predict_proba(splits["dev"], mode).tolist()
        threshold = select_fpr_constrained_threshold([r["gold_label"] for r in splits["dev"]], dev_scores, max_fpr=0.10)
        test_scores = detector.predict_proba(splits["test"], mode).tolist()
        predictions = ["unsafe" if score >= threshold["threshold"] else "safe" for score in test_scores]
        results[mode] = {**binary_metrics([r["gold_label"] for r in splits["test"]], predictions, test_scores), "threshold": threshold}
        prediction_rows.extend({"id": row["id"], "mode": mode, "gold_label": row["gold_label"], "pred_label": prediction, "pred_score": score} for row, prediction, score in zip(splits["test"], predictions, test_scores))
    qy = [row for row in prediction_rows if row["mode"] == "q_y"]
    best_single = "q_only" if results["q_only"]["macro_f1"] >= results["y_only"]["macro_f1"] else "y_only"
    single = [row for row in prediction_rows if row["mode"] == best_single]
    paired = mcnemar_exact([row["gold_label"] for row in qy], [row["pred_label"] for row in qy], [row["pred_label"] for row in single])
    results["acceptance"] = {"best_single": best_single, "macro_f1_gain": results["q_y"]["macro_f1"] - results[best_single]["macro_f1"], "recall_gain": results["q_y"]["recall"] - results[best_single]["recall"], "fpr_delta": results["q_y"]["fpr"] - results[best_single]["fpr"], "mcnemar": paired}
    write_jsonl(output / "predictions.jsonl", prediction_rows)

    teacher_summary = None
    if provider:
        config = get_provider_config(provider)
        require_api_key(config)
        teacher = MultiAgentTeacher(OpenAIJsonClient(config.default_model, config.api_key, config.base_url, timeout=90.0))
        teacher_rows = [teacher.run(row) for row in splits["dev"]]
        write_jsonl(output / "teacher_dev.jsonl", teacher_rows)
        statuses = Counter(str(row.get("status")) for row in teacher_rows)
        usable = [row for row in teacher_rows if row.get("status") == "ok"]
        teacher_summary = {"provider": config.name, "model": config.default_model, "total": len(teacher_rows), "status": dict(statuses), "coverage": len(usable) / max(1, len(teacher_rows))}
    manifest = {"run_id": run_id, "stage": stage, "archived_previous_outputs": archived, "input_rows": len(rows), "split_audit": audit, "context_pair_count": len(context_pairs), "e1": results, "teacher": teacher_summary, "test_frozen": True, "notes": "All thresholds were chosen on dev only. This run does not claim acceptance until the frozen test criteria are independently met."}
    (output / "metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text("# 整改实验运行\n\n该目录由 `scripts/run_remediation.py` 生成。阈值只在 dev 上选择；test 不参与调参。\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "input_rows": len(rows), "acceptance": results["acceptance"], "teacher": teacher_summary}, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--provider", choices=["deepseek", "qwen", "kimi", "glm"])
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    defaults = {"smoke": 20, "pilot": 800, "full": None}
    run(args.stage, args.limit if args.limit is not None else defaults[args.stage], args.provider, args.archive)


if __name__ == "__main__":
    main()
