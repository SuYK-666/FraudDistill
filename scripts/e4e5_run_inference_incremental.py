# -*- coding: utf-8 -*-
"""E4 v2: incremental inference for rebuilt panel (only missing ids).

Merges new predictions into existing prediction files so we never re-run
rows that were already inferred on the old panel.
Usage:
  python scripts/e4e5_run_inference_incremental.py --models final_student,neural_gold,neural_softdistill
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from frauddistill.e4e5_v2.schemas import read_jsonl
from frauddistill.e4e5_v2.student_inference import predict_scores, load_checkpoint

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
COMPARATORS = {
    "final_student": dict(ckpt=REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120",
                          threshold=0.5622, max_length=512, tag="Final Student"),
    "neural_gold": dict(ckpt=REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/gold_standard_seed11_final",
                        threshold=0.5, max_length=384, tag="Neural-Gold"),
    "neural_softdistill": dict(ckpt=REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/soft_distill_standard_seed11_final",
                               threshold=0.5, max_length=384, tag="Neural-SoftDistill"),
}

def load_preds(path: Path) -> dict:
    out = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); out[r["id"]] = r
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="final_student,neural_gold,neural_softdistill")
    ap.add_argument("--micro-batch", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0=auto)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    test_rows = read_jsonl(BASE / "manifests/frozen_test.jsonl")
    cal_rows = read_jsonl(BASE / "manifests/calibration.jsonl")
    print(f"[incr] test={len(test_rows)} cal={len(cal_rows)}")

    for key in [k for k in args.models.split(",") if k in COMPARATORS]:
        spec = COMPARATORS[key]
        for split_name, rows in (("frozen_test", test_rows),):
            out = BASE / "predictions" / f"{key}.jsonl"
            preds = load_preds(out)
            missing = [r for r in rows if r["id"] not in preds]
            print(f"[incr] {key}/{split_name}: existing={len(preds)} missing={len(missing)}")
            if missing:
                model, tok = load_checkpoint(spec["ckpt"], max_length=spec["max_length"])
                progress = BASE / "predictions" / f"incremental_progress_{key}.jsonl"
                new_preds, _ = predict_scores(model, tok, missing, max_length=spec["max_length"],
                                              micro_batch=args.micro_batch, progress_path=progress, tag=spec["tag"])
                del model, tok
                import gc; gc.collect()
                for p in new_preds:
                    p["threshold"] = spec["threshold"]; p["model"] = spec["tag"]
                with open(out, "w", encoding="utf-8") as f:
                    for p in sorted(preds.values(), key=lambda x: x["id"]):
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    for p in sorted(new_preds, key=lambda x: x["id"]):
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                print(f"[incr] {key} merged: total={len(preds)+len(new_preds)} new={len(new_preds)}")
            else:
                print(f"[incr] {key} nothing to do")
        if key == "final_student":
            out_cal = BASE / "predictions" / "final_student_calibration.jsonl"
            preds = load_preds(out_cal)
            test_preds = load_preds(BASE / "predictions" / "final_student.jsonl")
            union = dict(preds); union.update(test_preds)
            missing = [r for r in cal_rows if r["id"] not in union]
            print(f"[incr] final_student/calibration union-borrow: existing={len(union)} missing={len(missing)}")
            print(f"[incr] final_student/calibration: existing={len(preds)} missing={len(missing)}")
            if missing:
                model, tok = load_checkpoint(spec["ckpt"], max_length=spec["max_length"])
                progress = BASE / "predictions" / "incremental_progress_final_student_cal.jsonl"
                new_preds, _ = predict_scores(model, tok, missing, max_length=spec["max_length"],
                                              micro_batch=args.micro_batch, progress_path=progress, tag=spec["tag"])
                del model, tok
                import gc; gc.collect()
                for p in new_preds:
                    p["threshold"] = spec["threshold"]; p["model"] = spec["tag"]
                with open(out_cal, "w", encoding="utf-8") as f:
                    for p in sorted(preds.values(), key=lambda x: x["id"]):
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    for p in sorted(new_preds, key=lambda x: x["id"]):
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                print(f"[incr] final_student cal merged: new={len(new_preds)}")
    print("[incr] DONE")

if __name__ == "__main__":
    main()
