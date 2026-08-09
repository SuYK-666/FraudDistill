# -*- coding: utf-8 -*-
"""E4 inference + metrics + significance (exp4_unseen_student_v2).

Usage:
  python scripts/e4e5_run_inference.py --protocol-dir outputs/exp4_unseen_student_v2/e4v2_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import yaml

from frauddistill.e4e5_v2.schemas import read_jsonl, write_jsonl, manifest_sha256
from frauddistill.e4e5_v2.student_inference import run_inference, load_predictions
from frauddistill.e4e5_v2.metrics import binary_metrics
from frauddistill.e4e5_v2.cluster_bootstrap import paired_cluster_bootstrap, exact_mcnemar

COMPARATORS = {
    "final_student": dict(ckpt="experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120",
                          threshold=0.5622, max_length=512, tag="Final Student"),
    "neural_gold": dict(ckpt="experiments/exp3_agent_distillation_ablation/outputs/neural_student/gold_standard_seed11_final",
                        threshold=0.5, max_length=384, tag="Neural-Gold"),
    "neural_softdistill": dict(ckpt="experiments/exp3_agent_distillation_ablation/outputs/neural_student/soft_distill_standard_seed11_final",
                               threshold=0.5, max_length=384, tag="Neural-SoftDistill"),
}


def base_zeroshot_on_rows(rows, n=300, seed=20260809, out_path=None):
    """Reuse exp3 Base-1.5B zero-shot protocol on a fixed subset."""
    from run_exp3_base_zeroshot import (CLASSES, PROMPT_TMPL, build_prompt, logprobs_of_labels,
                                        parse_label, ZH_MAP)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = random.Random(seed)
    subset = rng.sample(rows, min(n, len(rows)))
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    label_ids = {c: tok.encode(" " + c, add_special_tokens=False) for c in CLASSES}
    preds = []
    t0 = time.time()
    for i, r in enumerate(subset):
        prompt = build_prompt(r)
        pids = tok(prompt, add_special_tokens=False).input_ids
        lps = logprobs_of_labels(model, tok, torch.tensor(pids, dtype=torch.long), label_ids, "cpu")
        pred_label = max(lps, key=lps.get)
        rec = {"id": r["id"], "label": "unsafe" if pred_label in ("fraud_assistance", "refusal_failure", "over_refusal") else "safe",
               "risk_type": pred_label, "risk_score": round(1.0 - lps["safe"], 4),
               "type_probabilities": {k: round(v, 4) for k, v in lps.items()},
               "threshold": 0.5, "model": "Base-1.5B-ZeroShot", "n_logprob": "ok"}
        preds.append(rec)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[base] {i+1}/{len(subset)} elapsed={el:.0f}s", flush=True)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        (out_path.parent / "base_subset_ids.json").write_text(
            json.dumps({"seed": seed, "n": len(subset), "ids": [r["id"] for r in subset]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return preds, subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--models", default="final_student,neural_gold,neural_softdistill")
    ap.add_argument("--micro-batch", type=int, default=6)
    args = ap.parse_args()

    proto = Path(args.protocol_dir)
    manifests = proto / "manifests"
    test_rows = read_jsonl(manifests / "frozen_test.jsonl")
    cal_rows = read_jsonl(manifests / "calibration.jsonl")
    print(f"[e4] frozen test n={len(test_rows)} calibration n={len(cal_rows)}")

    cfg = yaml.safe_load((REPO / "configs/experiments/exp4_unseen_student_v2.yaml").read_text(encoding="utf-8"))
    pred_dir = proto / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # lock marker: consume test exactly once
    consume = pred_dir / "TEST_CONSUME_TOKEN.json"
    if not consume.exists():
        consume.write_text(json.dumps({
            "consumed_once": True,
            "manifest_sha256": manifest_sha256(test_rows),
            "n": len(test_rows),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[e4] TEST_CONSUME_TOKEN: {consume}")

    # ---- student-family comparators ----
    for key in args.models.split(","):
        key = key.strip()
        if key not in COMPARATORS:
            continue
        spec = COMPARATORS[key]
        ckpt = REPO / spec["ckpt"]
        for split_name, rows in (("frozen_test", test_rows),):
            out = pred_dir / f"{key}.jsonl"
            if out.exists():
                print(f"[e4] {key}/{split_name} exists, skip ({len(load_predictions(out))} rows)")
                continue
            print(f"[e4] running {key} ({spec['tag']}) on {len(rows)} rows", flush=True)
            res = run_inference(rows, ckpt, out, spec["threshold"], spec["max_length"],
                                micro_batch=args.micro_batch, tag=spec["tag"])
            print(f"[e4] {key} done: {res}")
        # final student also on calibration (needed by E5)
        if key == "final_student":
            out_cal = pred_dir / "final_student_calibration.jsonl"
            if not out_cal.exists():
                print(f"[e4] final_student on calibration {len(cal_rows)}", flush=True)
                res = run_inference(cal_rows, ckpt, out_cal, spec["threshold"], spec["max_length"],
                                    micro_batch=args.micro_batch, tag=spec["tag"])
                print(f"[e4] final_student cal done: {res}")

    # ---- base zero-shot on 300-row subset ----
    if not args.skip_base:
        base_out = pred_dir / "base_zeroshot.jsonl"
        if not base_out.exists():
            print("[e4] base zero-shot 300-row subset", flush=True)
            base_zeroshot_on_rows(test_rows, n=cfg["panel"]["pilot_n"], seed=cfg["seed"], out_path=base_out)

    # ---- metrics ----
    from frauddistill.e4e5_v2.shortcut_audit import shortcut_audit
    shortcut_audit(test_rows, proto / "audits")

    print("[e4] metrics table (pooled + shift + cell)")
    for key in args.models.split(","):
        key = key.strip()
        out = pred_dir / f"{key}.jsonl"
        if not out.exists():
            continue
        preds = load_predictions(out)
        rows = [r for r in test_rows if r["id"] in preds]
        y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
        scores = np.array([preds[r["id"]]["risk_score"] for r in rows])
        th = COMPARATORS[key]["threshold"]
        m = binary_metrics(y, scores, threshold=th, label=COMPARATORS[key]["tag"])
        print(f"[e4] {key} pooled: MF1={m['macro_f1']} Recall={m['recall']} FPR={m['fpr']} MCC={m['mcc']} AUPRC={m.get('auprc')}")
        for shift in ("U1_category", "U2_source", "U3_target_style"):
            idx = [i for i, r in enumerate(rows) if r["primary_shift"] == shift]
            if idx:
                ms = binary_metrics(y[idx], scores[idx], threshold=th, label=shift)
                print(f"[e4]   {shift}: MF1={ms['macro_f1']} Recall={ms['recall']} FPR={ms['fpr']} n={len(idx)}")


if __name__ == "__main__":
    main()
