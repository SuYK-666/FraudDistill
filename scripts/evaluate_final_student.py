# -*- coding: utf-8 -*-
"""Final student evaluation: dev checkpoint selection, threshold calibration,
fresh reload checksum and the single official test run (guide 16, 19-21).

Usage:
  python scripts/evaluate_final_student.py --select-best-on-dev --out-dir .../final_distilled_student
  python scripts/evaluate_final_student.py --checkpoint <best> --reload-check --checksum-samples 128
  python scripts/evaluate_final_student.py --checkpoint <best> --split test --frozen-calibration --out-dir .../final_distilled_student/test_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from frauddistill.student.collator import neural_collate
from frauddistill.student.dataset import ID_TO_LABEL, LABEL_TO_ID, build_neural_examples
from frauddistill.student.model import NeuralStudentConfig, build_neural_student

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FINAL_DIR = OUT_ROOT / "neural_student" / "final_distilled_student"


def load_rows(split: str):
    dataset = [json.loads(l) for l in (REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl").open(encoding="utf-8") if l.strip()]
    teacher = {}
    for sp in ("dev", "test"):
        for r in [json.loads(l) for l in (OUT_ROOT / "agent_predictions" / f"{sp}.jsonl").open(encoding="utf-8") if l.strip()]:
            teacher[r["id"]] = r
    out = []
    for r in dataset:
        if r["split"] != split:
            continue
        t = teacher.get(r["id"]) or {}
        sig = t.get("signal") or {}
        out.append({**r,
                    "teacher_label": str(sig.get("teacher_label", "safe")),
                    "teacher_score": float(sig.get("teacher_score", 0.5)),
                    "teacher_type": str(sig.get("teacher_type", "safe")),
                    "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.5))),
                    "agent_agreement": float(sig.get("agent_agreement", 0.0)),
                    "confidence_tier": "high" if float(sig.get("teacher_confidence", 0)) >= 0.8 else "medium",
                    "conflict_flags": list((sig.get("conflict_flags") or []) + (t.get("conflict_flags") or [])),
                    "gold_source": "procedural_weak" if r["source"] == "synthetic"
                                   else ("audit" if r["source"] in ("e1_context_r2", "fraudr1_all") else "official")})
    return out


def ece_score(p, y, bins=10):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out += (mask.sum() / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def load_checkpoint(ckpt_dir: Path, architecture: str = "standard", max_length: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = NeuralStudentConfig(model_name=BASE_MODEL, architecture=architecture,
                              max_length=max_length, lora_r=32, lora_alpha=64, use_lora=False)
    model = build_neural_student(cfg, freeze_base=True)
    if (ckpt_dir / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.eval()
    return model, tokenizer


def predict_scores(model, tokenizer, rows, max_length=512, architecture="standard",
                   micro_batch=16, with_logits=False):
    exs = build_neural_examples(rows, max_length=max_length, use_teacher_soft=True, use_pairwise=False)
    class D(Dataset):
        def __len__(self): return len(exs)
        def __getitem__(self, i): return exs[i]
    loader = DataLoader(D(), batch_size=micro_batch, shuffle=False,
                        collate_fn=lambda b: neural_collate(b, tokenizer, max_length=max_length, architecture=architecture))
    preds, logits_list = [], []
    with torch.no_grad():
        for batch in loader:
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            probs = torch.softmax(out.logits, dim=-1).numpy()
            if with_logits:
                logits_list.append(out.logits.numpy())
            for i, rid in enumerate(batch["ids"]):
                type_probs = {ID_TO_LABEL[j]: round(float(probs[i, j]), 4) for j in range(4)}
                risk = float(1.0 - probs[i, 0])
                preds.append({"id": rid, "label": "unsafe" if risk >= 0.5 else "safe",
                              "risk_type": max(type_probs, key=type_probs.get),
                              "risk_score": round(risk, 4), "type_probabilities": type_probs})
    return preds, logits_list


def full_metrics(rows, preds, threshold=0.5):
    pmap = {p["id"]: p for p in preds}
    y = [1 if r["gold_label"] == "unsafe" else 0 for r in rows]
    scores = np.array([pmap[r["id"]]["risk_score"] for r in rows], dtype=float)
    y_arr = np.array(y, dtype=int)
    p_arr = np.array([1 if s >= threshold else 0 for s in scores], dtype=int)
    tn = int(((p_arr == 0) & (y_arr == 0)).sum()); fp = int(((p_arr == 1) & (y_arr == 0)).sum())
    fn = int(((p_arr == 0) & (y_arr == 1)).sum()); tp = int(((p_arr == 1) & (y_arr == 1)).sum())
    m = {
        "n": len(rows),
        "acc": round(float(accuracy_score(y_arr, p_arr)), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "unsafe_f1": round(float(f1_score(y_arr, p_arr, pos_label=1, zero_division=0)), 4),
        "safe_f1": round(float(f1_score(y_arr, p_arr, pos_label=0, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_arr, p_arr, average="macro", zero_division=0)), 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_arr, p_arr)), 4),
        "mcc": round(float(matthews_corrcoef(y_arr, p_arr)), 4),
        "auprc": round(float(average_precision_score(y_arr, scores)), 4),
        "auroc": round(float(roc_auc_score(y_arr, scores)), 4),
        "ece": round(ece_score(scores, y_arr), 4),
        "brier": round(float(np.mean((scores - y_arr) ** 2)), 4),
    }
    gold_types = [LABEL_TO_ID.get(r.get("gold_type", ""), (0 if r["gold_label"] == "safe" else 1)) for r in rows]
    pred_types = [LABEL_TO_ID.get(pmap[r["id"]]["risk_type"], 0) for r in rows]
    m["4class_macro_f1"] = round(float(f1_score(gold_types, pred_types, average="macro", zero_division=0)), 4)
    m["confusion_4class"] = confusion_matrix(gold_types, pred_types, labels=[0, 1, 2, 3]).tolist()
    # slices
    sl = {}
    for key, cond in [("real_only", lambda r: r["source"] != "synthetic"),
                      ("synthetic_only", lambda r: r["source"] == "synthetic"),
                      ("en", lambda r: r.get("language") == "en"),
                      ("zh", lambda r: r.get("language") == "zh")]:
        idx = [i for i, r in enumerate(rows) if cond(r)]
        if not idx:
            continue
        ys = y_arr[idx]; ps = p_arr[idx]; sc = scores[idx]
        tp2 = int(((ps == 1) & (ys == 1)).sum()); fn2 = int(((ps == 0) & (ys == 1)).sum())
        fp2 = int(((ps == 1) & (ys == 0)).sum()); tn2 = int(((ps == 0) & (ys == 0)).sum())
        sl[key] = {"n": len(idx),
                   "macro_f1": round(float(f1_score(ys, ps, average="macro", zero_division=0)), 4),
                   "recall": round(tp2 / max(tp2 + fn2, 1), 4),
                   "fpr": round(fp2 / max(tn2 + fp2, 1), 4),
                   "auprc": round(float(average_precision_score(ys, sc)), 4)
                   if len(np.unique(ps)) > 1 and 0 < ys.sum() < len(ys) else None}
    for sub, key, kind in [("direct_fraud", "direct", "recall"), ("trust_facilitation", "trust", "recall"),
                           ("partial_leakage", "leakage", "recall"), ("clean_refusal", "clean_refusal", "fpr"),
                           ("hard_safe", "hard_safe", "fpr"), ("over_refusal", "over_refusal", "recall"),
                           ("quotation_analysis", "quotation", "fpr")]:
        idx = [i for i, r in enumerate(rows) if r.get("subtype") == sub]
        if not idx:
            continue
        ys = y_arr[idx]; ps = p_arr[idx]
        if kind == "recall":
            tp2 = int(((ps == 1) & (ys == 1)).sum()); fn2 = int(((ps == 0) & (ys == 1)).sum())
            sl[f"{key}_recall"] = round(tp2 / max(tp2 + fn2, 1), 4)
        else:
            fp2 = int(((ps == 1) & (ys == 0)).sum()); tn2 = int(((ps == 0) & (ys == 0)).sum())
            sl[f"{key}_fpr"] = round(fp2 / max(tn2 + fp2, 1), 4)
        sl[f"{key}_n"] = len(idx)
    pairs = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("pair_id"):
            pairs[r["pair_id"]].append((i, r["gold_label"]))
    correct = total = 0
    for pid, members in pairs.items():
        if len(members) != 2:
            continue
        (i1, g1), (i2, g2) = members
        total += 1
        if p_arr[i1] == (1 if g1 == "unsafe" else 0) and p_arr[i2] == (1 if g2 == "unsafe" else 0):
            correct += 1
    sl["context_flip_pair_acc"] = round(correct / max(total, 1), 4)
    sl["context_flip_pairs"] = total
    m["slices"] = sl
    return m, scores, y_arr


def select_best_on_dev(out_dir: Path, rows, architecture: str, max_length: int,
                     fast_rows=None, top_k_full: int = 3):
    """Two-phase dev selection (guide 16):
      1) fast pass over every checkpoint on a fixed subset (default: all rows);
      2) full-dev evaluation of the top-k candidates, pick the best by
         dev macro-F1 within FPR<=0.055 & recall>=0.82.
    """
    ckpts = sorted([d for d in out_dir.glob("checkpoint-*") if d.is_dir()],
                   key=lambda d: int(d.name.split("-")[1]))
    best_dirs = sorted([d for d in out_dir.glob("best_step*") if d.is_dir()],
                       key=lambda d: int(d.name.replace("best_step", "")))
    all_ck = ckpts + best_dirs
    # dedupe by dir name
    seen = set(); ckpts = []
    for d in all_ck:
        if d.name not in seen:
            seen.add(d.name); ckpts.append(d)
    ckpts.sort(key=lambda d: int("".join(ch for ch in d.name if ch.isdigit()) or 0))
    if not ckpts:
        print("no checkpoint-* / best_step* dirs found; abort")
        sys.exit(2)

    eval_rows = fast_rows if fast_rows is not None else rows
    quick = []
    for ck in ckpts:
        step = int("".join(ch for ch in ck.name if ch.isdigit()) or 0)
        model, tok = load_checkpoint(ck, architecture, max_length)
        preds, _ = predict_scores(model, tok, eval_rows, max_length, architecture)
        m, _, _ = full_metrics(eval_rows, preds, threshold=0.5)
        quick.append({"step": step, "dir": str(ck), "metrics": m})
        print(f"quick {ck.name}: mf1={m['macro_f1']} recall={m['recall']} fpr={m['fpr']} "
              f"real={m['slices'].get('real_only', {}).get('macro_f1')}", flush=True)
    quick.sort(key=lambda r: (r["metrics"]["macro_f1"], -r["step"]), reverse=True)
    candidates = quick[:top_k_full] if fast_rows is not None else quick

    results = []
    for r in candidates:
        ck = Path(r["dir"])
        step = r["step"]
        model, tok = load_checkpoint(ck, architecture, max_length)
        preds, _ = predict_scores(model, tok, rows, max_length, architecture)
        m, scores, y = full_metrics(rows, preds, threshold=0.5)
        print(f"full {ck.name}: mf1={m['macro_f1']} recall={m['recall']} fpr={m['fpr']} "
              f"real={m['slices'].get('real_only', {}).get('macro_f1')}", flush=True)
        results.append({"step": step, "dir": str(ck), "metrics": m,
                        "scores": scores.tolist(), "y": y.tolist()})
    feasible = [r for r in results if r["metrics"]["fpr"] <= 0.055 and r["metrics"]["recall"] >= 0.82]
    pool = feasible if feasible else results
    def key(r):
        m = r["metrics"]
        return (m["macro_f1"],
                m["slices"].get("real_only", {}).get("macro_f1", 0.0),
                m["auprc"], m["mcc"], -r["step"])
    best = max(pool, key=key)
    return best, results, bool(feasible)


def calibrate_threshold(scores, y, fpr_max=0.05, recall_min=0.82, fpr_fallback=0.06):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    cands = np.unique(np.round(scores, 4))
    best = None
    for t in cands:
        p = (scores >= t).astype(int)
        tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tp = int(((p == 1) & (y == 1)).sum())
        fpr = fp / max(tn + fp, 1); rec = tp / max(tp + fn, 1)
        mf1 = float(f1_score(y, p, average="macro", zero_division=0))
        if fpr <= fpr_max and rec >= recall_min:
            if best is None or mf1 > best[0]:
                best = (mf1, t, fpr, rec)
    if best is None:
        for t in cands:
            p = (scores >= t).astype(int)
            tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum()); tp = int(((p == 1) & (y == 1)).sum())
            fpr = fp / max(tn + fp, 1); rec = tp / max(tp + fn, 1)
            mf1 = float(f1_score(y, p, average="macro", zero_division=0))
            if fpr <= fpr_fallback:
                if best is None or mf1 > best[0]:
                    best = (mf1, t, fpr, rec)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-best-on-dev", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out-dir", default=str(FINAL_DIR))
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--architecture", default="standard", choices=["standard", "interaction"])
    ap.add_argument("--reload-check", action="store_true")
    ap.add_argument("--checksum-samples", type=int, default=128)
    ap.add_argument("--frozen-calibration", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--fast-subset", type=int, default=300, help="fast-pass dev subset size for candidate ranking")
    ap.add_argument("--top-k-full", type=int, default=3, help="top-k candidates for full-dev evaluation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = load_rows("dev")
    test_rows = load_rows("test")

    if args.select_best_on_dev:
        rng = __import__("random").Random(20260804)
        fast_rows = rng.sample(dev_rows, min(args.fast_subset, len(dev_rows))) if args.fast_subset < len(dev_rows) else None
        best, results, feasible = select_best_on_dev(out_dir, dev_rows, args.architecture, args.max_length,
                                                     fast_rows=fast_rows, top_k_full=args.top_k_full)
        m = best["metrics"]
        cal = calibrate_threshold(np.array(best["scores"]), np.array(best["y"]))
        if cal is None:
            cal = (m["macro_f1"], 0.5, m["fpr"], m["recall"])
        (out_dir / "best_checkpoint.json").write_text(
            json.dumps({"best_step": best["step"], "checkpoint": best["dir"],
                        "dev_gate_warning": not feasible,
                        "selection": {"fpr_max": 0.055, "recall_min": 0.82},
                        "selected_by": "dev macro_f1 within FPR<=0.055 & recall>=0.82, "
                                       "tie-break real_macro_f1>AUPRC>MCC>earlier"}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out_dir / "calibration.json").write_text(
            json.dumps({"threshold": float(cal[1]), "dev_macro_f1_at_threshold": cal[0],
                        "dev_fpr": cal[2], "dev_recall": cal[3],
                        "criterion": "FPR<=0.05 & recall>=0.82 -> max MF1; else FPR<=0.06 best",
                        "note": "threshold frozen; test runs once"}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out_dir / "dev_metrics.json").write_text(
            json.dumps({"best_step": best["step"], "metrics": m,
                        "all_checkpoints": [{**r, "scores": None, "y": None} for r in results]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("BEST:", json.dumps({"step": best["step"], "metrics": m, "threshold": cal[1],
                                   "gate_warning": not feasible}, ensure_ascii=False, indent=2))

    if args.reload_check:
        ckpt = Path(args.checkpoint) if args.checkpoint else Path(json.loads((out_dir / "best_checkpoint.json").read_text(encoding="utf-8"))["checkpoint"])
        rng = np.random.RandomState(20260807)
        idx = rng.choice(len(dev_rows), min(args.checksum_samples, len(dev_rows)), replace=False)
        rows = [dev_rows[i] for i in sorted(idx)]
        # path 1: PeftModel.from_pretrained on plain base (canonical)
        model1, tok = load_checkpoint(ckpt, args.architecture, args.max_length)
        _, logits1 = predict_scores(model1, tok, rows, args.max_length, args.architecture,
                                    micro_batch=args.micro_batch, with_logits=True)
        # path 2: build with LoRA (training-time structure) then load the adapter
        # NOTE: PeftModel.from_pretrained on an already-wrapped PeftModel would
        # double-wrap it: modules_to_save keys (score/classifier) shift and the
        # trained classification head silently fails to load (random head ->
        # O(1e1) logit diffs). Use load_adapter to overwrite in place instead.
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        from peft import LoraConfig, get_peft_model
        cfg = NeuralStudentConfig(model_name=BASE_MODEL, architecture=args.architecture,
                                  max_length=args.max_length, lora_r=32, lora_alpha=64, use_lora=True)
        model2 = build_neural_student(cfg, freeze_base=True)
        model2.load_adapter(str(ckpt), adapter_name="default")
        model2.eval()
        _, logits2 = predict_scores(model2, tok, rows, args.max_length, args.architecture,
                                    micro_batch=args.micro_batch, with_logits=True)
        diff = max(float(np.abs(np.concatenate(logits1) - np.concatenate(logits2)).max()), 0.0)
        result = {"checkpoint": str(ckpt), "checksum_samples": len(rows),
                  "max_logit_diff": diff, "pass": diff <= 1e-5,
                  "classifier_present": any("score" in k for k in model1.state_dict()),
                  "adapter_present": (ckpt / "adapter_config.json").exists()}
        (out_dir / "reload_checksum.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RELOAD CHECK:", json.dumps(result, ensure_ascii=False))
        if not result["pass"]:
            sys.exit(1)

    if args.split == "test" and args.checkpoint:
        ckpt = Path(args.checkpoint)
        cal = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8")) if args.frozen_calibration else None
        threshold = float(cal["threshold"]) if cal else 0.5
        model, tok = load_checkpoint(ckpt, args.architecture, args.max_length)
        preds, _ = predict_scores(model, tok, test_rows, args.max_length, args.architecture,
                                  micro_batch=args.micro_batch)
        m, scores, y = full_metrics(test_rows, preds, threshold=threshold)
        m["threshold"] = threshold
        te_dir = out_dir / "test_eval"
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "predictions_test.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n", encoding="utf-8")
        (te_dir / "test_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "test_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TEST (once, frozen calibration):", json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
