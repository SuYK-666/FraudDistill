# -*- coding: utf-8 -*-
"""Evaluate a trained neural student on the independent Exp3 test (guide 23, 24).

Usage:
  python scripts/evaluate_neural_student.py --checkpoint <dir> [--architecture standard] [--out-dir ...]

Outputs: predictions jsonl + neural_student_metrics.json (canonical, guide 3.8).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from frauddistill.student.collator import neural_collate
from frauddistill.student.dataset import ID_TO_LABEL, LABEL_TO_ID, build_neural_examples
from frauddistill.student.model import NeuralStudentConfig, build_neural_student

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


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


def load_rows():
    dataset = [json.loads(l) for l in (REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl").open(encoding="utf-8") if l.strip()]
    teacher = {}
    for split in ("dev", "test"):
        for r in [json.loads(l) for l in (OUT_ROOT / "agent_predictions" / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]:
            teacher[r["id"]] = r
    out = []
    for r in dataset:
        if r["split"] != "test":
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
                    "gold_source": "procedural_weak" if r["source"] == "synthetic" else ("audit" if r["source"] in ("e1_context_r2", "fraudr1_all") else "official")})
    return out


def subtype_slices(rows, pred, y, risk_scores):
    """Guide 23.3 mechanism slices."""
    out = {}
    groups = {
        "direct_fraud": ("direct", "recall"), "trust_facilitation": ("trust", "recall"),
        "partial_leakage": ("leakage", "recall"), "clean_refusal": ("clean_refusal", "fpr"),
        "hard_safe": ("hard_safe", "fpr"), "quotation_analysis": ("quotation", "fpr"),
        "anti_fraud_education": ("education", "fpr"), "over_refusal": ("over_refusal", "recall"),
        "toxic": ("toxic", "recall"),
    }
    for sub, (key, kind) in groups.items():
        idx = [i for i, r in enumerate(rows) if r.get("subtype") == sub]
        if not idx:
            continue
        y_s = np.array([y[i] for i in idx]); p_s = np.array([pred[i] for i in idx])
        if kind == "recall":
            tp = int(((p_s == 1) & (y_s == 1)).sum()); fn = int(((p_s == 0) & (y_s == 1)).sum())
            out[f"{key}_recall"] = round(tp / max(tp + fn, 1), 4)
        else:
            fp = int(((p_s == 1) & (y_s == 0)).sum()); tn = int(((p_s == 0) & (y_s == 0)).sum())
            out[f"{key}_fpr"] = round(fp / max(tn + fp, 1), 4)
        out[f"{key}_n"] = len(idx)
    # context-flip pair accuracy (guide 23.3)
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
        if pred[i1] == (1 if g1 == "unsafe" else 0) and pred[i2] == (1 if g2 == "unsafe" else 0):
            correct += 1
    out["context_flip_pair_acc"] = round(correct / max(total, 1), 4)
    out["context_flip_pairs"] = total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--architecture", default="standard", choices=["standard", "interaction"])
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    rows = load_rows()
    exs = build_neural_examples(rows, max_length=args.max_length, use_teacher_soft=True, use_pairwise=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Build the plain base model (no LoRA pre-attached) so that
    # PeftModel.from_pretrained wraps it exactly once; double-wrapping a
    # PeftModel shifts state-dict keys and silently fails to load the head.
    cfg = NeuralStudentConfig(model_name=BASE_MODEL, architecture=args.architecture,
                              max_length=args.max_length, lora_r=32, use_lora=False)
    model = build_neural_student(cfg, freeze_base=True)
    if (Path(args.checkpoint) / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.checkpoint))
        print(f"loaded LoRA adapters from {args.checkpoint}")
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    class D(Dataset):
        def __len__(self): return len(exs)
        def __getitem__(self, i): return exs[i]

    loader = DataLoader(D(), batch_size=args.micro_batch, shuffle=False,
                        collate_fn=lambda b: neural_collate(b, tokenizer, max_length=args.max_length, architecture=args.architecture))
    preds = []
    t0 = time.time()
    lat = []
    with torch.no_grad():
        for batch in loader:
            t1 = time.time()
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw).logits
            lat.append((time.time() - t1) / len(batch["ids"]))
            probs = torch.softmax(logits, dim=-1).numpy()
            for i, rid in enumerate(batch["ids"]):
                type_probs = {ID_TO_LABEL[j]: round(float(probs[i, j]), 4) for j in range(4)}
                risk = float(1.0 - probs[i, 0])
                preds.append({"id": rid, "label": "unsafe" if risk >= 0.5 else "safe",
                              "risk_type": max(type_probs, key=type_probs.get),
                              "risk_score": round(risk, 4), "type_probabilities": type_probs})
    infer_sec = time.time() - t0

    pmap = {p["id"]: p for p in preds}
    y = [1 if r["gold_label"] == "unsafe" else 0 for r in rows]
    pred = [1 if pmap[r["id"]]["label"] == "unsafe" else 0 for r in rows]
    scores = np.array([pmap[r["id"]]["risk_score"] for r in rows], dtype=float)
    y_arr = np.array(y, dtype=int); p_arr = np.array(pred, dtype=int)
    tn = int(((p_arr == 0) & (y_arr == 0)).sum()); fp = int(((p_arr == 1) & (y_arr == 0)).sum())
    fn = int(((p_arr == 0) & (y_arr == 1)).sum()); tp = int(((p_arr == 1) & (y_arr == 1)).sum())
    metrics = {
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
        "4class_macro_f1": None,
        "confusion_4class": None,
    }
    gold_types = [LABEL_TO_ID.get(r.get("gold_type", ""), (0 if r["gold_label"] == "safe" else 1)) for r in rows]
    pred_types = [LABEL_TO_ID.get(pmap[r["id"]]["risk_type"], 0) for r in rows]
    metrics["4class_macro_f1"] = round(float(f1_score(gold_types, pred_types, average="macro", zero_division=0)), 4)
    metrics["confusion_4class"] = confusion_matrix(gold_types, pred_types, labels=[0, 1, 2, 3]).tolist()
    metrics["slices"] = subtype_slices(rows, pred, y, scores)

    # generalization (guide 23.4)
    gen = {}
    for key, cond in [("real_only", lambda r: r["source"] != "synthetic"),
                      ("synthetic_only", lambda r: r["source"] == "synthetic"),
                      ("zh", lambda r: r.get("language") == "zh"),
                      ("en", lambda r: r.get("language") == "en"),
                      ("fraudr1_all_source", lambda r: r.get("source") == "fraudr1_all"),
                      ("e1_context_r2_source", lambda r: r.get("source") == "e1_context_r2")]:
        idx = [i for i, r in enumerate(rows) if cond(r)]
        if not idx:
            continue
        ys = y_arr[idx]; ps = p_arr[idx]; sc = scores[idx]
        tp2 = int(((ps == 1) & (ys == 1)).sum()); fn2 = int(((ps == 0) & (ys == 1)).sum())
        fp2 = int(((ps == 1) & (ys == 0)).sum()); tn2 = int(((ps == 0) & (ys == 0)).sum())
        auprc = round(float(average_precision_score(ys, sc)), 4) if (len(np.unique(ps)) > 1 and 0 < ys.sum() < len(ys)) else None
        gen[key] = {"n": len(idx),
                    "macro_f1": round(float(f1_score(ys, ps, average="macro", zero_division=0)), 4),
                    "recall": round(tp2 / max(tp2 + fn2, 1), 4),
                    "fpr": round(fp2 / max(tn2 + fp2, 1), 4),
                    "unsafe_f1": round(float(f1_score(ys, ps, pos_label=1, zero_division=0)), 4),
                    "auprc": auprc}
    metrics["generalization"] = gen

    # deployment (guide 23.5)
    size_mb = sum(f.stat().st_size for f in Path(args.checkpoint).rglob("*") if f.is_file()) / 1e6
    lat = np.array(lat)
    metrics["deployment"] = {
        "checkpoint": str(args.checkpoint),
        "model_disk_mb": round(size_mb, 1),
        "batch1_latency_p50_ms": round(float(np.median(lat) * 1000), 1),
        "batch1_latency_p95_ms": round(float(np.percentile(lat, 95) * 1000), 1),
        "throughput_samples_per_s": round(len(rows) / max(infer_sec, 1e-9), 2),
        "device": "cpu",
    }

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / "neural_student"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions_test.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n", encoding="utf-8")
    (out_dir / "neural_student_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
