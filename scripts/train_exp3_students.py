# -*- coding: utf-8 -*-
"""Train the lightweight Student gradient S0-S4 for Exp3 (guide 18).

Students: TF-IDF + Logistic Regression on q+y only (guide 18.1/18.4).
S0 gold, S1 +high-confidence hard teacher labels, S2 +teacher score weights,
S3 +risk type heads, S4 +evidence-aware weights & pair ranking.
5 seeds -> mean +/- std on held-out test.

Usage: python scripts/train_exp3_students.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
AGENT_DIR = OUT_ROOT / "agent_predictions"
SEEDS = [11, 23, 37, 53, 71]
LAMBDA_H = 0.5      # hard teacher label duplication weight
LAMBDA_S = 0.5      # score weight strength
LAMBDA_PAIR = 0.4   # pair ranking boost
HIGH_CONF = 0.80
# The arbiter's self-reported confidence is almost always >= 0.8 (degenerate),
# so "high confidence" is gated on score decisiveness: a decisive band of
# +-0.25 around the frozen operating threshold (0.84): score>=0.95 (strong
# unsafe) or score<=0.60 (strong safe). Determined from train statistics only.
DECISIVE_UNSAFE_SCORE = 0.95
DECISIVE_SAFE_SCORE = 0.60
TYPES = ["fraud_assistance", "refusal_failure", "over_refusal", "safe"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_all() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    dataset = read_jsonl(DATASET)
    teacher = {r["id"]: r for r in read_jsonl(AGENT_DIR / "train.jsonl")}
    dev_t = {r["id"]: r for r in read_jsonl(AGENT_DIR / "dev.jsonl")}
    test_t = {r["id"]: r for r in read_jsonl(AGENT_DIR / "test.jsonl")}
    teacher.update(dev_t)
    teacher.update(test_t)
    return dataset, teacher, teacher


class QyTfidf:
    """TF-IDF on q+y with a binary LR head (student, guide 18.1)."""

    def __init__(self, max_features: int = 60000, C: float = 1.0, seed: int = 11):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="liblinear", random_state=seed)

    def fit(self, rows, labels, sample_weight=None):
        X = self.vec.fit_transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
        y = np.array([1 if str(l) == "unsafe" else 0 for l in labels], dtype=int)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.clf.fit(X, y, **kw)
        return self

    def predict_proba(self, rows):
        X = self.vec.transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
        return self.clf.predict_proba(X)[:, 1]


def teacher_signal(rec: dict) -> dict:
    sig = rec.get("signal") or {}
    score = float(sig.get("teacher_score", 0.5))
    confidence = float(sig.get("confidence", 0.5))
    decisive = score >= DECISIVE_UNSAFE_SCORE or score <= DECISIVE_SAFE_SCORE
    return {
        "label": str(sig.get("teacher_label", "safe")),
        "score": score,
        "type": str(sig.get("teacher_type", "safe")),
        "confidence": confidence,
        "decisive": bool(confidence >= HIGH_CONF and decisive),
        "conflict": bool(rec.get("conflict_flags")),
        "correction_used": bool(sig.get("correction_used")),
    }


def build_teacher_rows(train_rows, teacher_map) -> tuple[list, list, list]:
    """Return (rows, labels, weights) with high-confidence teacher duplication."""
    rows, labels, weights = [], [], []
    for r in train_rows:
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(1.0)
        if t["decisive"] and t["label"] in ("safe", "unsafe"):
            rows.append(r)
            labels.append(t["label"])
            weights.append(LAMBDA_H)
    return rows, labels, weights


def full_distill_rows(train_rows, teacher_map):
    """S4: score distill + evidence-aware weights + pair ranking (soft signals).

    Hard teacher-label duplication was net-negative on dev (hard pseudo-label
    error amplification), so S4 keeps soft score weights, evidence weights and
    pair ranking, and drops the hard-label duplicates (documented deviation).
    """
    rows, labels, weights = [], [], []
    base_ew = evidence_weights(train_rows, teacher_map)
    for i, r in enumerate(train_rows):
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        score_w = 1.0 + LAMBDA_S * abs(2 * t["score"] - 1)
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(score_w * base_ew[i])
    for r in train_rows:
        if r.get("pair_id") and r["gold_label"] == "unsafe":
            t = teacher_signal(teacher_map.get(r["id"]) or {})
            w = 1.0 + LAMBDA_PAIR
            if t["decisive"]:
                w += 0.3
            rows.append(r)
            labels.append("unsafe")
            weights.append(w)
    return rows, labels, weights


def evidence_weights(train_rows, teacher_map) -> list[float]:
    out = []
    for r in train_rows:
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        w = 1.0
        if t["confidence"] >= HIGH_CONF:
            w *= 1.5
        if t["conflict"]:
            w *= 0.5
        if r.get("subtype") in ("hard_safe", "partial_leakage"):
            w *= 1.5
        out.append(w)
    return out


def pair_boosted_rows(train_rows, teacher_map):
    """Duplicate context-flip pair rows with relative-order emphasis (guide 18.2 S4)."""
    rows, labels, weights = [], [], []
    pairs: dict[str, list[dict]] = {}
    for r in train_rows:
        pid = r.get("pair_id")
        if pid:
            pairs.setdefault(pid, []).append(r)
    for r in train_rows:
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(1.0)
        pid = r.get("pair_id")
        if pid and r["gold_label"] == "unsafe":
            t = teacher_signal(teacher_map.get(r["id"]) or {})
            w = 1.0 + LAMBDA_PAIR
            if t["decisive"]:
                w += 0.3
            rows.append(r)
            labels.append("unsafe")
            weights.append(w)
    return rows, labels, weights


def type_head_fit(train_rows, teacher_map, seed):
    """4 one-vs-rest type heads on gold type (teacher type when high-conf)."""
    rows, labels = [], []
    for r in train_rows:
        gold_type = str(r.get("gold_type") or (r["gold_label"] if r["gold_label"] == "safe" else "fraud_assistance"))
        rows.append(r)
        labels.append(gold_type)
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        if t["confidence"] >= HIGH_CONF and t["type"] in TYPES:
            rows.append(r)
            labels.append(t["type"])
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
    heads = {}
    for tp in TYPES:
        y = np.array([1 if l == tp else 0 for l in labels], dtype=int)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
        clf.fit(X, y)
        heads[tp] = clf
    return vec, heads


def type_proba(vec, heads, rows) -> dict[str, np.ndarray]:
    X = vec.transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
    return {tp: heads[tp].predict_proba(X)[:, 1] for tp in TYPES}


def metrics_binary(labels, scores, threshold=0.5) -> dict:
    """Guide 3.1: report unsafe_f1 / safe_f1 and the TRUE macro-F1 separately."""
    from sklearn.metrics import f1_score as _skf1
    y = np.array([1 if l == "unsafe" else 0 for l in labels], dtype=int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(tn + fp, 1)
    acc = (tp + tn) / max(len(y), 1)
    auprc = float(average_precision_score(y, scores)) if len(set(scores)) > 1 else float("nan")
    return {"acc": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "unsafe_f1": round(float(_skf1(y, pred, pos_label=1, zero_division=0)), 4),
            "safe_f1": round(float(_skf1(y, pred, pos_label=0, zero_division=0)), 4),
            "macro_f1": round(float(_skf1(y, pred, average="macro", zero_division=0)), 4),
            "fpr": round(fpr, 4), "auprc": round(auprc, 4)}


def type_macro_f1(labels, proba: dict[str, np.ndarray]) -> float:
    y = np.array([str(l) for l in labels])
    pred = np.array([max(TYPES, key=lambda tp: proba[tp][i]) for i in range(len(y))])
    f1s = []
    for tp in TYPES:
        tp_y = (y == tp).astype(int)
        tp_p = (pred == tp).astype(int)
        t = int((tp_p & tp_y).sum())
        fp = int((tp_p & (tp_y == 0)).sum())
        fn = int(((tp_p == 0) & tp_y).sum())
        prec = t / max(t + fp, 1)
        rec = t / max(t + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9) if t + fp + fn else 1.0)
    return round(float(np.mean(f1s)), 4)


def run_setting(name, train_rows, test_rows, dev_rows, teacher_map, seeds) -> dict:
    results = []
    for seed in seeds:
        if name == "S0_gold":
            rows, labels, weights = train_rows, [r["gold_label"] for r in train_rows], None
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S1_hard_teacher":
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S2_score_distill":
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            w = np.asarray(weights, dtype=float)
            # score-based confidence weighting on all rows
            base_w = []
            for r in train_rows:
                t = teacher_signal(teacher_map.get(r["id"]) or {})
                base_w.append(1.0 + LAMBDA_S * abs(2 * t["score"] - 1))
            # duplicated teacher rows keep lambda_h * score weight
            dups = len(train_rows)
            w[dups:] = w[dups:] * np.asarray([1.0 + LAMBDA_S * abs(2 * teacher_signal(teacher_map.get(rows[i]["id"]) or {})["score"] - 1) for i in range(dups, len(rows))])
            w[:dups] = w[:dups] * np.asarray(base_w)
            model = QyTfidf(seed=seed).fit(rows, labels, w)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S3_type_distill":
            # Guide 9.2/28.3: type distillation must affect the binary risk score.
            # Risk score = 0.5 * binary LR proba + 0.5 * (1 - P(safe)) from the
            # four one-vs-rest type heads, so type signal is no longer inert.
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            bin_scores = model.predict_proba(test_rows)
            vec, heads = type_head_fit(train_rows, teacher_map, seed)
            proba = type_proba(vec, heads, test_rows)
            type_unsafe = 1.0 - proba["safe"]
            scores = 0.5 * np.asarray(bin_scores) + 0.5 * np.asarray(type_unsafe)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
            m["type_macro_f1"] = type_macro_f1([r.get("gold_type") or (r["gold_label"] if r["gold_label"] == "safe" else "fraud_assistance") for r in test_rows], proba)
        elif name == "S4_evidence_distill":
            rows, labels, weights = full_distill_rows(train_rows, teacher_map)
            w = np.asarray(weights, dtype=float)
            model = QyTfidf(seed=seed).fit(rows, labels, w)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        else:
            raise ValueError(name)
        results.append(m)
    keys = [k for k in results[0] if k != "n"]
    agg = {"setting": name, "seeds": seeds}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            agg[k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)
    return agg


def main_neural(args) -> None:
    """Neural 1.5B student training/eval (guide 18, 22).

    --setting gold|soft_distill|full_distill (guide 15.5)
    --architecture standard|interaction (guide 13)
    --eval-only runs Stage A zero-shot without training (guide 18.1).
    """
    import time
    import torch
    torch.set_num_threads(16)
    from torch.utils.data import DataLoader, Dataset
    from frauddistill.student.dataset import build_neural_examples
    from frauddistill.student.model import NeuralStudentConfig, build_neural_student
    from frauddistill.student.losses import FraudDistillLoss
    from frauddistill.student.collator import neural_collate
    from frauddistill.student.trainer import train_neural, evaluate_neural, save_checkpoint
    from transformers import AutoTokenizer

    out_root = Path(args.out_root or (OUT_ROOT / "neural_student"))
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("manifest missing; run scripts/audit_student_training_data.py first")
        sys.exit(2)
    manifest = [json.loads(l) for l in manifest_path.open(encoding="utf-8") if l.strip()]
    dataset, teacher_map, _ = load_all()
    dev_rows = [r for r in dataset if r["split"] == "dev"]
    test_rows = [r for r in dataset if r["split"] == "test"]

    def with_teacher(rows):
        out = []
        for r in rows:
            t = teacher_map.get(r["id"]) or {}
            sig = t.get("signal") or {}
            out.append({**r, "teacher_label": str(sig.get("teacher_label", "safe")),
                        "teacher_score": float(sig.get("teacher_score", 0.5)),
                        "teacher_type": str(sig.get("teacher_type", "safe")),
                        "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.5))),
                        "agent_agreement": float(sig.get("agent_agreement", 0.0)),
                        "confidence_tier": "high" if float(sig.get("teacher_confidence", 0)) >= 0.8 else "medium",
                        "conflict_flags": list((sig.get("conflict_flags") or []) + (t.get("conflict_flags") or [])),
                        "gold_source": "procedural_weak" if r["source"] == "synthetic" else ("audit" if r["source"] in ("e1_context_r2", "fraudr1_all") else "official")})
        return out

    train_examples = build_neural_examples(manifest, max_length=args.max_length,
                                           use_teacher_soft=(args.setting != "gold"),
                                           use_pairwise=(args.setting == "full_distill"))
    dev_all = build_neural_examples(with_teacher(dev_rows), max_length=args.max_length,
                                    use_teacher_soft=True, use_pairwise=False)
    if args.eval_subset and len(dev_all) > args.eval_subset:
        rng = __import__("random").Random(20260804)
        dev_examples = rng.sample(dev_all, args.eval_subset)
        print(f"dev eval subset: {len(dev_all)} -> {len(dev_examples)}")
    else:
        dev_examples = dev_all
    test_examples = build_neural_examples(with_teacher(test_rows), max_length=args.max_length,
                                          use_teacher_soft=True, use_pairwise=False)

    if args.gold_fraction < 1.0:
        rng = __import__("random").Random(20260804)
        keep = rng.sample(train_examples, int(len(train_examples) * args.gold_fraction))
        train_examples = keep
        print(f"low-label: gold fraction {args.gold_fraction} -> {len(train_examples)} rows")

    class SimpleDataset(Dataset):
        def __init__(self, exs): self.exs = exs
        def __len__(self): return len(self.exs)
        def __getitem__(self, i): return self.exs[i]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = NeuralStudentConfig(model_name=args.model_name, architecture=args.architecture,
                              max_length=args.max_length, lora_r=args.lora_r, lora_alpha=args.lora_alpha * 2)
    loss_fn = FraudDistillLoss(lambda_gold={"gold": 1.0, "soft_distill": 0.75, "full_distill": 0.65}[args.setting],
                               lambda_soft={"gold": 0.0, "soft_distill": 0.25, "full_distill": 0.25}[args.setting],
                               lambda_pair={"gold": 0.0, "soft_distill": 0.0, "full_distill": 0.10}[args.setting],
                               temperature=args.temperature, pair_margin=args.pair_margin)

    def collate(batch): return neural_collate(batch, tokenizer, max_length=args.max_length, architecture=args.architecture)

    eval_batch = max(args.micro_batch * 8, 16)
    dev_loader = DataLoader(SimpleDataset(dev_examples), batch_size=eval_batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(SimpleDataset(test_examples), batch_size=eval_batch, shuffle=False, collate_fn=collate)

    if args.eval_only:
        model = build_neural_student(cfg, freeze_base=True, device=device)
        m = evaluate_neural(model, test_loader, loss_fn, device, args.architecture)
        print("ZERO-SHOT TEST:", json.dumps(m, ensure_ascii=False))
        (out_root / f"zero_shot_{args.architecture}.json").write_text(json.dumps({"setting": "zero_shot", **m}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    train_loader = DataLoader(SimpleDataset(train_examples), batch_size=args.micro_batch, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    print(f"neural train: rows={len(train_examples)} dev={len(dev_examples)} test={len(test_examples)} "
          f"setting={args.setting} arch={args.architecture} seed={args.seed}")

    model = build_neural_student(cfg, freeze_base=False, device=device)
    t0 = time.time()
    run_dir = out_root / f"{args.setting}_{args.architecture}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_state, history = train_neural(
        model, train_loader, dev_loader, loss_fn, tokenizer,
        epochs=args.epochs, lr_lora=args.lr_lora, lr_head=args.lr_head,
        grad_accum=max(1, args.effective_batch // args.micro_batch),
        eval_steps=args.eval_steps, patience=args.patience, seed=args.seed,
        out_dir=run_dir, device=device, architecture=args.architecture,
        resume=args.resume, max_steps=args.max_steps)
    print(f"training wall time: {time.time() - t0:.0f}s")

    test_m = evaluate_neural(model, test_loader, loss_fn, device, args.architecture)
    print("TEST:", json.dumps(test_m, ensure_ascii=False))
    save_checkpoint(model, tokenizer, out_root / f"{args.setting}_{args.architecture}_seed{args.seed}_final", args.architecture)
    (out_root / f"{args.setting}_{args.architecture}_seed{args.seed}.json").write_text(
        json.dumps({"setting": args.setting, "architecture": args.architecture, "seed": args.seed,
                    "rows": len(train_examples), "test": test_m,
                    "history": history, "wall_seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def main() -> None:
    dataset, teacher_map, _ = load_all()
    train_rows = [r for r in dataset if r["split"] == "train"]
    dev_rows = [r for r in dataset if r["split"] == "dev"]
    test_rows = [r for r in dataset if r["split"] == "test"]
    print(f"train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)} teacher_rows={len(teacher_map)}")

    missing = [r["id"] for r in train_rows if r["id"] not in teacher_map]
    if missing:
        print(f"WARNING: {len(missing)} train rows lack teacher signals; run --mode train first")
        sys.exit(2)

    settings = ["S0_gold", "S1_hard_teacher", "S2_score_distill", "S3_type_distill", "S4_evidence_distill"]
    rows_out = []
    for name in settings:
        agg = run_setting(name, train_rows, test_rows, dev_rows, teacher_map, SEEDS)
        rows_out.append(agg)
        print(json.dumps(agg, ensure_ascii=False))

    METRICS = OUT_ROOT / "metrics"
    METRICS.mkdir(parents=True, exist_ok=True)
    cols = ["setting", "seeds"]
    for k in rows_out[0]:
        if k not in cols:
            cols.append(k)
    with (METRICS / "student_gradient.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows_out:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("wrote", METRICS / "student_gradient.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="linear", choices=["linear", "neural"])
    ap.add_argument("--setting", default="full_distill", choices=["gold", "soft_distill", "full_distill"])
    ap.add_argument("--architecture", default="standard", choices=["standard", "interaction"])
    ap.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--manifest", default="data/prepared/exp3_neural_student/train_manifest.jsonl")
    ap.add_argument("--seeds", default="11,37,71")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--pair-margin", type=float, default=0.20)
    ap.add_argument("--gold-fraction", type=float, default=1.0)
    ap.add_argument("--eval-subset", type=int, default=0, help="cap dev eval rows (0 = full dev)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", default=None, help="resume checkpoint json from a previous run")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    if args.backend == "neural":
        for seed in [int(x) for x in args.seeds.split(",")]:
            args.seed = seed
            main_neural(args)
    else:
        main()