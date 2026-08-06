# -*- coding: utf-8 -*-
"""Offline multi-head rescoring (guide sections 20-22, 24, 32.5).

Zero-API. Combines the saved specialist evidence (agent_fraud_json /
agent_refusal_json / agent_context_json) into task-aligned risk scores for
each benchmark:

  Fraud-R1   -> fraud_assistance_score
  OR native  -> did_refuse_score (refusal behavior)
  OR safety  -> over_refusal_score + harmful_compliance_score
  DNA        -> general_harmful_compliance_score
  Aegis resp -> general_harmful_compliance_score (793/813 rows augmented
                from the Exp3 agent-prediction index)
  Aegis prom -> prompt_risk_score

Outputs: experiments/exp2_prior_work_comparison/offline_rescore/deterministic/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.adapter_model import load_models  # noqa: E402
from frauddistill.exp2_static_repair.evidence import row_features  # noqa: E402
from frauddistill.exp2_static_repair.heads import all_heads  # noqa: E402
from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402

EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"
TEACHER_T6_DIR = EXPERIMENT_DIR / "teacher_predictions_t6"
OUT_DIR = EXPERIMENT_DIR / "offline_rescore" / "deterministic"
EXP3_AGENT_DIR = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "agent_predictions"

T6_FILES = {
    "fraudr1": "fraudr1_t6_predictions.jsonl",
    "orbench": "orbench_t6_predictions.jsonl",
    "do_not_answer": "do_not_answer_t6_predictions.jsonl",
    "aegis2_response": "aegis2_t6_predictions.jsonl",
    "aegis2_prompt": "aegis2_t6_prompt_predictions.jsonl",
    "aegis2_validation": "aegis_validation_t6_predictions.jsonl",
}

SHARED_ADAPTER_DIR = EXPERIMENT_DIR / "offline_rescore" / "shared_adapter"

# Benchmark -> adapter head used in shared-adapter mode (guide 23.6, 24).
ADAPTER_HEAD = {
    "fraudr1": "fraud",
    "orbench": "harmful_compliance",
    "do_not_answer": "harmful_compliance",
    "aegis2_response": "harmful_compliance",
    "aegis2_prompt": "harmful_compliance",
    "aegis2_validation": "harmful_compliance",
}

SCORE_HEAD = {
    "fraudr1": "fraud_assistance_score",
    "orbench": "harmful_compliance_score",
    "do_not_answer": "general_harmful_compliance_score",
    "aegis2_response": "general_harmful_compliance_score",
    "aegis2_prompt": "prompt_risk_score",
    "aegis2_validation": "general_harmful_compliance_score",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_exp3_agent_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for fname in ("train.jsonl", "test.jsonl", "dev.jsonl"):
        p = EXP3_AGENT_DIR / fname
        if not p.exists():
            continue
        for l in p.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            index.setdefault(r.get("id", ""), r)
    return index


def merge_exp3_agents(row: dict, index: dict[str, dict]) -> dict:
    """Augment an Exp2 row with Exp3-generated specialist evidence."""
    if row.get("agent_fraud_json") or row.get("agent_refusal_json"):
        return row, False
    src = index.get(row.get("id", ""))
    if src is None:
        return row, False
    merged = dict(row)
    merged["agent_fraud_json"] = src.get("fraud") or {}
    merged["agent_refusal_json"] = src.get("refusal") or {}
    merged["agent_context_json"] = src.get("context") or {}
    merged["_specialist_source"] = "exp3_agent_predictions"
    return merged, True


def load_manifest_index() -> dict[str, dict]:
    idx = {}
    p = EXPERIMENT_DIR / "manifests" / "full_manifest.jsonl"
    if not p.exists():
        return idx
    for l in p.open(encoding="utf-8"):
        r = json.loads(l)
        idx[r.get("sample_id", "")] = r
    return idx


def rescore(name: str, rows: list[dict], manifest: dict[str, dict], exp3_index: dict[str, dict], models: dict | None = None) -> list[dict]:
    out = []
    n_augmented = 0
    for r in rows:
        sid = r.get("id") or r.get("sample_id")
        m = manifest.get(sid, {})
        augmented = False
        if name == "aegis2_response":
            r, augmented = merge_exp3_agents(r, exp3_index)
            n_augmented += 1 if augmented else 0

        heads = all_heads(r)
        if models is not None:
            head = ADAPTER_HEAD[name]
            model = models.get(head)
            if model is not None:
                x = row_features(r).reshape(1, -1)
                heads["adapter_score"] = float(model.predict_proba(x)[0, 1])
                heads["adapter_head"] = head
            else:
                heads["adapter_score"] = None
                heads["adapter_head"] = head
        gold = m.get("official_gold_binary")
        if name == "aegis2_response":
            gold = m.get("official_response_label") if m.get("has_response") else m.get("official_prompt_label")
        elif name == "aegis2_prompt":
            gold = m.get("official_prompt_label")

        rec = {
            "sample_id": sid,
            "benchmark": name.split("_")[0] if name != "aegis2_prompt" and name != "aegis2_validation" else "aegis2",
            "track": "response" if "response" in name or name == "aegis2_validation" else ("prompt" if "prompt" in name else "response"),
            "group_id": r.get("group_id") or m.get("group_id", ""),
            "query": r.get("query", "")[:3000],
            "answer": r.get("answer", "")[:3000],
            "gold": gold,
            "gold_type": m.get("official_gold_type"),
            "prompt_type": m.get("official_gold_type") if name == "orbench" else r.get("prompt_type"),
            "category": m.get("official_category") or m.get("category"),
            "target_model": m.get("target_model") or r.get("target_model"),
            "overlap_exp3": bool(m.get("overlap_exp3") or r.get("overlap_exp3")),
            "teacher_prediction_binary": r.get("prediction_binary"),
            "teacher_risk_score": r.get("risk_score"),
            "teacher_type": r.get("prediction_type"),
            "specialist_available": bool(r.get("agent_fraud_json") or r.get("agent_refusal_json")),
            "specialist_source": r.get("_specialist_source", "t6_run"),
            **heads,
        }
        out.append(rec)
    print(f"{name}: n={len(out)} augmented_from_exp3={n_augmented} specialist_available={sum(1 for r in out if r['specialist_available'])}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="deterministic", choices=["deterministic", "shared-adapter"])
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    manifest = load_manifest_index()
    exp3_index = build_exp3_agent_index()
    print(f"manifest rows: {len(manifest)}, exp3 agent rows: {len(exp3_index)}")

    models = None
    out_dir = OUT_DIR
    if args.mode == "shared-adapter":
        models = load_models(SHARED_ADAPTER_DIR)
        out_dir = SHARED_ADAPTER_DIR
        print("loaded shared adapter:", {k: (v is not None) for k, v in models.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fname in T6_FILES.items():
        rows = read_jsonl(TEACHER_T6_DIR / fname)
        recs = rescore(name, rows, manifest, exp3_index, models=models)
        (out_dir / f"{name}_rescore.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in recs),
            encoding="utf-8",
        )
    print("outputs ->", out_dir)


if __name__ == "__main__":
    main()
