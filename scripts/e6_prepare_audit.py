# -*- coding: utf-8 -*-
"""E6 S5: freeze Silver audit set (Random-30 + Boundary-10 per model), offline."""
from __future__ import annotations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6_common import (DATA_DIR, GEN_DIR, STUDENT_DIR, SILVER_DIR, PROTOCOL_DIR, SEED,
                       STUDENT_THRESHOLD, read_jsonl, write_jsonl, write_json)

def main():
    manifest = read_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl")
    mrow = {r["prompt_id"]: r for r in manifest}
    registry = json.loads((PROTOCOL_DIR / "model_registry_frozen.json").read_text(encoding="utf-8"))
    slots = registry.get("slots") or registry
    preds = read_jsonl(STUDENT_DIR / "predictions_all.jsonl")
    pred_map = {(p["slot"], p["prompt_id"]): p for p in preds}

    rng = random.Random(SEED)
    audit_rows = []
    counts = {}
    for slot_key in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        spec = slots[slot_key]
        gen = {}
        for r in read_jsonl(GEN_DIR / "per_model" / f"{slot_key}.jsonl"):
            if r.get("generation_success"):
                gen[r["prompt_id"]] = r
        pids = sorted(gen.keys())
        rng.shuffle(pids)  # deterministic given SEED
        rand30 = pids[:30]
        rest = pids[30:]
        # boundary: 10 smallest |risk - threshold|
        rest_scored = [(p, abs(pred_map[(slot_key, p)]["risk_score"] - STUDENT_THRESHOLD)) for p in rest]
        rest_scored.sort(key=lambda x: (x[1], x[0]))
        bnd10 = [p for p, _ in rest_scored[:10]]
        for split, picked in (("random", rand30), ("boundary", bnd10)):
            for i, pid in enumerate(picked, 1):
                m = mrow[pid]
                g = gen[pid]
                pr = pred_map[(slot_key, pid)]
                audit_rows.append({
                    "audit_id": f"{slot_key}-{split[:4]}-{i:03d}",
                    "slot": slot_key,
                    "provider": spec["provider"],
                    "requested_model": spec["requested_model"],
                    "served_model": g.get("served_model"),
                    "split": split,
                    "prompt_id": pid,
                    "prompt_stratum": m["prompt_stratum"],
                    "prompt_risk_label": m["prompt_risk_label"],
                    "should_refuse": m.get("should_refuse"),
                    "language": m["language"],
                    "user_query": m["user_query"],
                    "target_model_answer": g["target_model_answer"],
                    "risk_score": pr["risk_score"],
                    "threshold": STUDENT_THRESHOLD,
                    "student_pred_label": pr["pred_label"],
                    "input_tokens_qy": pr.get("input_tokens_qy"),
                    "truncated": pr.get("truncated"),
                })
        counts[slot_key] = {"successful": len(pids), "random": len(rand30), "boundary": len(bnd10)}
        print(f"[S5] {slot_key}: successful={len(pids)} random={len(rand30)} boundary={len(bnd10)}")

    audit_rows.sort(key=lambda r: (r["slot"], r["split"], r["audit_id"]))
    write_jsonl(SILVER_DIR / "audit_set.jsonl", audit_rows)
    # public version: drop user_query
    pub = [{k: v for k, v in r.items() if k != "user_query"} for r in audit_rows]
    write_jsonl(SILVER_DIR / "audit_set_public.jsonl", pub)
    write_json(SILVER_DIR / "audit_coverage.json", {
        "n_total": len(audit_rows),
        "n_random": sum(1 for r in audit_rows if r["split"] == "random"),
        "n_boundary": sum(1 for r in audit_rows if r["split"] == "boundary"),
        "per_model": counts,
        "seed": SEED,
    })
    # boundary min/max distance sanity
    bnd = [r for r in audit_rows if r["split"] == "boundary"]
    print(f"[S5] boundary rows={len(bnd)} |risk-thr| min={min(abs(r['risk_score']-STUDENT_THRESHOLD) for r in bnd):.4f} "
          f"max={max(abs(r['risk_score']-STUDENT_THRESHOLD) for r in bnd):.4f}")
    # overlap check
    keys = [(r["slot"], r["prompt_id"], r["split"]) for r in audit_rows]
    print(f"[S5] unique (slot,pid,split)={len(set(keys))} total={len(keys)}")
    print(f"[S5] DONE -> {SILVER_DIR / 'audit_set.jsonl'}")

if __name__ == "__main__":
    main()

