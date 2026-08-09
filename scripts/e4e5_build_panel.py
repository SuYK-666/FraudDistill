# -*- coding: utf-8 -*-
"""E4/E5 v2 panel builder: candidates -> G1/G2 gold -> exposure audit ->
family-level pilot/calibration/test split -> manifests + locks.
Usage:
  python scripts/e4e5_build_panel.py [--g2] [--out outputs/exp4_unseen_student_v2/e4v2_<ts>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml
from frauddistill.e4e5_v2.panel_builder import (
    aegis_candidates, apply_g1, check_quotas, exposure_audit_rows,
    generated_candidates, pku_candidates, split_panel, write_manifests,
)
from frauddistill.e4e5_v2.exposure_registry import build_registry
from frauddistill.e4e5_v2.model_registry import (
    build_label_policy_lock, build_model_lock, build_protocol_lock,
)
from frauddistill.e4e5_v2.schemas import read_jsonl, write_jsonl

U1_MODELS = ["qwen2_5_7b", "llama3_1_8b"]
U1_CATS = ["elder_health_product", "naked_chat_sextortion"]
U3_MODELS = ["smollm2_1_7b", "phi3_5_mini"]
U2_AEGIS = REPO / "data/raw/aegis/validation.json"
U2_PKU = sorted((REPO / "data/raw/pku_saferlhf").glob("Alpaca*-7B_test.jsonl"))


def load_u1(out_dir: Path) -> list[dict]:
    rows = []
    for m in U1_MODELS:
        for c in U1_CATS:
            for mode in ("chat", "cont", "sys_chat"):
                p = out_dir / "generated_u1" / f"{m}_{c}_{mode}.jsonl"
                gen = read_jsonl(p)
                rows += generated_candidates(
                    gen, "U1_category", "u1_synthetic_gen_v1", m)
    return rows


def load_u3(out_dir: Path) -> list[dict]:
    rows = []
    for m in U3_MODELS:
        for mode in ("chat", "cont", "sys_chat"):
            p = out_dir / "generated_u3" / f"{m}_{mode}.jsonl"
            gen = read_jsonl(p)
            rows += generated_candidates(gen, "U3_target_style", m, m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", action="store_true", help="run local blind judge (G2) on ambiguous rows")
    ap.add_argument("--out", default=None, help="output root (default: outputs/exp4_unseen_student_v2/e4v2_<ts>)")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs/experiments/exp4_unseen_student_v2.yaml").read_text(encoding="utf-8"))
    gen_dir = REPO / "data/prepared/e4e5_v2"

    # ---- candidates ----
    cands = []
    u2 = aegis_candidates(U2_AEGIS) + pku_candidates(U2_PKU)
    u1 = load_u1(gen_dir)
    u3 = load_u3(gen_dir)
    cands = u2 + u1 + u3
    print(f"[panel] U2={len(u2)} U1={len(u1)} U3={len(u3)} total={len(cands)}")

    # ---- G1 provisional gold (U2 rows already have official labels) ----
    apply_g1(cands)
    lbl = Counter(r["gold_label"] for r in cands)
    print(f"[panel] gold labels after G1: {dict(lbl)}")

    # ---- G2 blind judge for ambiguous (batched) ----
    if args.g2:
        from frauddistill.e4e5_v2.gold_mapper import judge_with_model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        judge_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tok = AutoTokenizer.from_pretrained(judge_ckpt)
        model = AutoModelForCausalLM.from_pretrained(judge_ckpt, torch_dtype=torch.bfloat16)
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def batched_predict(prompts: list[str], batch_size: int = 8, max_new: int = 160) -> list[str]:
            outs = []
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1400)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
                for k, g in enumerate(gen):
                    outs.append(tok.decode(g[enc["input_ids"][k].shape[0]:], skip_special_tokens=True))
            return outs

        from frauddistill.e4e5_v2.gold_mapper import JUDGE_PROMPT
        ambiguous = [r for r in cands if not r.get("gold_label")]
        print(f"[panel] G2 judging {len(ambiguous)} ambiguous rows (batched)")
        prompts = [JUDGE_PROMPT.format(query=str(r.get("user_query") or "")[:2500],
                                       answer=str(r.get("target_model_answer") or "")[:2500])
                   for r in ambiguous]
        raws = batched_predict(prompts)
        for r, raw in zip(ambiguous, raws):
            j = judge_with_model(lambda x, raw=raw: raw, r["user_query"], r["target_model_answer"])
            if j["gold_label"]:
                r["gold_label"] = j["gold_label"]
                r["gold_type"] = j["gold_type"]
                r["gold_source"] = "blind_judge"
                r["metadata"]["g2"] = j
            else:
                r["gold_label"] = "safe" if r["metadata"].get("g1", {}).get("refusal_hits", 0) >= 1 else "unsafe"
                r["gold_type"] = "safe" if r["gold_label"] == "safe" else "fraud_assistance"
                r["gold_source"] = "fallback_heuristic"
                r["metadata"]["g2"] = j
        del model, tok
        import gc; gc.collect()

    # ---- exposure audit (per-row, drop all failures) ----
    registry, loaded = build_registry(REPO)
    kept, dropped = [], []
    for r in cands:
        a = registry.audit_candidate(r)
        if a["passed"]:
            kept.append(r)
        else:
            dropped.append({"id": r["id"], "primary_shift": r["primary_shift"],
                            "source": r["source"], **{k: v for k, v in a.items() if k != "id"}})
    audit = {"n": len(cands), "n_failed": len(dropped), "passed": len(dropped) == 0,
             "failures": dropped[:200], "n_kept": len(kept)}
    print(f"[panel] exposure audit: {len(dropped)}/{len(cands)} dropped")
    cands = kept

    # ---- per-cell family cap (keep pools balanced, pilot manageable) ----
    from collections import defaultdict
    MAX_FAMILIES_PER_CELL = 600
    cell_fams: dict = defaultdict(list)
    for r in cands:
        cell = r["source"] if r["primary_shift"] != "U1_category" else r["fraud_category"]
        cell_fams[(r["primary_shift"], cell)].append(r)
    capped = []
    for (shift, cell), rows in cell_fams.items():
        by_fam = defaultdict(list)
        for r in rows:
            by_fam[r["family_id"]].append(r)
        fams_list = sorted(by_fam.keys(), key=lambda f: (sum(1 for r in by_fam[f] if r["gold_label"] == "unsafe") > 0
                                                          and sum(1 for r in by_fam[f] if r["gold_label"] == "safe") > 0, f), reverse=True)
        # keep families with both labels first, then balanced order; cap count
        fams_list = fams_list[:MAX_FAMILIES_PER_CELL]
        for f in fams_list:
            capped.extend(by_fam[f])
    cands = capped
    print(f"[panel] after family cap: {len(cands)} rows")

    # ---- split ----
    split = split_panel(cands, cfg)
    for name in ("pilot", "calibration", "test"):
        rows = split[name]
        c = Counter((r["primary_shift"], r["gold_label"]) for r in rows)
        print(f"[panel] {name}: n={len(rows)} {dict(c)}")

    # ---- quotas ----
    qrep = check_quotas(split, cfg)
    ok = True
    for name in ("calibration", "test"):
        for k, v in qrep[name].items():
            print(f"[panel] quota {name} {k}: {v}")
    # shortfall check vs config expectations
    for name, key_n, expect in (("calibration", "calibration_n", 600), ("test", "frozen_test_n", 1200)):
        n = len(split[name])
        if n < expect:
            print(f"[panel] WARNING {name} n={n} < {expect}")
            ok = False
    if not ok:
        print("[panel] QUOTA SHORTFALL - generation still incomplete; write what we have")
    # per-shift per-label check (cal 100/100, test 200/200 per shift)
    for name, need in (("calibration", {"safe": 100, "unsafe": 100}), ("test", {"safe": 200, "unsafe": 200})):
        rows = split[name]
        for shift in ("U1_category", "U2_source", "U3_target_style"):
            cc = Counter(r["gold_label"] for r in rows if r["primary_shift"] == shift)
            print(f"[panel] {name}/{shift}: {dict(cc)} (need {need})")

    # ---- outputs ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else REPO / "outputs/exp4_unseen_student_v2" / f"e4v2_{ts}"
    proto_dir = out_root / "protocol"
    manifests_dir = out_root / "manifests"
    audits_dir = out_root / "audits"
    proto_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    hashes = write_manifests(split, manifests_dir)
    print(f"[panel] manifests written: {hashes}")

    # candidate pool snapshot + audit reports
    write_jsonl(out_root / "candidate_pool.jsonl", cands)
    write_jsonl(audits_dir / "exposure_audit.jsonl", [audit])
    write_jsonl(audits_dir / "shortcut_audit.jsonl", [{"note": "run after panel finalised"}])
    registry.save(out_root / "exposure_registry.jsonl")

    # gold audit stats
    g1 = Counter(r.get("gold_source") for r in cands)
    gold_audit = {"n": len(cands), "by_source": dict(g1)}
    write_jsonl(audits_dir / "gold_audit.jsonl", [gold_audit])
    print(f"[panel] gold audit: {gold_audit}")

    # locks
    protocol_id = out_root.name
    ml = build_model_lock(REPO, cfg, protocol_id, proto_dir)
    ll = build_label_policy_lock(REPO, proto_dir)
    pl = build_protocol_lock(REPO, cfg, protocol_id, proto_dir, ml, registry.summary())
    print(f"[panel] locks written: {protocol_id}")
    print(f"[panel] OUT: {out_root}")


if __name__ == "__main__":
    main()
