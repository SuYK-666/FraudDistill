# -*- coding: utf-8 -*-
"""Exp2 balanced diagnostics Phase 8: re-run local official baselines on the
new balanced q+y (guide 15.3). Aegis reuses the frozen 813 baseline.

  fraudr1 : local Official Judge / GPTCheck_v1 (deepseek, max_tokens=32)
  orbench : local Official Response Checker (deepseek)
  dna     : LibrAI Longformer-Harmful (local transformers, zero API)
  aegis   : reuse existing nemoguard 813 predictions

Usage:
  python scripts/run_exp2_balanced_baselines.py            # all
  python scripts/run_exp2_balanced_baselines.py --bench dna
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "frauddistill" / "exp2_cross_benchmark" / "baselines"))

from frauddistill.exp2_cross_benchmark import baselines as bl

BASE = REPO / "experiments" / "exp2_prior_work_comparison" / "balanced_design"
MANIFEST = BASE / "manifests" / "balanced_test_manifest.jsonl"
IN_DIR = BASE / "baselines" / "inputs"
OUT_DIR = BASE / "baselines"
AEGIS_SRC = REPO / "experiments" / "exp2_prior_work_comparison" / "aegis2" / "baseline_predictions" / "aegis_nemoguard_predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_inputs() -> dict[str, Path]:
    rows = read_jsonl(MANIFEST)
    IN_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    fr = []
    for r in rows:
        if r["source"] != "fraudr1":
            continue
        lang = "Chinese" if "Chinese" in str(r.get("language", "")) else "English"
        fr.append({"id": r["sample_id"], "benchmark": "fraudr1", "group_id": r["group_id"],
                   "language": lang, "query": r["query"], "answer": r["answer"],
                   "answer_status": "frozen"})
    p = IN_DIR / "fraudr1_input.jsonl"
    write_jsonl(p, fr)
    out["fraudr1"] = p
    print(f"[prep] fraudr1 rows={len(fr)}")
    orb = []
    for r in rows:
        if r["source"] != "orbench":
            continue
        orb.append({"id": r["sample_id"], "benchmark": "orbench", "group_id": r["group_id"],
                    "query": r["query"], "answer": r["answer"], "answer_status": "frozen",
                    "metadata": {"orbench_prompt_type": r.get("stratum") or r.get("request_gold_type") or "hard_safe"}})
    p = IN_DIR / "orbench_input.jsonl"
    write_jsonl(p, orb)
    out["orbench"] = p
    print(f"[prep] orbench rows={len(orb)}")
    dna = []
    for r in rows:
        if r["source"] != "do_not_answer":
            continue
        dna.append({"id": r["sample_id"], "benchmark": "do_not_answer", "group_id": r["group_id"],
                    "query": r["query"], "answer": r["answer"], "answer_status": "frozen",
                    "gold_binary": None, "gold_type": None})
    p = IN_DIR / "dna_input.jsonl"
    write_jsonl(p, dna)
    out["dna"] = p
    print(f"[prep] dna rows={len(dna)}")
    return out


def run_fraud(input_path: Path) -> None:
    out = OUT_DIR / "fraudr1_baseline_predictions.jsonl"
    from frauddistill.exp2_cross_benchmark.baselines import fraudr1_judge
    asyncio.run(fraudr1_judge.generate(limit=None, concurrency=120,
                                       input_path=str(input_path), out_path_override=str(out)))
    print(f"[baseline fraudr1] -> {out}")


def run_or(input_path: Path) -> None:
    out = OUT_DIR / "orbench_baseline_predictions.jsonl"
    from frauddistill.exp2_cross_benchmark.baselines import orbench_checker
    if hasattr(orbench_checker, "generate_overrides"):
        asyncio.run(orbench_checker.generate_overrides(str(input_path), str(out)))
    else:
        raise RuntimeError("orbench_checker needs generate_overrides (patched)")
    print(f"[baseline orbench] -> {out}")


def run_dna(input_path: Path) -> None:
    out = OUT_DIR / "dna_baseline_predictions.jsonl"
    from frauddistill.exp2_cross_benchmark.baselines import dna_longformer
    if hasattr(dna_longformer, "generate_overrides"):
        dna_longformer.generate_overrides(str(input_path), str(out))
    else:
        raise RuntimeError("dna_longformer needs generate_overrides (patched)")
    print(f"[baseline dna] -> {out}")


def run_aegis() -> None:
    out = OUT_DIR / "aegis_baseline_predictions.jsonl"
    manifest_ids = {r["sample_id"] for r in read_jsonl(MANIFEST) if r.get("source") == "aegis2"}
    if AEGIS_SRC.exists():
        by_id = {r["id"]: r for r in read_jsonl(AEGIS_SRC) if r.get("id") in manifest_ids}
        rows = [by_id[sid] for sid in manifest_ids if sid in by_id]
        write_jsonl(out, rows)
        print(f"[baseline aegis] reused {AEGIS_SRC.name} rows={len(rows)} (manifest matched, deduped)")
    else:
        print(f"[baseline aegis] WARNING {AEGIS_SRC} missing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["fraudr1", "orbench", "dna", "aegis", "all"], default="all")
    args = ap.parse_args()
    inputs = prepare_inputs()
    if args.bench in ("all", "fraudr1"):
        run_fraud(inputs["fraudr1"])
    if args.bench in ("all", "orbench"):
        run_or(inputs["orbench"])
    if args.bench in ("all", "dna"):
        run_dna(inputs["dna"])
    if args.bench in ("all", "aegis"):
        run_aegis()


if __name__ == "__main__":
    main()
