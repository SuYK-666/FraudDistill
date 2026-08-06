# -*- coding: utf-8 -*-
"""Prepare the Exp2 FINAL baseline subsets (guide 9/18) by aligning the
reused original-work predictions to the final manifests by sample_id.

Outputs under experiments/exp2_prior_work_comparison/baselines/:
  fraudr1_official_judge_3000.jsonl / orbench_official_checker_3000.jsonl
  dna_longformer_5634.jsonl / aegis_nemoguard_response_813.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "experiments" / "exp2_prior_work_comparison"
MANIFEST_DIR = BASE / "manifests"
BASELINES_DIR = BASE / "baselines"

SPECS = {
    "fraudr1": ("fraudr1_content_final_3000.jsonl",
                BASE / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl",
                "fraudr1_official_judge_3000.jsonl"),
    "orbench": ("orbench_final_3000.jsonl",
                BASE / "orbench" / "baseline_predictions" / "orbench_response_checker_predictions.jsonl",
                "orbench_official_checker_3000.jsonl"),
    "do_not_answer": ("dna_final_5634.jsonl",
                      BASE / "do_not_answer" / "baseline_predictions" / "dna_longformer_harmful_predictions.jsonl",
                      "dna_longformer_5634.jsonl"),
    "aegis2": ("aegis_response_test_813.jsonl",
               BASE / "aegis2" / "baseline_predictions" / "aegis_nemoguard_predictions.jsonl",
               "aegis_nemoguard_response_813.jsonl"),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    for b, (mf, src, out_name) in SPECS.items():
        mani = read_jsonl(MANIFEST_DIR / mf)
        base = {str(r["id"]): r for r in read_jsonl(src)}
        out, missing = [], []
        for m in mani:
            r = base.get(m["sample_id"])
            if r is None:
                missing.append(m["sample_id"])
                continue
            rec = {**r, "qy_hash": m["qy_hash"]}
            out.append(rec)
        out.sort(key=lambda r: r["id"])
        with (BASELINES_DIR / out_name).open("w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[baselines:{b}] manifest={len(mani)} written={len(out)} missing={len(missing)}")


if __name__ == "__main__":
    main()