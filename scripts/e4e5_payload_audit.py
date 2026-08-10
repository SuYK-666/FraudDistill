# -*- coding: utf-8 -*-
"""E4 v2 payload audit: verify the inference input contains only context/q/y text
(no source/benchmark/split/target-model/gold metadata) for every formal row."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from frauddistill.e4e5_v2.schemas import read_jsonl
from frauddistill.student.dataset import neural_input_text

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

def main():
    rows = read_jsonl(BASE / "manifests/frozen_test.jsonl") + read_jsonl(BASE / "manifests/calibration.jsonl")
    # distinctive metadata tokens that must never appear in the input text:
    # row id / family id / benchmark-source versions. Generic words (e.g. "safe")
    # are allowed because they can legitimately occur inside answers.
    source_values = sorted({str(r.get("source_version") or "") for r in rows if r.get("source_version")})
    probes = []
    for r in rows:
        probes.append(("id", str(r.get("id") or "")))
        probes.append(("family_id", str(r.get("family_id") or "")))
    for sv in source_values:
        if sv:
            probes.append(("source_version", sv))
    bad = []
    for r in rows:
        text = neural_input_text(str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""),
                                 r.get("context"))
        if not text.startswith("Task:"):
            bad.append({"id": r["id"], "issue": "prefix"})
            continue
        for tag, probe in probes:
            if probe and len(probe) >= 6 and probe in text:
                bad.append({"id": r["id"], "issue": f"metadata:{tag}"})
                break
    out = {"n": len(rows), "n_bad": len(bad), "bad": bad[:20],
           "source_probes": source_values,
           "template": "neural_input_text(context, q, y) only",
           "checked_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")}
    out_dir = BASE / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "payload_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[payload-audit] n={out['n']} n_bad={out['n_bad']} -> {out_dir / 'payload_audit.json'}")

if __name__ == "__main__":
    main()
