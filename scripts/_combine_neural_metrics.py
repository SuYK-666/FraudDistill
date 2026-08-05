# -*- coding: utf-8 -*-
"""Combine neural-student eval metrics into one canonical file (guide 3.8)."""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(r"experiments\exp3_agent_distillation_ablation\outputs\neural_student")
METRICS_DIR = Path(r"experiments\exp3_agent_distillation_ablation\outputs\metrics")

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None

names = ["gold", "soft", "full", "lowlabel_gold10", "zero_shot"]
out = {}
for name in names:
    m = load(BASE / f"eval_{name}" / "neural_student_metrics.json")
    if m:
        out[name] = m
    else:
        print("missing", name)

train = {}
for key, fname in [("gold", "gold_standard_seed11.json"), ("soft", "soft_distill_standard_seed11.json"),
                   ("full", "full_distill_standard_seed11.json"),
                   ("lowlabel_gold10", "lowlabel/gold_standard_seed11_gf0.1.json")]:
    d = load(BASE / fname)
    if d:
        train[key] = {"rows": d.get("rows"), "test": d.get("test"), "wall_seconds": d.get("wall_seconds"),
                      "gold_fraction": d.get("gold_fraction", 1.0), "setting": d.get("setting")}

canon = {"eval": out, "training_test": train}
(METRICS_DIR / "neural_student_metrics.json").write_text(
    json.dumps(canon, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", METRICS_DIR / "neural_student_metrics.json", "eval keys:", list(out.keys()))
