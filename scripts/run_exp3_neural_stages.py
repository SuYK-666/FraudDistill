# -*- coding: utf-8 -*-
"""Sequential runner for neural-student training stages (guide 18/19/22).

Runs each stage via scripts/train_exp3_students.py with fixed common args,
logging per stage; retries once with --resume on crash. No API usage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/train_exp3_students.py"
OUT = REPO / "outputs"
MANIFEST_DEFAULT = "data/prepared/exp3_neural_student/train_manifest.jsonl"

COMMON = [
    "--backend", "neural",
    "--architecture", "standard",
    "--seeds", "11",
    "--max-length", "384",
    "--micro-batch", "2",
    "--effective-batch", "32",
    "--eval-steps", "200",
    "--patience", "2",
    "--eval-subset", "200",
    "--lora-r", "32",
    "--lora-alpha", "64",
]

STAGES = {
    # name -> (extra args, out-root, manifest override)
    # fairness (guide 18.6): full/semi stages use the same step budget;
    # low-label stages scale epochs up so step counts stay comparable.
    "gold": (["--setting", "gold", "--epochs", "2"], None, None),
    "soft_distill": (["--setting", "soft_distill", "--epochs", "2"], None, None),
    # final_student: reserved interface for the Final Student retrain (follow-up
    # guide); same recipe as soft_distill until the retrain recipe is finalized.
    "final_student": (["--setting", "final_student", "--epochs", "2"], None, None),
    # max_steps counts optimizer steps: 2236 micro-batches / 16 accum = 140
    "full_distill": (["--setting", "full_distill", "--epochs", "1", "--max-steps", "140"], None,
                     "data/prepared/exp3_neural_student/train_manifest_expanded.jsonl"),
    "gold10": (["--setting", "gold", "--gold-fraction", "0.1", "--epochs", "8"], "lowlabel", None),
    "gold25": (["--setting", "gold", "--gold-fraction", "0.25", "--epochs", "4"], "lowlabel", None),
    "gold50": (["--setting", "gold", "--gold-fraction", "0.5", "--epochs", "2"], "lowlabel", None),
    "soft10": (["--setting", "soft_distill", "--gold-fraction", "0.1", "--epochs", "8"], "lowlabel", None),
    "soft25": (["--setting", "soft_distill", "--gold-fraction", "0.25", "--epochs", "4"], "lowlabel", None),
    "soft50": (["--setting", "soft_distill", "--gold-fraction", "0.5", "--epochs", "2"], "lowlabel", None),
}


def run_stage(name: str, manifest: str, extra: list[str], out_root: str | None = None,
              manifest_override: str | None = None) -> bool:
    if manifest_override:
        manifest = manifest_override
    log = (OUT / f"train_stage_{name}.log").open("a", encoding="utf-8")
    err = (OUT / f"train_stage_{name}.err").open("a", encoding="utf-8")
    base = [sys.executable, str(SCRIPT), "--manifest", manifest] + COMMON + extra
    if out_root:
        base += ["--out-root", str(REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student" / out_root)]
    env = {"PYTHONUNBUFFERED": "1"}
    for attempt in (1, 2):
        print(f"[stage {name}] attempt {attempt}: {' '.join(base)}", flush=True)
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt} =====\n")
        log.flush()
        p = subprocess.Popen(base, stdout=log, stderr=err, env=env)
        rc = p.wait()
        if rc == 0:
            print(f"[stage {name}] OK rc=0", flush=True)
            log.close(); err.close()
            return True
        print(f"[stage {name}] FAIL rc={rc}; will retry with --resume", flush=True)
        setting = extra[1]  # value after --setting
        gf_tag = ""
        if "--gold-fraction" in extra:
            gf_tag = f"_gf{extra[extra.index('--gold-fraction') + 1]}"
        resume_dir = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student"
        if out_root:
            resume_dir = resume_dir / out_root
        resume = resume_dir / f"{setting}_standard_seed11{gf_tag}" / "resume.pt"
        if not resume.exists():
            print(f"[stage {name}] no resume.pt at {resume}", flush=True)
            continue
        base = base + ["--resume", str(resume)]
    log.close(); err.close()
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma list: gold,soft_distill,full_distill,gold10,...")
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--settings-json", default=None, help="optional extra stages json [{name, args: [...]}]")
    args = ap.parse_args()

    plan = []
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if name not in STAGES:
            print(f"unknown stage {name}; skip")
            continue
        plan.append((name, *STAGES[name]))
    if args.settings_json:
        extra = json.loads(Path(args.settings_json).read_text(encoding="utf-8"))
        plan.extend((e["name"], e["args"], e.get("out_root"), e.get("manifest")) for e in extra)

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, extra, out_root, manifest_override in plan:
        ok = run_stage(name, args.manifest, extra, out_root, manifest_override)
        results[name] = "ok" if ok else "failed"
        print(f"[stage {name}] -> {results[name]}", flush=True)
    (OUT / "neural_stage_results.json").write_text(
        json.dumps({"results": results, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("ALL DONE:", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
