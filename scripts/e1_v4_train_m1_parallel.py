# -*- coding: utf-8 -*-
"""E1 v4 M1 XLM-R parallel training driver (CPU).

Splits the 5 seeds x 3 modes grid across --nshards workers; each worker writes
E1_V4_TRAIN_PART_{shard}.json and appends to E1_V4_TRAIN_PROGRESS.jsonl.
Use --merge after all shards finish to combine parts with the existing LR
results into E1_V4_TRAIN_RESULTS.json (run once, offline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.io import read_json, read_jsonl, write_json

MODES = ["q_only", "y_only", "q_y"]


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "experiments" / "e1_final_triad_v4.yaml").read_text(encoding="utf-8"))


def run_shard(shard: int, nshards: int, threads: int) -> None:
    import torch
    torch.set_num_threads(threads)
    from frauddistill.e1_final_v4.detectors import run_neural_seed
    cfg = load_cfg()
    out_dir = ROOT / cfg["data"]["output_dir"]
    dev = read_jsonl(out_dir / "E1_V4_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(out_dir / "E1_V4_PANEL_CALIBRATION.jsonl")
    anchor = read_jsonl(out_dir / "E1_V4_PANEL_ANCHOR.jsonl")
    seeds = cfg["e1_v4"]["seeds"]
    jobs = [(m, s) for m in MODES for s in seeds]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard]
    # resume: skip jobs whose model checkpoint already exists on disk
    resume = []
    for m, s in mine:
        meta = out_dir / "models" / f"{m}_seed{s}" / "meta.json"
        if meta.exists():
            print(f"[shard {shard}] skip {m}_seed{s} (checkpoint exists)", flush=True)
            continue
        resume.append((m, s))
    mine = resume
    print(f"[shard {shard}/{nshards}] jobs: {mine}", flush=True)
    results: dict[str, list] = {}
    part_path = out_dir / f"E1_V4_TRAIN_PART_{shard}.json"
    t0 = time.time()
    for mode, seed in mine:
        t1 = time.time()
        r = run_neural_seed(dev, cal, anchor, mode, seed, cfg, wrong_q_map=None, out_dir=out_dir)
        results.setdefault(f"m1_{mode}", []).append(r)
        write_json(part_path, results)
        line = {"mode": mode, "seed": seed, "anchor_macro_f1": round(float(r["anchor"]["macro_f1"]), 4),
                "anchor_auroc": round(float(r["anchor"]["auroc"]), 4), "anchor_auprc": round(float(r["anchor"]["auprc"]), 4),
                "threshold": float(r["threshold"]), "train_s": round(float(r["fit"]["elapsed_s"]), 1),
                "wall_s": round(time.time() - t1, 1), "total_s": round(time.time() - t0, 1)}
        with open(out_dir / "E1_V4_TRAIN_PROGRESS.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(json.dumps(line, ensure_ascii=False), flush=True)
    print(f"[shard {shard}] DONE", flush=True)


def merge() -> None:
    cfg = load_cfg()
    out_dir = ROOT / cfg["data"]["output_dir"]
    merged: dict[str, Any] = {}
    train_path = out_dir / "E1_V4_TRAIN_RESULTS.json"
    if train_path.exists():
        merged = read_json(train_path)
    for part in sorted(out_dir.glob("E1_V4_TRAIN_PART_*.json")):
        data = read_json(part)
        for k, v in data.items():
            merged.setdefault(k, [])
            merged[k].extend(v)
    # sanity: every mode x seed present
    seeds = cfg["e1_v4"]["seeds"]
    missing = []
    for m in MODES:
        key = f"m1_{m}"
        have = {r["seed"] for r in merged.get(key, [])}
        for s in seeds:
            if s not in have:
                missing.append(f"{key}_seed{s}")
    write_json(train_path, merged)
    print("merged keys:", sorted(merged))
    print("missing:", missing if missing else "NONE")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_shard(args.shard, args.nshards, args.threads)
