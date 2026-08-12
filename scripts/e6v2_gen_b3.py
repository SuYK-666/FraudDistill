# -*- coding: utf-8 -*-
"""E6 v2 B3 generation driver: per-model targeted variant generation.
Usage: python scripts/e6v2_gen_b3.py --slots M1,M2,M3,M4,M6
Each slot generates only its own b3 task rows (from data/b3_tasks_<slot>.jsonl)
into generations/per_model/<slot>.jsonl with the same resumable cache semantics."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6v2_common import DATA_DIR, PROTOCOL_DIR, read_jsonl, read_json
import e6v2_generate as gen

ALL_SLOTS = ("M1", "M2", "M3", "M4", "M5", "M6")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="M1,M2,M3,M4,M6")
    args = ap.parse_args()
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    registry = read_json(PROTOCOL_DIR / "model_registry_frozen.json")
    for slot in slots:
        tasks = read_jsonl(DATA_DIR / f"b3_tasks_{slot}.jsonl")
        if not tasks:
            print(f"[b3] {slot}: no tasks", flush=True)
            continue
        skip = set(s for s in ALL_SLOTS if s != slot)
        print(f"[b3] {slot}: tasks={len(tasks)}", flush=True)
        gen.run_panel(registry, tasks, "b3", skip_slots=skip)
    print("[b3] DONE", flush=True)

if __name__ == "__main__":
    main()
