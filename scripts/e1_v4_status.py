# -*- coding: utf-8 -*-
"""E1 v4 live status: votes, M1 jobs, logs, RAM/disk, cost."""
import json, collections, glob, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prepared" / "e1_final_triad_v4"

print("=== E1 v4 live status ===")
# votes
vp = OUT / "E1_V4_ANCHOR_VIEW_VOTES.jsonl"
if vp.exists():
    rows = [json.loads(l) for l in vp.open(encoding="utf-8")]
    st = collections.Counter(r.get("status") for r in rows)
    print(f"[anchor votes] total={len(rows)} ok={st.get('ok',0)} bad={st.get('bad_json',0)}")
else:
    print("[Anchor LLM votes] ?????")

# M1 training jobs
parts = sorted(OUT.glob("E1_V4_TRAIN_PART_*.json"))
done = collections.Counter()
for p in parts:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d.items():
            for r in v:
                done[(r.get("mode"), r.get("seed"))] = 1
    except Exception:
        pass
print(f"[m1 jobs] completed={len(done)}/15")
for mode in ["q_only", "y_only", "q_y"]:
    ms = [f"{m}_{s}" for (m, s) in done if m == mode]
    print(f"  {mode}: {len(ms)} done {sorted(ms)}")

# live training logs
print("[train logs]")
for lg in sorted(OUT.glob("logs/m1_shard*.out.log")):
    lines = [l.rstrip() for l in lg.open(encoding="utf-8", errors="ignore") if l.strip()]
    tail = lines[-3:] if lines else ["(empty)"]
    print(f"  {lg.name}: {tail}")

# RAM / disk
try:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print(f"[ram] load={m.dwMemoryLoad}% free={m.ullAvailPhys/1e9:.1f}GB")
except Exception:
    pass
d = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-PSDrive C).Free/1GB"], capture_output=True, text=True)
print(f"[disk] C free ~{d.stdout.strip()}GB")

# cost
ledger = OUT / "E1_V4_BUDGET_LEDGER.jsonl"
if ledger.exists():
    tot = 0.0; byp = collections.Counter()
    for l in ledger.open(encoding="utf-8"):
        e = json.loads(l)
        tot += e.get("cost_cny", 0) or 0
        byp[e.get("provider")] += e.get("cost_cny", 0) or 0
    print(f"[cost] total=RMB{tot:.2f} by-provider={{{', '.join(f'{k}:?{v:.2f}' for k,v in byp.items())}}}")
