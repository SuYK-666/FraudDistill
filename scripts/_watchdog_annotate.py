# -*- coding: utf-8 -*-
"""Watchdog launcher for annotate_exp3_expansion.py.

Restarts the annotator if the annotated-row count stalls for >5 minutes.
Safe: the annotator skips rows already written (id-based) and the API cache
dedupes identical requests, so restarts never double-bill.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_FILE = REPO / "data/prepared/exp3_neural_student/expansion_annotated.jsonl"
LOG = (REPO / "outputs/annotate_watchdog.log").open("a", encoding="utf-8")
ERR = (REPO / "outputs/annotate_watchdog.err").open("a", encoding="utf-8")


def count_rows() -> int:
    if not OUT_FILE.exists():
        return 0
    n = 0
    with OUT_FILE.open(encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    cmd = [sys.executable, "scripts/annotate_exp3_expansion.py", "--concurrency", "120",
           "--token-caps", "cheap", "--timeout", "300"]
    stall_minutes = 5
    while True:
        n0 = count_rows()
        LOG.write(f"[watchdog] launching annotator at {time.strftime('%H:%M:%S')} rows={n0}\n")
        LOG.flush()
        p = subprocess.Popen(cmd, stdout=LOG, stderr=ERR, cwd=REPO)
        last = n0
        last_change = time.time()
        while p.poll() is None:
            time.sleep(45)
            n = count_rows()
            if n > last:
                last = n
                last_change = time.time()
                LOG.write(f"[watchdog] progress rows={n}\n")
                LOG.flush()
            elif time.time() - last_change > stall_minutes * 60:
                LOG.write(f"[watchdog] stall >{stall_minutes} min at rows={n}; restarting\n")
                LOG.flush()
                p.kill()
                time.sleep(3)
                break
        if p.poll() is not None:
            rc = p.returncode
            LOG.write(f"[watchdog] annotator exited rc={rc} rows={count_rows()}\n")
            LOG.flush()
            if rc == 0:
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
