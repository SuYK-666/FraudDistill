# -*- coding: utf-8 -*-
"""Watchdog for a neural-student training run.

Restarts with --resume (if a resume.pt exists) when the process dies.
Pass the train command via --cmd (list of args after python scripts/train...).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", nargs="+", required=True, help="train_exp3_students.py args")
    ap.add_argument("--resume-dir", default=None, help="run dir that may contain resume.pt")
    ap.add_argument("--log", default="outputs/train_watchdog.log")
    ap.add_argument("--out", default="outputs/train_watchdog.out.log")
    ap.add_argument("--err", default="outputs/train_watchdog.err.log")
    args = ap.parse_args()

    out_f = (REPO / args.out).open("a", encoding="utf-8")
    err_f = (REPO / args.err).open("a", encoding="utf-8")
    log_f = (REPO / args.log).open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    attempt = 0
    while True:
        attempt += 1
        cmd = [sys.executable, "scripts/train_exp3_students.py"] + args.cmd
        if attempt > 1 and args.resume_dir:
            resume = Path(args.resume_dir) / "resume.pt"
            if resume.exists():
                cmd += ["--resume", str(resume)]
                log(f"attempt {attempt}: resuming from {resume}")
        log(f"attempt {attempt}: launching {' '.join(cmd)}")
        p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, cwd=REPO)
        rc = p.wait()
        log(f"attempt {attempt}: exited rc={rc}")
        if rc == 0:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
