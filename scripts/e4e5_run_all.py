# -*- coding: utf-8 -*-
"""E4/E5 final orchestration: panel -> inference -> calibration -> reports.

Usage:
  python scripts/e4e5_run_all.py [--g2] [--skip-base] [--api] [--no-panel]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], tag: str) -> None:
    print(f"\n===== [{tag}] {' '.join(cmd)} =====", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"[{tag}] FAILED rc={proc.returncode} after {time.time()-t0:.0f}s")
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", action="store_true")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--protocol-dir", default=None)
    ap.add_argument("--no-panel", action="store_true")
    args = ap.parse_args()

    if args.protocol_dir:
        proto = REPO / args.protocol_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proto = REPO / "outputs/exp4_unseen_student_v2" / f"e4v2_{ts}"
    proto.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] protocol dir: {proto}")

    if not args.no_panel:
        cmd = [PY, "scripts/e4e5_build_panel.py", "--out", str(proto)]
        if args.g2:
            cmd.append("--g2")
        if args.api:
            cmd.append("--g2-api")
        run(cmd, "panel")

    run([PY, "scripts/e4e5_run_inference.py", "--protocol-dir", str(proto)]
        + (["--skip-base"] if args.skip_base else []), "inference")
    run([PY, "scripts/e4e5_run_calibration.py", "--protocol-dir", str(proto)]
        + (["--api"] if args.api else []), "calibration")
    run([PY, "scripts/e4e5_write_reports.py", "--protocol-dir", str(proto)], "reports")
    print(f"\n[orchestrator] ALL DONE -> {proto}")


if __name__ == "__main__":
    main()
