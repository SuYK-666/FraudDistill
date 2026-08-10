# -*- coding: utf-8 -*-
"""E4/E5 realtime inference progress monitor.

Reads the inference log + process CPU to estimate stage progress & ETA, and
writes PROGRESS.md for live viewing. Calibrates per-row wall time whenever a
stage completes so later-stage ETAs get more accurate.

Usage:
  python scripts/e4e5_monitor.py            # print once
  python scripts/e4e5_monitor.py --loop     # keep updating every 60s
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "data/prepared/e4e5_v2/logs/inference_FINAL.out.log"
PROTO = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
PROGRESS = REPO / "experiments/exp4_unseen/PROGRESS.md"

STAGES = [  # (log marker, label, n_rows, pred file name, default row_s)
    ("running final_student (Final Student)", "Final Student test", 1200, "final_student.jsonl", 3.0),
    ("running neural_gold", "Neural-Gold test", 1200, "neural_gold.jsonl", 3.0),
    ("running neural_softdistill", "Neural-SoftDistill test", 1200, "neural_softdistill.jsonl", 3.0),
    ("final_student on calibration", "Final Student calibration", 600, "final_student_calibration.jsonl", 3.0),
    ("base zero-shot", "Base-1.5B zero-shot", 300, "base_zeroshot.jsonl", 8.0),
]
MODEL_LOAD_S = 75.0  # python boot + data + model load before row loop starts


def ps(cmd: str) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def find_inference_pid() -> int | None:
    for row in ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  "ForEach-Object { \"$($_.ProcessId):$($_.CommandLine)\" }").splitlines():
        if "e4e5_run_inference.py" in row:
            try:
                return int(row.split(":")[0])
            except Exception:
                return None
    return None


def proc_start(pid: int) -> float | None:
    raw = ps(f"([datetime](Get-Process -Id {pid}).StartTime).ToUniversalTime().ToString('o')")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def cpu_seconds(pid: int) -> float:
    try:
        return float(ps(f"(Get-Process -Id {pid}).CPU") or 0)
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    state = {"last_cpu": None, "last_t": None, "rate_per_s": None,
             "stage_start": None, "last_stage": -2, "row_s_calib": {}}
    while True:
        lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines() if LOG.exists() else []

        stage_idx = -1
        # last log line wins: the most recent stage marker is the current stage
        for ln in reversed(lines):
            for i, (marker, *_rest) in enumerate(STAGES):
                if marker in ln:
                    stage_idx = i
                    break
            if stage_idx >= 0:
                break
        if stage_idx < 0:
            stage_label, stage_n, pred_file, row_s = "starting / model load", 1200, "", 13.0
        else:
            _m, stage_label, stage_n, pred_file, row_s = STAGES[stage_idx]
            row_s = state["row_s_calib"].get(stage_idx, row_s)

        pid = find_inference_pid()
        cpu = cpu_seconds(pid) if pid else 0.0
        now = time.time()
        rate = None
        if state["last_cpu"] is not None and cpu >= state["last_cpu"]:
            dt = now - state["last_t"]
            if dt > 0:
                rate = (cpu - state["last_cpu"]) / dt
        state["last_cpu"], state["last_t"] = cpu, now
        if rate:
            state["rate_per_s"] = rate

        # stage anchor: first stage anchored to process start (+model load),
        # later stages anchored when the monitor first observes the marker.
        if stage_idx != state["last_stage"]:
            if stage_idx >= 0 and state["last_stage"] >= 0:
                # a stage finished: calibrate its real per-row wall time
                dur = now - state["stage_start"]
                if dur > 0 and state["last_stage"] < len(STAGES):
                    state["row_s_calib"][state["last_stage"]] = dur / STAGES[state["last_stage"]][2]
            # anchor stage start: prior completed pred-file mtime + model load,
            # so estimates survive monitor restarts (execution order != STAGES order)
            anchor = {1: "final_student_calibration.jsonl", 2: "neural_gold.jsonl",
                      3: "final_student.jsonl", 4: "neural_softdistill.jsonl"}.get(stage_idx)
            prev_file = (PROTO / "predictions" / anchor) if anchor else None
            if prev_file is not None and prev_file.exists():
                state["stage_start"] = prev_file.stat().st_mtime + MODEL_LOAD_S
            elif stage_idx == 0:
                pst = proc_start(pid) if pid else None
                state["stage_start"] = (pst + MODEL_LOAD_S) if pst else now
            else:
                state["stage_start"] = now - MODEL_LOAD_S  # marker precedes model load
            state["last_stage"] = stage_idx

        stage_elapsed = max(0.0, now - state["stage_start"]) if stage_idx >= 0 else 0.0
        rows_done = min(stage_n, int(stage_elapsed / row_s)) if stage_idx >= 0 else 0
        # if prediction file already complete, show true numbers
        if pred_file:
            pf = PROTO / "predictions" / pred_file
            if pf.exists():
                try:
                    n_lines = sum(1 for _ in open(pf, encoding="utf-8"))
                    rows_done = max(rows_done, n_lines)
                except Exception:
                    pass
        rows_left = max(0, stage_n - rows_done)
        stage_eta = rows_left * row_s
        total_eta = stage_eta + sum(
            st[2] * state["row_s_calib"].get(i, st[4])
            for i, st in enumerate(STAGES)
            if i != stage_idx and not (PROTO / "predictions" / st[3]).exists())

        cores = state["rate_per_s"] or 9.5
        wall = now - (state["stage_start"] or now)
        txt = [
            f"# E4/E5 Inference Progress (updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            "",
            f"- PID: {pid}  |  CPU total: {cpu/3600:.1f} h  |  cores in use: {cores:.1f}",
            f"- Stage: **{stage_label}**  |  est. rows: {rows_done}/{stage_n} ({100.0*rows_done/max(stage_n,1):.1f}%)  |  stage elapsed: {wall/60:.0f} min",
            f"- Stage ETA: ~{stage_eta/3600:.1f} h  |  Overall ETA: ~{total_eta/3600:.1f} h from now",
            "",
            "| Stage | Rows | Status | ETA |",
            "|---|---|---|---|",
        ]
        for i, (marker, label, n, pfile, drow) in enumerate(STAGES):
            pf = PROTO / "predictions" / pfile
            done = 0
            if pf.exists():
                try:
                    done = sum(1 for _ in open(pf, encoding="utf-8"))
                except Exception:
                    done = 0
            has_marker = any(marker in ln for ln in lines)
            if done >= n:
                txt.append(f"| {label} | {n} | done ({done}) | - |")
            elif i == stage_idx and stage_idx >= 0:
                txt.append(f"| {label} | {n} | running ~{rows_done}/{n} | ~{stage_eta/3600:.1f}h |")
            elif has_marker:
                txt.append(f"| {label} | {n} | starting | - |")
            else:
                txt.append(f"| {label} | {n} | pending | - |")
        if stage_idx >= 0 and rows_done >= stage_n and not (PROTO / "predictions" / pred_file).exists():
            txt.append("")
            txt.append(f"> Note: current stage rows estimated complete; waiting for `{pred_file}` to be written...")
        txt.append("")
        txt.append("_Estimates use wall-clock per-row times; calibrated automatically after each stage completes._")
        PROGRESS.write_text("\n".join(txt), encoding="utf-8")

        if not args.loop:
            print("\n".join(txt))
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

