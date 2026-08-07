# -*- coding: utf-8 -*-
"""Watchdog for the final 1.5B student training run.

Stage A: wait until training completes (final_distill_metrics.json exists and
         no train_exp3_students.py process is alive). If the process dies
         without finishing, relaunch with --resume (same trajectory, max 3).
Stage B: run the official evaluation pipeline once:
         1) dev checkpoint selection (fast subset -> top-k full dev)
         2) threshold calibration (frozen to calibration.json)
         3) fresh-process reload checksum (128 samples, <=1e-5)
         4) single test run with frozen calibration
Logs go to outputs/train_final_student_watchdog.log; status JSON is written
to outputs/train_final_student_watchdog_status.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "neural_student" / "final_distilled_student"
LOG_PATH = REPO / "outputs" / "train_final_student_watchdog.log"
STATUS_PATH = REPO / "outputs" / "train_final_student_watchdog_status.json"
MARKER = RUN_DIR / "final_distill_metrics.json"
RESUME_PT = RUN_DIR / "resume.pt"

ENV = dict(os.environ)
ENV.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(payload: dict) -> None:
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def training_process_alive() -> bool:
    import subprocess as sp
    out = sp.run(["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  "Where-Object { $_.CommandLine -match 'train_exp3_students.py' -and $_.CommandLine -match 'final_distill' } | "
                  "Select-Object -ExpandProperty ProcessId"],
                 capture_output=True, text=True, timeout=60)
    return bool(out.stdout.strip())


def wait_for_training(attempt: int) -> bool:
    log(f"stage A (attempt {attempt}): waiting for training to finish")
    last_heartbeat = time.time()
    relaunched = 0
    while True:
        if MARKER.exists():
            log("training finished: final_distill_metrics.json present")
            return True
        if time.time() - last_heartbeat >= 900:
            log("heartbeat: still waiting for training (marker not present)")
            last_heartbeat = time.time()
        if not training_process_alive():
            log("training process is not alive and marker absent; checking crash state")
            if RESUME_PT.exists():
                relaunched += 1
                if relaunched > 3:
                    log("too many auto-resumes; giving up")
                    return False
                log(f"relaunching training with --resume (auto-resume {relaunched})")
                args = [sys.executable, "scripts/train_exp3_students.py",
                        "--backend", "neural", "--setting", "final_distill", "--seeds", "11",
                        "--manifest", "data/prepared/exp3_neural_student/final_train_manifest.jsonl",
                        "--max-length", "512", "--micro-batch", "4", "--effective-batch", "32",
                        "--eval-steps", "40", "--patience", "4", "--lora-r", "32", "--lora-alpha", "64",
                        "--epochs", "2", "--eval-subset", "300",
                        "--out-root", "experiments/exp3_agent_distillation_ablation/outputs/neural_student",
                        "--resume", str(RESUME_PT)]
                with (REPO / "outputs" / "train_final_distill.log").open("a", encoding="utf-8") as fo, \
                     (REPO / "outputs" / "train_final_distill.err").open("a", encoding="utf-8") as fe:
                    subprocess.Popen(args, cwd=str(REPO), env=ENV, stdout=fo, stderr=fe)
                time.sleep(120)
            else:
                log("no resume.pt and no marker; training died before first checkpoint")
                return False
        time.sleep(60)


def run_step(name: str, args: list) -> bool:
    log(f"stage B: {name} -> " + " ".join(args))
    t0 = time.time()
    proc = subprocess.run(args, cwd=str(REPO), env=ENV, capture_output=True, text=True, timeout=24 * 3600)
    dt = time.time() - t0
    tail = (proc.stdout or "").strip().splitlines()[-5:]
    log(f"stage B: {name} exit={proc.returncode} wall={dt/60:.1f}min")
    for line in tail:
        log(f"  out: {line}")
    if proc.returncode != 0:
        err_tail = (proc.stderr or "").strip().splitlines()[-8:]
        for line in err_tail:
            log(f"  err: {line}")
        return False
    return True


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("watchdog started (pid=%d)" % os.getpid())
    ok = wait_for_training(1)
    if not ok:
        status({"stage": "A", "ok": False, "note": "training did not complete", "ts": time.time()})
        log("FINAL STATUS: FAILED (training)")
        sys.exit(1)

    best_ck = None
    if not run_step("select-best-on-dev", [sys.executable, "scripts/evaluate_final_student.py", "--select-best-on-dev"]):
        status({"stage": "B1", "ok": False, "ts": time.time()})
        log("FINAL STATUS: FAILED (selection)")
        sys.exit(1)
    best_json = json.loads((RUN_DIR / "best_checkpoint.json").read_text(encoding="utf-8"))
    best_ck = best_json["checkpoint"]
    log(f"best checkpoint: {best_ck}")

    if not run_step("reload-check", [sys.executable, "scripts/evaluate_final_student.py",
                                     "--checkpoint", best_ck, "--reload-check", "--checksum-samples", "128"]):
        status({"stage": "B2", "ok": False, "ts": time.time()})
        log("FINAL STATUS: FAILED (reload checksum)")
        sys.exit(1)

    if not run_step("official-test", [sys.executable, "scripts/evaluate_final_student.py",
                                      "--checkpoint", best_ck, "--split", "test", "--frozen-calibration"]):
        status({"stage": "B3", "ok": False, "ts": time.time()})
        log("FINAL STATUS: FAILED (test)")
        sys.exit(1)

    test_json = json.loads((RUN_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    status({"stage": "done", "ok": True, "best_checkpoint": best_ck,
            "test": {k: test_json[k] for k in ("n", "macro_f1", "acc", "recall", "fpr", "auprc", "mcc") if k in test_json},
            "ts": time.time()})
    log("FINAL STATUS: ALL DONE")


if __name__ == "__main__":
    main()
