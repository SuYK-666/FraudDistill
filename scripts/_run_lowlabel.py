# -*- coding: utf-8 -*-
"""Low-label curve driver: run gold10/soft10/gold25/soft25/gold50/soft50
sequentially, each with crash-retry via --resume. Direct subprocess launches
(avoids the transient Winsock issue seen with long-lived chain parents)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_exp3_neural_stages import STAGES, COMMON  # noqa: E402

NEURAL = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student"
LOW = NEURAL / "lowlabel"
LOG = (REPO / "outputs/lowlabel_driver.log").open("a", encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def done_json(stage: str, extra: list[str]) -> Path:
    setting = extra[1]
    gf_tag = ""
    if "--gold-fraction" in extra:
        gf_tag = f"_gf{extra[extra.index('--gold-fraction') + 1]}"
    return LOW / f"{setting}_standard_seed11{gf_tag}.json"


def main() -> None:
    stages = ["gold10", "soft10", "gold25", "soft25", "gold50", "soft50"]
    for name in stages:
        extra, out_root, manifest_override = STAGES[name]
        out = done_json(name, extra)
        if out.exists():
            log(f"{name}: already done, skip")
            continue
        log(f"{name}: starting")
        for attempt in range(1, 4):
            cmd = [sys.executable, "scripts/train_exp3_students.py", "--manifest",
                   manifest_override or "data/prepared/exp3_neural_student/train_manifest.jsonl"] + COMMON + extra
            cmd += ["--out-root", str(LOW)]
            if attempt > 1:
                setting = extra[1]
                gf_tag = ""
                if "--gold-fraction" in extra:
                    gf_tag = f"_gf{extra[extra.index('--gold-fraction') + 1]}"
                resume = LOW / f"{setting}_standard_seed11{gf_tag}" / "resume.pt"
                if resume.exists():
                    cmd += ["--resume", str(resume)]
                    log(f"{name}: attempt {attempt} resuming from {resume}")
            log(f"{name}: attempt {attempt}: {' '.join(cmd)}")
            with (REPO / f"outputs/lowlabel_{name}.out.log").open("a", encoding="utf-8") as fo, \
                 (REPO / f"outputs/lowlabel_{name}.err.log").open("a", encoding="utf-8") as fe:
                rc = subprocess.call(cmd, stdout=fo, stderr=fe, cwd=REPO)
            if out.exists() or rc == 0:
                log(f"{name}: done rc={rc}")
                break
            log(f"{name}: failed rc={rc}; retrying")
            time.sleep(30)
    log("low-label driver finished")


if __name__ == "__main__":
    main()
