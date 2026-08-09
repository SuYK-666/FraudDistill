# -*- coding: utf-8 -*-
"""Start a persistent llama-server for one model (background helper)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODEL_GGUF = {
    "smollm2_1_7b": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "phi3_5_mini": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    "qwen2_5_7b": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "llama3_1_8b": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_GGUF))
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    exe = REPO / "third_party" / "llama_cpp" / "llama-server.exe"
    gguf = REPO / "data" / "gguf" / MODEL_GGUF[args.model]
    cmd = [str(exe), "-m", str(gguf), "--port", str(args.port), "-t", str(args.threads),
           "-c", "4096", "--no-webui", "--log-disable"]
    print(f"[server:{args.model}] starting on port {args.port}", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[server:{args.model}] pid={proc.pid}", flush=True)
    # keep alive; if killed, exit
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
    print(f"[server:{args.model}] exited rc={proc.returncode}", flush=True)


if __name__ == "__main__":
    main()
