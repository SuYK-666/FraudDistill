# -*- coding: utf-8 -*-
"""Background downloader for DeepSeek-R1-Distill-Qwen-1.5B (guide 9.1).
Uses proxy 127.0.0.1:7897 for huggingface.co (user-approved). Logs to outputs/download_r1qwen.log."""
import os, sys, traceback
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("ALL_PROXY", "http://127.0.0.1:7897")
from huggingface_hub import snapshot_download
LOG = os.path.join("outputs", "download_r1qwen.log")
def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
try:
    log("start download (proxy 7897)")
    p = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_workers=4)
    log("DONE " + p)
except Exception:
    log("FAIL\n" + traceback.format_exc())
    sys.exit(1)
