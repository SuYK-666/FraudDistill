import os, time, sys
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download
log = open(r"data\raw\_download_log.txt", "a", encoding="utf-8")
def dl(repo, ftype, allow_patterns):
    t0 = time.time()
    log.write(f"[{time.strftime('%H:%M:%S')}] start {repo}\n"); log.flush()
    p = snapshot_download(repo_id=repo, repo_type=ftype, allow_patterns=allow_patterns)
    log.write(f"[{time.strftime('%H:%M:%S')}] done {repo} -> {p} in {time.time()-t0:.0f}s\n"); log.flush()
    return p
dl("nvidia/llama-3.1-nemoguard-8b-content-safety", "model", ["*"])
dl("LibrAI/longformer-harmful-ro", "model", ["*"])
log.write("ALL DOWNLOADS DONE\n"); log.flush()
