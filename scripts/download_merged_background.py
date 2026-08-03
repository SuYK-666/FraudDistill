import os, time
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download
log = open(r"data\raw\_download_log.txt", "a", encoding="utf-8")
t0 = time.time()
log.write("[" + time.strftime("%H:%M:%S") + "] start oneonlee merged\n")
log.flush()
p = snapshot_download("oneonlee/llama-3.1-nemoguard-8b-content-safety-merged", repo_type="model")
log.write("[" + time.strftime("%H:%M:%S") + "] done merged -> " + p + " in " + str(int(time.time()-t0)) + "s\n")
log.flush()
