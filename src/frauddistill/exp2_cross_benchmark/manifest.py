"""Write dataset_manifest.json and sample_ids files for exp2."""
from __future__ import annotations

import hashlib
import json
import os
import time

from frauddistill.exp2_cross_benchmark.paths import BENCHMARKS, EXPECTED_POOL, EXPERIMENT_DIR, out_dir

SOURCES = {
    "fraudr1": {
        "source_url": "https://github.com/Chouoftears/Fraud-R1 (dataset in repo, ACL 2025 Findings)",
        "commit": "local checkout in data/raw/fraudr1/repo",
        "license": "research use (see Fraud-R1 repo)",
        "files": [
            "data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-Chinese.json",
            "data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-English.json",
            "data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-Chinese.json",
            "data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-English.json",
        ],
    },
    "orbench": {
        "source_url": "https://huggingface.co/datasets/bench-llm/or-bench (revision e36d8b80e81837c8a8f264bbb2a49f1b32c7e272)",
        "license": "CC BY 4.0 (or-bench repo)",
        "files": ["data/raw/or_bench/or-bench-hard-1k.csv", "data/raw/or_bench/or-bench-80k.csv", "data/raw/or_bench/or-bench-toxic.csv"],
    },
    "do_not_answer": {
        "source_url": "https://huggingface.co/datasets/LibrAI/do-not-answer (data_en.csv)",
        "license": "CC BY-NC-SA 4.0",
        "files": ["data/raw/do_not_answer/data_en.csv"],
    },
    "aegis2": {
        "source_url": "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (test.json)",
        "revision": "ef1f9de54f760180f70b517dd10362d9463ddc58",
        "license": "CC BY 4.0 (NVIDIA Aegis)",
        "files": ["data/raw/aegis/test.json"],
    },
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {
        "experiment": "exp2_cross_benchmark_teacher",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": 20260803,
        "benchmarks": {},
    }
    for b in BENCHMARKS:
        unified = out_dir(b, "unified") / f"{b}_eval.jsonl"
        rows = [json.loads(l) for l in open(unified, encoding="utf-8")]
        ids = [r["id"] for r in rows]
        manifest["benchmarks"][b] = {
            "pool_n": len(rows),
            "expected_n": EXPECTED_POOL[b],
            "sha256_unified": sha256(unified),
            "sources": SOURCES[b],
        }
        sample_file = EXPERIMENT_DIR / f"sample_ids_{b}.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
        print(f"[{b}] {len(rows)} ids -> {sample_file.name}")
    with open(EXPERIMENT_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("manifest written")


if __name__ == "__main__":
    main()
