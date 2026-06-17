from __future__ import annotations

import argparse
from pathlib import Path


def download_hf_file(repo_id: str, filename: str, target_dir: str) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir = Path(target_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=local_dir,
    )
    return Path(path)


def download_halubench() -> Path:
    return download_hf_file("PatronusAI/HaluBench", "data/test-00000-of-00001.parquet", "data/raw/halubench")


def download_felm() -> Path:
    return download_hf_file("hkust-nlp/felm", "all.jsonl", "data/raw/felm")


def download_or_bench() -> Path:
    return download_hf_file("bench-llm/or-bench", "data/test-00000-of-00001.parquet", "data/raw/or_bench")


def download_aegis() -> Path:
    return download_hf_file("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "data/train-00000-of-00001.parquet", "data/raw/aegis")


def download_do_not_answer_repo() -> Path:
    import subprocess

    target = Path("data/raw/do_not_answer/repo")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/libr-ai/do-not-answer", str(target)],
        check=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["halubench", "felm", "or_bench", "aegis", "do_not_answer"], required=True)
    args = parser.parse_args()
    if args.dataset == "halubench":
        print(download_halubench())
    elif args.dataset == "felm":
        print(download_felm())
    elif args.dataset == "or_bench":
        print(download_or_bench())
    elif args.dataset == "aegis":
        print(download_aegis())
    elif args.dataset == "do_not_answer":
        print(download_do_not_answer_repo())


if __name__ == "__main__":
    main()
