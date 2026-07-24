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
    files = ["train.json", "validation.json", "test.json", "refusals_train.json", "refusals_validation.json"]
    last = None
    for filename in files:
        last = download_hf_file("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", filename, "data/raw/aegis")
    return last or Path("data/raw/aegis")


def download_do_not_answer() -> Path:
    download_hf_file("LibrAI/do-not-answer", "data/train-00000-of-00001-6ba0076b818accff.parquet", "data/raw/do_not_answer")
    return download_hf_file("LibrAI/do-not-answer", "data_en.csv", "data/raw/do_not_answer")


def download_wildguard() -> Path:
    files = ["train/wildguard_train.parquet", "test/wildguard_test.parquet"]
    last = None
    for filename in files:
        last = download_hf_file("allenai/wildguardmix", filename, "data/raw/wildguard")
    return last or Path("data/raw/wildguard")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["halubench", "felm", "or_bench", "aegis", "do_not_answer", "wildguard"], required=True)
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
        print(download_do_not_answer())
    elif args.dataset == "wildguard":
        print(download_wildguard())


if __name__ == "__main__":
    main()
