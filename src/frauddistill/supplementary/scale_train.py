"""Preflight and resumable training orchestration for the five Qwen scales."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import REPO, read_json, read_jsonl, resolve_repo, sha256_file, write_json

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
RUN_ROOT = SUPP_DIR / "runs"
CONFIG = REPO / "configs/supplementary/lighteval_scale.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _environment() -> dict:
    env = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "disk_free_gb": round(shutil.disk_usage(REPO).free / 1024 ** 3, 2),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        env["gpus"] = []
        for i in range(env["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            env["gpus"].append({
                "index": i, "name": props.name,
                "total_vram_gb": round(total / 1024 ** 3, 2),
                "free_vram_gb": round(free / 1024 ** 3, 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
    except Exception as exc:
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    for package in ("transformers", "peft", "accelerate", "datasets"):
        try:
            from importlib.metadata import version
            env[package] = version(package)
        except Exception:
            env[package] = None
    return env


def _model_cmd(model: dict, scale: dict, args) -> tuple[list[str], Path, Path]:
    train = scale["training"]
    micro_batch = int(model.get("micro_batch", train["micro_batch_default"]))
    out_root = RUN_ROOT / model["id"]
    run_dir = out_root / "final_distilled_student"
    resume = run_dir / "resume.pt"
    cmd = [
        sys.executable,
        str(REPO / "scripts/train_exp3_students.py"),
        "--backend", "neural",
        "--setting", "final_distill",
        "--architecture", "standard",
        "--model-name", model["hf_id"],
        *(["--model-revision", model["hf_revision"]] if model.get("hf_revision") else []),
        "--manifest", str(resolve_repo(scale["paths"]["train_manifest"])),
        "--seeds", str(scale["seed"]),
        "--epochs", str(train["epochs"]),
        "--max-length", str(train["max_length"]),
        "--micro-batch", str(micro_batch),
        "--eval-batch", str(int(model.get("eval_batch", 16))),
        "--effective-batch", str(train["effective_batch"]),
        "--lr-lora", str(model.get("lr_lora", train["lr_lora"])),
        "--lr-head", str(model.get("lr_head", train["lr_head"])),
        "--warmup-ratio", str(model.get("warmup_ratio", train["warmup_ratio"])),
        "--lora-r", str(train["lora_r"]),
        "--lora-alpha", str(train["lora_alpha"]),
        "--eval-steps", str(model.get("eval_steps", train["eval_steps"])),
        "--patience", str(train["patience"]),
        "--eval-subset", "300",
        "--out-root", str(out_root),
        "--run-name", "final_distilled_student",
        "--device", args.device,
        "--dtype", args.dtype if args.dtype != "auto" else "auto",
    ]
    if train.get("gradient_checkpointing", False):
        cmd.append("--gradient-checkpointing")
    if model.get("master_weights_fp32", False):
        cmd.append("--master-weights-fp32")
    if resume.exists():
        cmd.extend(["--resume", str(resume)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    return cmd, out_root, run_dir


def _select_models(scale: dict, names: str) -> list[dict]:
    wanted = [part.strip() for part in names.split(",") if part.strip()]
    models = [m for m in scale["models"] if bool(m.get("train"))]
    if "all" in wanted:
        return models
    by_id = {m["id"]: m for m in scale["models"]}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise ValueError(f"unknown model ids: {missing}")
    selected = [by_id[name] for name in wanted]
    if any(not m.get("train") for m in selected):
        raise ValueError("reuse-only models cannot be sent to the train command")
    return selected


def run_training(models: str = "all", execute: bool = False, device: str = "cuda",
                 dtype: str = "bf16", max_steps: int | None = None,
                 skip_existing: bool = True) -> dict:
    scale = _cfg()
    env = _environment()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_ROOT / "environment.json", env)
    chosen = _select_models(scale, models)
    manifest_path = resolve_repo(scale["paths"]["train_manifest"])
    rows = read_jsonl(manifest_path)
    if len(rows) != int(scale["expected_counts"]["train"]):
        raise ValueError(f"training manifest count {len(rows)} != {scale['expected_counts']['train']}")
    lock_path = SUPP_DIR / "prepared" / "asset_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("run scripts/supp_prepare.py before training")
    lock = read_json(lock_path)
    if lock["configs"]["scale"]["sha256"] != sha256_file(CONFIG):
        raise ValueError("scale config changed after preparation; rerun scripts/supp_prepare.py")
    frozen_manifest = lock["prepared"]["sources"]["train_manifest"]
    if frozen_manifest["sha256"] != sha256_file(manifest_path):
        raise ValueError("training manifest no longer matches the frozen asset lock")
    plan = {
        "protocol": scale["protocol"],
        "mode": "execute" if execute else "dry_run",
        "manifest": {"path": str(manifest_path), "rows": len(rows), "sha256": sha256_file(manifest_path)},
        "device": device,
        "dtype": dtype,
        "environment": env,
        "models": [],
    }
    for model in chosen:
        cmd, out_root, run_dir = _model_cmd(model, scale, argparse.Namespace(
            device=device, dtype=dtype, max_steps=max_steps,
        ))
        item = {
            "id": model["id"], "hf_id": model["hf_id"], "params_b": model["params_b"],
            "out_root": str(out_root), "run_dir": str(run_dir),
            "command": cmd, "skipped_existing": bool(skip_existing and (run_dir / "final_distill_metrics.json").exists()),
        }
        plan["models"].append(item)
    write_json(RUN_ROOT / "training_plan.json", plan)

    if execute and device == "cuda" and not env.get("cuda_available"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if execute and device == "cuda":
        gpu_free = max([g.get("free_vram_gb", 0) for g in env.get("gpus", [])] or [0])
        for item, model in zip(plan["models"], chosen):
            if gpu_free < float(model["min_free_vram_gb"]):
                print(f"[WARN] {model['id']} wants >= {model['min_free_vram_gb']} GB free; "
                      f"current max free VRAM is {gpu_free:.2f} GB", flush=True)

    if not execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    for item in plan["models"]:
        if item["skipped_existing"]:
            print(f"[skip] {item['id']} already has final_distill_metrics.json", flush=True)
            continue
        out_root = Path(item["out_root"])
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / "train.log"
        err_path = out_root / "train.err.log"
        print(f"[train] {item['id']} -> {out_root}", flush=True)
        start = time.time()
        with log_path.open("a", encoding="utf-8") as stdout, err_path.open("a", encoding="utf-8") as stderr:
            proc = subprocess.run(item["command"], cwd=str(REPO), stdout=stdout, stderr=stderr, text=True)
        item["exit_code"] = proc.returncode
        item["elapsed_s"] = round(time.time() - start, 1)
        write_json(RUN_ROOT / "training_plan.json", plan)
        if proc.returncode != 0:
            raise RuntimeError(f"training failed for {item['id']} (exit {proc.returncode}); see {err_path}")
    plan["status"] = "DONE"
    write_json(RUN_ROOT / "training_plan.json", plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--execute", action="store_true", help="run training; default is a no-download dry run")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)
    result = run_training(args.models, args.execute, args.device, args.dtype,
                          args.max_steps, not args.no_skip_existing)
    if not args.execute:
        print("\nDry run only. Add --execute on the GPU host to download weights and train.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
