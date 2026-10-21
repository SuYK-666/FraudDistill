"""One-command orchestration for preparation, GPU runs, audit and reporting.

Without --execute every model/API stage is a dry run. This is intentional: the
local preparation machine must never download weights or spend API budget.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .a0 import DEFAULT_OUT as A0_DIR, run_a0
from .ablation import evaluate_ablation
from .audit import audit_execute
from .common import REPO, write_json
from .prepare import prepare
from .report import build_report
from .scale_eval import run_scale_eval
from .scale_train import _cfg, _select_models, run_training

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
ORCHESTRATION_LOG = SUPP_DIR / "results" / "orchestration.json"


def _models(names: str) -> list[dict]:
    return _select_models(_cfg(), names)

def _eval_models(names: str) -> list[dict]:
    models = _cfg()["models"]
    if names == "all":
        return models
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_id = {model["id"]: model for model in models}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise ValueError(f"unknown model ids: {missing}")
    return [by_id[name] for name in wanted]


def run_all(stage: str = "prepare", execute: bool = False, models: str = "all",
            device: str = "cuda", dtype: str = "bf16", gpu_hour_cny: float | None = None,
            audit_workers: int | None = None, audit_max_cny: float | None = None,
            bootstrap_replicates: int = 10000) -> dict:
    started = time.time()
    log = {
        "started_at_unix": started,
        "stage": stage,
        "execute": execute,
        "models": models,
        "stages": [],
    }

    def record(name: str, **payload) -> None:
        log["stages"].append({"stage": name, **payload})
        ORCHESTRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_json(ORCHESTRATION_LOG, log)

    if stage in {"all", "prepare"}:
        lock = prepare("configs/supplementary/lighteval_scale.yaml",
                       "configs/supplementary/lighteval_ablation.yaml")
        record("prepare", status="done", counts=lock["prepared"]["counts"])
    if stage in {"all", "preflight"}:
        plan = run_training(models=models, execute=False, device=device, dtype=dtype)
        record("preflight", status="done", cuda=plan["environment"].get("cuda_available"),
               gpu_count=plan["environment"].get("gpu_count"))
    if stage in {"all", "train"}:
        plan = run_training(models=models, execute=execute, device=device, dtype=dtype)
        record("train", status="done" if execute else "dry_run",
               queued=[item["id"] for item in plan["models"]])
    if stage in {"all", "eval"}:
        eval_models = _eval_models(models)
        results = []
        for model in eval_models:
            result = run_scale_eval(
                model["id"], checkpoint=model.get("model_path"), device=device, dtype=dtype,
                dry_run=not execute, skip_selection=bool(model.get("reuse")),
                gpu_hour_cny=gpu_hour_cny,
            )
            results.append({"model_id": model["id"], "status": result.get("status", "dry_run")})
        record("eval", status="done" if execute else "dry_run", models=results)
    if stage in {"all", "a0"}:
        result = run_a0(device=device, dtype=dtype, dry_run=not execute)
        record("a0", status=result.get("status", "done"))
    if stage in {"all", "audit"}:
        result = audit_execute(execute=execute, cache_only=not execute,
                               workers=audit_workers, max_cny=audit_max_cny)
        record("audit", status=result.get("status"), todo=result.get("todo"), cached=result.get("cached"))
    if stage in {"all", "ablation", "report"}:
        a0_path = A0_DIR / "predictions.jsonl"
        ablation = evaluate_ablation(a0_path if a0_path.exists() else None,
                                     bootstrap_replicates=bootstrap_replicates,
                                     allow_missing_audit=True)
        record("ablation", status="done",
               ready=[key for key, value in ablation["policies"].items()
                      if value.get("status") == "ready"])
    if stage in {"all", "report"}:
        report = build_report()
        record("report", status="done", outputs=report["outputs"])

    log["elapsed_s"] = round(time.time() - started, 2)
    write_json(ORCHESTRATION_LOG, log)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="prepare",
                        choices=["all", "prepare", "preflight", "train", "eval", "a0",
                                 "audit", "ablation", "report"])
    parser.add_argument("--execute", action="store_true",
                        help="allow weight loading/download and reviewer API calls")
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gpu-hour-cny", type=float, default=None)
    parser.add_argument("--audit-workers", type=int, default=None)
    parser.add_argument("--audit-max-cny", type=float, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    run_all(args.stage, args.execute, args.models, args.device, args.dtype,
            args.gpu_hour_cny, args.audit_workers, args.audit_max_cny,
            args.bootstrap_replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
