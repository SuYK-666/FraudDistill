"""Unified multi-scale checkpoint selection, scoring, calibration and cost accounting."""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .common import (REPO, binary_label, binary_metrics, four_class_metrics,
                     read_json, read_jsonl, resolve_repo, row_key,
                     select_target_threshold, sha256_file, write_json, write_jsonl)

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
PREP_DIR = SUPP_DIR / "prepared"
RAW_DIR = PREP_DIR / "raw"
RESULT_ROOT = SUPP_DIR / "results" / "scale"
CONFIG = REPO / "configs/supplementary/lighteval_scale.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _torch_dtype(torch, name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _model_for(model_cfg: dict, checkpoint: Path, max_length: int, dtype: str):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from frauddistill.student.model import NeuralStudentConfig, build_neural_student

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = NeuralStudentConfig(
        model_name=model_cfg["hf_id"], architecture="standard", max_length=max_length,
        lora_r=32, lora_alpha=64, use_lora=False, torch_dtype=dtype,
        model_revision=model_cfg.get("hf_revision"),
    )
    model = build_neural_student(cfg, freeze_base=True)
    if (checkpoint / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()
    return model, tokenizer


def _prepare_jobs(rows: list[dict]) -> list[dict]:
    jobs = []
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"duplicate evaluation ID: {key}")
        seen.add(key)
        jobs.append({**row, "id": key})
    return jobs


def _score_rows(model, tokenizer, rows: list[dict], max_length: int, batch_size: int,
                device: str) -> tuple[list[dict], dict]:
    import torch
    from torch.utils.data import DataLoader
    from frauddistill.student.collator import neural_collate
    from frauddistill.student.dataset import ID_TO_LABEL, build_neural_examples

    jobs = _prepare_jobs(rows)
    job_by_id = {job["id"]: job for job in jobs}
    examples = build_neural_examples(jobs, max_length=max_length,
                                     use_teacher_soft=True, use_pairwise=False)
    for example, job in zip(examples, jobs):
        # Estimate the retained prompt tokens under the same 512-token cap. The
        # classifier is not generative, so completion tokens are exactly zero.
        example["_input_tokens"] = min(len(tokenizer(example["text"], add_special_tokens=False)["input_ids"]),
                                       max_length)
        example["_key"] = row_key(job)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: neural_collate(
                            batch, tokenizer, max_length=max_length, architecture="standard"))
    preds = []
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            tensors = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                       for key, value in batch.items()}
            out = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
            probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
            for i, key in enumerate(tensors["ids"]):
                type_probs = {ID_TO_LABEL[j]: round(float(probs[i, j]), 6) for j in range(4)}
                risk = float(1.0 - probs[i, 0])
                job = job_by_id[key]
                preds.append({
                    "id": key,
                    "_key": key,
                    "slot": job.get("slot"),
                    "prompt_id": job.get("prompt_id"),
                    "risk_score": round(risk, 6),
                    "risk_type": max(type_probs, key=type_probs.get),
                    "type_probabilities": type_probs,
                })
    elapsed = time.perf_counter() - started
    index = {example["id"]: example for example in examples}
    for pred in preds:
        pred["input_tokens"] = int(index[pred["_key"]]["_input_tokens"])
        pred["output_tokens"] = 0
        del pred["_key"]
    return preds, {
        "rows": len(preds),
        "seconds": round(elapsed, 3),
        "rows_per_s": round(len(preds) / max(elapsed, 1e-9), 3),
        "input_tokens": int(sum(pred["input_tokens"] for pred in preds)),
        "output_tokens": 0,
    }


def _score_map(preds: list[dict], rows: list[dict]) -> dict[str, dict]:
    by_id = {pred["id"]: pred for pred in preds}
    missing = [row_key(row) for row in rows if row_key(row) not in by_id]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} rows")
    return by_id


def _dev_metrics(rows: list[dict], preds: dict[str, dict], threshold: float = 0.5) -> dict:
    y = [binary_label(row) for row in rows]
    scores = [float(preds[row_key(row)]["risk_score"]) for row in rows]
    pred = [1 if score >= threshold else 0 for score in scores]
    metrics = binary_metrics(y, scores, pred, threshold=threshold)
    real_idx = [i for i, row in enumerate(rows) if row.get("source") != "synthetic"]
    if real_idx:
        metrics["real_macro_f1"] = binary_metrics(
            [y[i] for i in real_idx], [scores[i] for i in real_idx], [pred[i] for i in real_idx]
        )["macro_f1"]
    return metrics


def _checkpoint_step(path: Path) -> int:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits or 0)


def _candidate_checkpoints(run_dir: Path) -> list[Path]:
    candidates = [path for path in run_dir.iterdir()
                  if path.is_dir() and (path.name.startswith("best_step") or
                                        path.name.startswith("checkpoint-") or
                                        path.name.startswith("best_epoch"))]
    unique = {}
    for path in candidates:
        step = _checkpoint_step(path)
        prior = unique.get(step)
        if prior is None or (prior.name.startswith("checkpoint-") and path.name.startswith("best_step")):
            unique[step] = path
    return [unique[step] for step in sorted(unique)]


def _select_checkpoint(model_cfg: dict, run_dir: Path, dev_rows: list[dict], max_length: int,
                       batch_size: int, device: str, dtype: str, fast_subset: int,
                       top_k: int, scale: dict) -> tuple[Path, dict]:
    candidates = _candidate_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no candidate checkpoints under {run_dir}")
    rng = random.Random(20260804)
    fast_rows = rng.sample(dev_rows, min(fast_subset, len(dev_rows))) if fast_subset else dev_rows
    fast = []
    for checkpoint in candidates:
        model, tokenizer = _model_for(model_cfg, checkpoint, max_length, dtype)
        model.to(device)
        preds, timing = _score_rows(model, tokenizer, fast_rows, max_length, batch_size, device)
        metrics = _dev_metrics(fast_rows, _score_map(preds, fast_rows))
        fast.append({"checkpoint": str(checkpoint), "step": _checkpoint_step(checkpoint),
                     "fast_metrics": metrics, "timing": timing})
        print(f"[select] {checkpoint.name}: MF1={metrics['macro_f1']:.4f} "
              f"recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}", flush=True)
        del model
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    fast.sort(key=lambda row: (row["fast_metrics"]["macro_f1"],
                               row["fast_metrics"].get("real_macro_f1", 0.0),
                               row["step"]), reverse=True)
    shortlist = fast[:top_k]
    full = []
    for item in shortlist:
        checkpoint = Path(item["checkpoint"])
        model, tokenizer = _model_for(model_cfg, checkpoint, max_length, dtype)
        model.to(device)
        preds, timing = _score_rows(model, tokenizer, dev_rows, max_length, batch_size, device)
        metrics = _dev_metrics(dev_rows, _score_map(preds, dev_rows))
        full.append({**item, "full_metrics": metrics, "full_timing": timing})
        print(f"[select-full] {checkpoint.name}: MF1={metrics['macro_f1']:.4f} "
              f"recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}", flush=True)
        del model
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    gate = scale["training"]["selection"]
    feasible = [row for row in full
                if row["full_metrics"]["fpr"] <= gate["fpr_max"]
                and row["full_metrics"]["recall"] >= gate["recall_min"]]
    pool = feasible or full
    best = max(pool, key=lambda row: (
        row["full_metrics"]["macro_f1"], row["full_metrics"].get("real_macro_f1", 0.0),
        row["full_metrics"].get("auprc") or 0.0, row["full_metrics"]["mcc"], -row["step"],
    ))
    summary = {
        "candidate_count": len(candidates),
        "fast_subset": len(fast_rows),
        "shortlist": shortlist,
        "full_evaluated": full,
        "feasible": bool(feasible),
        "gate": gate,
        "selected": best,
    }
    return Path(best["checkpoint"]), summary


def _benchmark(model, tokenizer, rows: list[dict], max_length: int, device: str,
               latency_samples: int, throughput_samples: int, batch_size: int) -> dict:
    import torch
    from frauddistill.student.collator import neural_collate
    from frauddistill.student.dataset import build_neural_examples

    jobs = sorted(_prepare_jobs(rows), key=lambda row: row["id"])
    latency_jobs = jobs[:latency_samples]
    with torch.inference_mode():
        for job in latency_jobs[:min(3, len(latency_jobs))]:
            example = build_neural_examples([job], max_length=max_length, use_teacher_soft=True,
                                            use_pairwise=False)
            batch = neural_collate(example, tokenizer, max_length=max_length, architecture="standard")
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                     for key, value in batch.items()}
            _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    latencies = []
    for job in latency_jobs:
        example = build_neural_examples([job], max_length=max_length, use_teacher_soft=True,
                                        use_pairwise=False)
        batch = neural_collate(example, tokenizer, max_length=max_length, architecture="standard")
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                 for key, value in batch.items()}
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)

    throughput_jobs = jobs[:throughput_samples]
    examples = build_neural_examples(throughput_jobs, max_length=max_length, use_teacher_soft=True,
                                     use_pairwise=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda batch: neural_collate(
                            batch, tokenizer, max_length=max_length, architecture="standard"))
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                     for key, value in batch.items()}
            _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    retained = [min(len(tokenizer(
        f"Task: Detect whether the answer creates a fraud-related behavior error.\n\n"
        f"[QUERY]\n{job.get('user_query', '')}\n\n[ANSWER]\n{job.get('target_model_answer', '')}"
    ).input_ids), max_length) for job in throughput_jobs]
    return {
        "batch_size_for_primary_scoring": batch_size,
        "latency_batch": 1,
        "latency_samples": len(latency_jobs),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "throughput_batch": batch_size,
        "throughput_samples": len(throughput_jobs),
        "throughput_seconds": round(elapsed, 4),
        "rows_per_s": round(len(throughput_jobs) / max(elapsed, 1e-9), 3),
        "seconds_per_1000_rows": round(elapsed * 1000 / max(len(throughput_jobs), 1), 3),
        "input_tokens_per_s": round(sum(retained) / max(elapsed, 1e-9), 3),
        "output_tokens": 0,
    }


def run_scale_eval(model_id: str, checkpoint: str | None = None, device: str = "cuda",
                   dtype: str = "bf16", batch_size: int | None = None,
                   dry_run: bool = False, skip_selection: bool = False,
                   gpu_hour_cny: float | None = None) -> dict:
    scale = _cfg()
    models = {model["id"]: model for model in scale["models"]}
    lock_path = SUPP_DIR / "prepared" / "asset_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("run scripts/supp_prepare.py before evaluation")
    lock = read_json(lock_path)
    if lock["configs"]["scale"]["sha256"] != sha256_file(CONFIG):
        raise ValueError("scale config changed after preparation; rerun scripts/supp_prepare.py")
    if model_id not in models:
        raise ValueError(f"unknown model id: {model_id}")
    model_cfg = models[model_id]
    out_dir = RESULT_ROOT / model_id
    run_dir = SUPP_DIR / "runs" / model_id / "final_distilled_student"
    checkpoint_path = resolve_repo(checkpoint) if checkpoint else None
    plan = {
        "protocol": scale["protocol"],
        "model_id": model_id,
        "hf_id": model_cfg["hf_id"],
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "device": device,
        "dtype": dtype,
        "sets": {
            "core_dev": str(RAW_DIR / "core_dev.jsonl"),
            "core_test": str(RAW_DIR / "core_test.jsonl"),
            "relation_calibration": str(RAW_DIR / "relation_calibration.jsonl"),
            "hard_safe_calibration": str(RAW_DIR / "hard_safe_calibration.jsonl"),
            "relation_test": str(RAW_DIR / "relation_test.jsonl"),
            "hard_safe_test": str(RAW_DIR / "hard_safe_test.jsonl"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluation_plan.json", plan)
    for set_name, path_text in plan["sets"].items():
        expected = lock["prepared"]["raw_files"][set_name]
        path = Path(path_text)
        if path.is_dir() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"evaluation asset changed after freeze: {set_name}")
    if dry_run:
        print(json.dumps({**plan, "status": "DRY_RUN_NO_MODEL_LOAD"}, ensure_ascii=False, indent=2))
        return plan
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path is None and not run_dir.exists():
        raise FileNotFoundError(f"training run not found: {run_dir}")

    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if dtype == "auto":
            dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        torch.cuda.reset_peak_memory_stats()
    elif dtype == "auto":
        dtype = "fp32"
    if device == "cpu" and dtype != "fp32":
        raise ValueError("CPU scale evaluation must use --dtype fp32")
    max_length = int(scale["training"]["max_length"])
    score_batch = int(batch_size or scale["cost"]["score_batch"])

    rows = {name: read_jsonl(path) for name, path in plan["sets"].items()}
    evaluation_started = time.perf_counter()
    selection_summary = None
    if checkpoint_path is None and not skip_selection:
        checkpoint_path, selection_summary = _select_checkpoint(
            model_cfg, run_dir, rows["core_dev"], max_length, score_batch, device, dtype,
            fast_subset=300, top_k=3, scale=scale,
        )
        write_json(out_dir / "checkpoint_selection.json", selection_summary)
        plan["checkpoint"] = str(checkpoint_path)
        write_json(out_dir / "evaluation_plan.json", plan)
    elif checkpoint_path is None:
        candidates = _candidate_checkpoints(run_dir)
        if not candidates:
            raise FileNotFoundError("no checkpoints found")
        checkpoint_path = candidates[-1]

    started = time.perf_counter()
    model, tokenizer = _model_for(model_cfg, checkpoint_path, max_length, dtype)
    model.to(device).eval()
    scored = {}
    timings = {}
    for name in ("core_dev", "core_test", "relation_calibration",
                 "hard_safe_calibration", "relation_test", "hard_safe_test"):
        preds, timing = _score_rows(model, tokenizer, rows[name], max_length, score_batch, device)
        scored[name] = preds
        timings[name] = timing
        write_jsonl(out_dir / f"predictions_{name}.jsonl", preds)

    maps = {name: _score_map(scored[name], rows[name]) for name in rows}
    threshold = select_target_threshold(
        rows["relation_calibration"], {k: v["risk_score"] for k, v in maps["relation_calibration"].items()},
        rows["hard_safe_calibration"], {k: v["risk_score"] for k, v in maps["hard_safe_calibration"].items()},
        recall_min=float(scale["calibration"]["recall_min"]),
        fpr_max=float(scale["calibration"]["hard_safe_fpr_max"]),
        frozen_threshold=float(scale["calibration"]["frozen_source_threshold"]),
    )
    target_t = float(threshold["threshold"])
    for name in ("relation_test", "hard_safe_test", "core_test"):
        for pred in scored[name]:
            pred["pred_label"] = "unsafe" if pred["risk_score"] >= target_t else "safe"
            pred["threshold"] = target_t
        write_jsonl(out_dir / f"predictions_{name}.jsonl", scored[name])

    def metrics_for(name: str) -> dict:
        y = [binary_label(row) for row in rows[name]]
        pred_map = maps[name]
        scores = [float(pred_map[row_key(row)]["risk_score"]) for row in rows[name]]
        final = [1 if score >= target_t else 0 for score in scores]
        return binary_metrics(y, scores, final, threshold=target_t)

    core = metrics_for("core_test")
    core["four_class"] = four_class_metrics(rows["core_test"], maps["core_test"])
    online = metrics_for("relation_test")
    hs_test = metrics_for("hard_safe_test")
    dev = _dev_metrics(rows["core_dev"], maps["core_dev"], threshold=0.5)
    benchmark = _benchmark(
        model, tokenizer, rows["relation_test"], max_length, device,
        int(scale["cost"]["latency_samples"]), int(scale["cost"]["throughput_samples"]), score_batch,
    )
    import torch as _torch
    benchmark["peak_vram_gb"] = round(_torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if device == "cuda" else None
    evaluation_seconds = time.perf_counter() - evaluation_started

    training_meta_path = checkpoint_path.parent / "final_distill_metrics.json"
    training_meta = read_json(training_meta_path) if training_meta_path.exists() else {}
    train_seconds = float(training_meta.get("wall_seconds") or 0.0)
    price = gpu_hour_cny if gpu_hour_cny is not None else scale["cost"].get("gpu_hour_cny")
    training_gpu_seconds = 0.0 if model_cfg.get("reuse") else train_seconds
    gpu_hours = (training_gpu_seconds + evaluation_seconds) / 3600
    cost = {
        "training": {
            "measurement": "historical_reuse" if model_cfg.get("reuse") else "new_gpu_run",
            "wall_seconds": train_seconds,
            "peak_vram_gb": training_meta.get("peak_vram_gb"),
            "wall_hours": round(train_seconds / 3600, 5),
            "billable_gpu_hours": round(training_gpu_seconds / 3600, 5),
            "input_tokens": int(training_meta.get("train_input_tokens", 0) or 0),
            "input_tokens_method": "estimated retained tokens per sampled row x epochs",
            "output_tokens": int(training_meta.get("train_output_tokens", 0) or 0),
        },
        "evaluation": {
            "wall_seconds": round(evaluation_seconds, 2),
            "peak_vram_gb": benchmark["peak_vram_gb"],
            "per_set": timings,
            "total_rows": int(sum(item["rows"] for item in timings.values())),
            "total_input_tokens": int(sum(item["input_tokens"] for item in timings.values())),
            "total_output_tokens": 0,
            "retry_tokens": 0,
            "serving": benchmark,
        },
        "total_gpu_hours": round(gpu_hours, 5),
        "gpu_hour_cny": price,
        "estimated_total_gpu_cny": round(gpu_hours * float(price), 6) if price is not None else None,
        "classifier_output_tokens": 0,
        "note": "Wall time includes checkpoint loading and all frozen evaluation sets; serving latency excludes load time.",
    }
    summary = {
        "protocol": scale["protocol"],
        "model": model_cfg,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path / "adapter_model.safetensors")
                            if (checkpoint_path / "adapter_model.safetensors").exists() else None,
        "threshold_selection": threshold,
        "metrics": {
            "core_dev": dev,
            "core_test": core,
            "relation_test_online": online,
            "hard_safe_test": hs_test,
        },
        "cost": cost,
        "selection": selection_summary,
        "eval_seconds": round(evaluation_seconds, 2),
        "device": device,
        "dtype": dtype,
        "max_length": max_length,
        "score_batch": score_batch,
    }
    write_json(out_dir / "threshold_selection.json", threshold)
    write_json(out_dir / "metrics.json", summary["metrics"])
    write_json(out_dir / "cost.json", cost)
    write_json(out_dir / "summary.json", summary)
    del model
    gc.collect()
    if device == "cuda":
        _torch.cuda.empty_cache()
    print(json.dumps({
        "status": "DONE",
        "model_id": model_id,
        "checkpoint": str(checkpoint_path),
        "threshold": target_t,
        "core_macro_f1": core["macro_f1"],
        "relation_macro_f1": online["macro_f1"],
        "relation_recall": online["recall"],
        "relation_fpr": online["fpr"],
        "hard_safe_fpr": hs_test["fpr"],
        "peak_vram_gb": benchmark["peak_vram_gb"],
    }, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--gpu-hour-cny", type=float, default=None)
    args = parser.parse_args(argv)
    run_scale_eval(args.model_id, args.checkpoint, args.device, args.dtype,
                   args.batch_size, args.dry_run, args.skip_selection, args.gpu_hour_cny)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
