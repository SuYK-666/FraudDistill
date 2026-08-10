# -*- coding: utf-8 -*-
"""E1-FINAL-TRIAD-v4 runner: relational q/y/q+y ablation.

Phases:
  p0              protocol lock
  a-reconcile     E1-A prevalence reconciliation (offline)
  b-build-tasks   B panel task manifests (offline)
  b-gen-y         generate B1/B2/B3 response + refusal texts (API)
  b-gen-qbenign   generate benign counterpart queries (API)
  b-gold          double-judge all new pairs (API)
  b-adjudicate    adjudicate disagreements (API)
  b-assemble      assemble 6000-row panel (offline)
  b-split-freeze  family split + audits + freeze (offline)
  b-train         M0 LR + M1 XLM-R training (offline, CPU)
  b-anchor-local  run frozen local models on anchor (offline)
  b-anchor-llm    M2 Qwen / M3 DeepSeek four-view single judge (API)
  b-stats         statistics + gates (offline)
  c-replay        E1-C independent replay (offline)
  final-report    report + tables (offline)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.io import read_json, read_jsonl, sha_text, write_json, write_jsonl
from frauddistill.e1_final_v3.api_executor import execute_json_tasks, execute_tasks

CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_final_triad_v4.yaml"


def rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_clean() -> bool:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip() == ""
    except Exception:
        return False


def limits_for(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg["budget"]
    return {
        "hard_stop_total_cny": float(b.get("hard_stop_total_cny", 0) or 0),
        "qwen_hard_stop_cny": float(b.get("qwen_hard_stop_cny", 0) or 0),
        "deepseek_hard_stop_cny": float(b.get("deepseek_hard_stop_cny", 0) or 0),
    }


def concurrency_for(cfg: dict[str, Any]) -> dict[str, int]:
    a = cfg["api"]
    return {
        "qwen": int(a["effective_qwen_concurrency"]),
        "deepseek": int(a["effective_deepseek_concurrency"]),
        "adjudicator": int(a["effective_adjudicator_concurrency"]),
    }


def progress(name: str, done: int, total: int) -> None:
    width = 40
    filled = int(width * done / max(1, total))
    print(f"[{name}] [{'#' * filled}{'.' * (width - filled)}] {done}/{total} {100 * done / max(1, total):5.1f}%", flush=True)


# ---------------------------------------------------------------- phases


def phase_p0(cfg: dict[str, Any], args) -> dict[str, Any]:
    out_dir = rel(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = {
        "protocol": cfg["experiment"]["protocol"],
        "seed": cfg["experiment"]["seed"],
        "baseline_commit": git_commit(),
        "config_sha256": sha_text(CONFIG_PATH.read_text(encoding="utf-8")),
        "budget": cfg["budget"],
        "strata_quotas": cfg["e1_v4"]["strata"],
        "target_gates": {
            "q_y_macro_f1": 0.90,
            "delta_joint_min": 0.05,
            "qy_gt_best_single_seeds": "4/5",
            "balanced_ratio_tolerance": 0.01,
        },
        "scientific_gate": ["delta>0", "ci_lower>0", "holm_p<0.05", "qy>wrong_qy", "shortcut_audits_pass"],
        "anchor_rule": "Anchor 冻结后一次性消费；失败不删样本重跑",
        "b1_amendment": "真实 y-matched 供给不足（SD exact-y 跨标签仅 136 行且为 Gold 不一致）；按协议 amendment：B1 采用受控构造对（生成 comply-style y + 真实诈骗 q + 生成良性 q），全部经双 Gold 确认；synthetic 占比超出 25% 软上限，以本 amendment 固定为 B1=2000。",
        "b2_amendment": "真实同 q 双标签对供给不足（SD base/levelup 同内容不同标签为 Gold 不一致，42 对冲突已重判）；B2 负侧采用 AEGIS 真实拒答（en）+ 生成防御性回复（zh），全部经双 Gold 确认。",
        "frozen": ["A7500 target generation", "v3.2 source pools content", "canonical gold protocol"],
    }
    write_json(out_dir / "E1_V4_PROTOCOL_LOCK.json", lock)
    return {"status": "P0_LOCKED", "lock": lock}


def phase_a_reconcile(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.a_reconcile import reconcile_a
    out_dir = rel(cfg["data"]["output_dir"])
    result = reconcile_a(rel(cfg["data"]["v32_dir"]) / "E1_V32_REAL_POOL.jsonl", out_dir)
    return {"status": "A_RECONCILED", "central_positives": result["central_positives"], "n": result["registry_rows"]}


def phase_b_build_tasks(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.panel import build_b_tasks
    out_dir = rel(cfg["data"]["output_dir"])
    manifest = build_b_tasks(cfg, rel(cfg["data"]["v32_dir"]), out_dir)
    return {"status": "TASKS_BUILT", "counts": manifest["counts"]}


def phase_b_gen_y(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.assemble import build_qbenign_tasks
    out_dir = rel(cfg["data"]["output_dir"])
    tasks = read_jsonl(out_dir / "E1_V4_GEN_Y_TASKS.jsonl")
    ref_tasks = read_jsonl(out_dir / "E1_V4_GEN_REFUSAL_TASKS.jsonl")
    if args.pilot:
        b1 = [t for t in tasks if t.get("task_kind") == "b1_y"]
        n = min(args.pilot, len(b1))
        idxs = sorted({round(i * (len(b1) - 1) / max(1, n - 1)) for i in range(n)})
        tasks = [b1[i] for i in idxs]
        ref_tasks = []
    total = len(tasks) + len(ref_tasks)
    done = 0
    def monitor(r):
        nonlocal done
        done += 1
        if done % 25 == 0:
            progress("gen-y", done, total)
    r1 = execute_tasks(
        tasks, output_path=out_dir / "E1_V4_GEN_Y_RESULTS.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    r2 = execute_tasks(
        ref_tasks, output_path=out_dir / "E1_V4_GEN_REFUSAL_RESULTS.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"gen_y": r1, "gen_refusal": r2}


def phase_b_gen_qbenign(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.assemble import build_qbenign_tasks
    out_dir = rel(cfg["data"]["output_dir"])
    n = build_qbenign_tasks(cfg, out_dir, pilot_n=args.pilot)
    tasks = read_jsonl(out_dir / "E1_V4_GEN_QBENIGN_TASKS.jsonl")
    result = execute_tasks(
        tasks, output_path=out_dir / "E1_V4_GEN_QBENIGN_RESULTS.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": n, "result": result}


def phase_b_gold(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.assemble import build_gold_tasks
    out_dir = rel(cfg["data"]["output_dir"])
    counts = build_gold_tasks(cfg, out_dir, pilot_n=args.pilot)
    new_tasks = read_jsonl(out_dir / "E1_V4_GOLD_TASKS_NEW.jsonl")
    salvage_tasks = read_jsonl(out_dir / "E1_V4_GOLD_TASKS_SALVAGE.jsonl")
    all_tasks = new_tasks + salvage_tasks
    result = execute_json_tasks(
        all_tasks, output_path=out_dir / "E1_V4_GOLD_VOTES.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"counts": counts, "result": result}


def phase_b_adjudicate(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.gold import adjudication_task, central_label, needs_adjudication, parse_vote
    out_dir = rel(cfg["data"]["output_dir"])
    votes: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(out_dir / "E1_V4_GOLD_VOTES.jsonl"):
        v = parse_vote(r)
        if v is None:
            continue
        votes.setdefault(r["response_id"], {})[r["judge"]] = v
    # row content (q,y) by response_id
    content: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(out_dir / "E1_V4_GOLD_VOTES.jsonl"):
        content.setdefault(r["response_id"], {"q_private": r.get("q_private", ""), "y_private": r.get("y_private", "")})
    existing_adj = {r["response_id"] for r in read_jsonl(out_dir / "E1_V4_GOLD_ADJUDICATION.jsonl") if r.get("status") == "ok"}
    tasks = []
    for rid, vv in votes.items():
        if rid in existing_adj:
            continue
        if needs_adjudication(vv.get("judge_a"), vv.get("judge_b")):
            row = {"response_id": rid, **content[rid]}
            tasks.append(adjudication_task(row, vv.get("judge_a"), vv.get("judge_b"), cfg, "E1-v4-adjudicate"))
    result = execute_json_tasks(
        tasks, output_path=out_dir / "E1_V4_GOLD_ADJUDICATION.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider={"qwen": concurrency_for(cfg)["adjudicator"]},
        pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def phase_b_assemble(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.assemble import assemble_panel
    out_dir = rel(cfg["data"]["output_dir"])
    audit = assemble_panel(cfg, out_dir)
    return {"status": "PANEL_ASSEMBLED", "audit": audit}


def phase_b_split_freezer(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.split_audit import build_wrong_q_map, freeze, run_audits, split_by_family
    out_dir = rel(cfg["data"]["output_dir"])
    panel = read_jsonl(out_dir / "E1_V4_PANEL_ALL.jsonl")
    frac = tuple(float(cfg["e1_v4"]["split_frac"][k]) for k in ["model_dev", "calibration", "anchor"])
    splits = split_by_family(panel, frac, seed=int(cfg["experiment"]["seed"]))
    audit = run_audits(splits["model_dev"], splits["calibration"], splits["anchor"], out_dir)
    wrong_q = build_wrong_q_map(splits["anchor"], rng_seed=int(cfg["experiment"]["seed"]))
    freeze(cfg, out_dir, splits["model_dev"], splits["calibration"], splits["anchor"], wrong_q)
    return {"status": "SPLIT_FROZEN", "audit": audit}


def phase_b_train(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v3.detector_v31 import ViewDetector, panel_rows_to_eval, run_seed as run_lr_seed
    from frauddistill.e1_final_v4.detectors import NeuralJointDetector, run_neural_seed
    out_dir = rel(cfg["data"]["output_dir"])
    dev = read_jsonl(out_dir / "E1_V4_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(out_dir / "E1_V4_PANEL_CALIBRATION.jsonl")
    anchor = read_jsonl(out_dir / "E1_V4_PANEL_ANCHOR.jsonl")
    wrong_q = {r["response_id"]: r["wrong_q"] for r in read_jsonl(out_dir / "E1_V4_WRONG_Q_MAP.jsonl")}
    seeds = cfg["e1_v4"]["seeds"]
    results = {}
    if args.mode in ("lr", "all"):
        for mode in ["q_only", "y_only", "q+y"]:
            per_seed = []
            for seed in seeds:
                r = run_lr_seed(dev, cal, anchor, mode, seed, cfg["e1_v4"])
                per_seed.append(r)
                results[f"lr_{mode}"] = per_seed
            print(f"[LR {mode}] done", flush=True)
    if args.mode in ("m1", "all"):
        for mode in ["q_only", "y_only", "q_y"]:
            per_seed = []
            for seed in seeds:
                r = run_neural_seed(dev, cal, anchor, mode, seed, cfg, wrong_q_map=None, out_dir=out_dir)
                per_seed.append(r)
                results[f"m1_{mode}"] = per_seed
                write_json(out_dir / "E1_V4_TRAIN_PROGRESS.json", {"mode": mode, "seed": seed, "done": True})
            print(f"[M1 {mode}] done", flush=True)
    write_json(out_dir / "E1_V4_TRAIN_RESULTS.json", results)
    return {"status": "TRAINED", "modes": sorted(results)}


def phase_b_anchor_local(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.detectors import NeuralJointDetector
    from frauddistill.e1_final_v4.stats import eval_rows
    out_dir = rel(cfg["data"]["output_dir"])
    anchor = read_jsonl(out_dir / "E1_V4_PANEL_ANCHOR.jsonl")
    wrong_q = {r["response_id"]: r["wrong_q"] for r in read_jsonl(out_dir / "E1_V4_WRONG_Q_MAP.jsonl")}
    seeds = cfg["e1_v4"]["seeds"]
    preds: dict[str, dict[str, Any]] = {}
    for mode in ["q_only", "y_only", "q_y", "wrong_q_y"]:
        # wrong_q_y view = frozen q_y model evaluated with wrong q (never trained separately)
        load_mode = "q_y" if mode == "wrong_q_y" else mode
        per_seed = []
        for seed in seeds:
            det = NeuralJointDetector(mode, model_name=cfg["e1_v4"]["neural"]["model_name"], max_length=int(cfg["e1_v4"]["neural"]["max_length"]), seed=seed)
            model_dir = out_dir / "models" / f"{load_mode}_seed{seed}"
            if not (model_dir / "meta.json").exists():
                continue
            det.model = type(det.model).from_pretrained(str(model_dir))
            det.tokenizer = type(det.tokenizer).from_pretrained(str(model_dir))
            meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
            det.threshold = float(meta.get("threshold", 0.5))
            wq = wrong_q if mode == "wrong_q_y" else None
            scores = det.predict_proba(anchor, wq)
            rows = eval_rows(anchor, scores, det.threshold, mode)
            per_seed.append({"seed": seed, "threshold": det.threshold, "rows": rows})
        preds[mode] = per_seed
    write_json(out_dir / "E1_V4_ANCHOR_LOCAL_PREDS.json", preds)
    return {"status": "ANCHOR_LOCAL_DONE", "modes": sorted(preds)}


def phase_b_anchor_llm(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.judge_views import VIEWS, view_task
    out_dir = rel(cfg["data"]["output_dir"])
    anchor = read_jsonl(out_dir / "E1_V4_PANEL_ANCHOR.jsonl")
    wrong_q = {r["response_id"]: r["wrong_q"] for r in read_jsonl(out_dir / "E1_V4_WRONG_Q_MAP.jsonl")}
    tasks = []
    for row in anchor:
        for view in VIEWS:
            for provider_key in ["gold_qwen_v31", "gold_deepseek_v31"]:
                tasks.append(view_task(row, view, provider_key, cfg, "E1-v4-anchor-view", wrong_q))
    result = execute_json_tasks(
        tasks, output_path=out_dir / "E1_V4_ANCHOR_VIEW_VOTES.jsonl", ledger_path=rel(cfg["data"]["v4_budget_ledger"]),
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def phase_b_stats(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_v10.metrics import binary_metrics
    from frauddistill.e1_final_v4.judge_views import view_label
    from frauddistill.e1_final_v4.stats import aggregate_results, eval_rows
    import numpy as np
    def np_mean(x): return float(np.mean(x))
    out_dir = rel(cfg["data"]["output_dir"])
    anchor = read_jsonl(out_dir / "E1_V4_PANEL_ANCHOR.jsonl")
    local = read_json(out_dir / "E1_V4_ANCHOR_LOCAL_PREDS.json")
    train = read_json(out_dir / "E1_V4_TRAIN_RESULTS.json")
    votes = read_jsonl(out_dir / "E1_V4_ANCHOR_VIEW_VOTES.jsonl")
    seeds = cfg["e1_v4"]["seeds"]
    out: dict[str, Any] = {}

    # ---- M1 local: per-seed metrics + seed-level gate
    m1 = {}
    for mode in ["q_only", "y_only", "q_y", "wrong_q_y"]:
        per_seed = []
        for s in seeds:
            rows = local[mode][seeds.index(s)]["rows"]
            per_seed.append(binary_metrics(rows))
        m1[mode] = {"per_seed": per_seed, "mean": {k: float(np_mean([x[k] for x in per_seed])) for k in ["macro_f1", "auprc", "recall", "fpr", "precision", "auroc", "balanced_accuracy", "mcc" if "mcc" in per_seed[0] else "accuracy"]}}
    seed_wins = []
    for i in range(len(seeds)):
        qy = m1["q_y"]["per_seed"][i]["macro_f1"]
        best = max(m1["q_only"]["per_seed"][i]["macro_f1"], m1["y_only"]["per_seed"][i]["macro_f1"])
        seed_wins.append(qy > best)
    m1["qy_beats_best_single"] = {"per_seed": seed_wins, "n_wins": sum(seed_wins), "n_seeds": len(seeds)}
    out["m1_local"] = m1

    # ---- M0 LR (from training phase)
    lr = {}
    for mode in ["q_only", "y_only", "q+y"]:
        rows_key = f"lr_{mode}"
        if rows_key not in train:
            continue
        per_seed = [r["anchor"] for r in train[rows_key]]
        lr[mode] = {"per_seed": per_seed, "mean": {k: float(np_mean([x[k] for x in per_seed])) for k in ["macro_f1", "auprc", "recall", "fpr"] if k in per_seed[0]}}
    out["m0_lr"] = lr

    # ---- M2/M3 LLM judge views
    votes_by_provider: dict[str, dict[str, dict[str, int]]] = {"qwen": {}, "deepseek": {}}
    for r in votes:
        prov = str(r.get("response_model", "") or "").lower()
        key = "qwen" if "qwen" in prov else ("deepseek" if "deepseek" in prov else None)
        if key is None:
            continue
        view = str(r.get("judge", "")).replace("view_", "")
        lab = view_label(r)
        if lab is None:
            continue
        votes_by_provider[key].setdefault(view, {})[r["response_id"]] = lab
    llm = {}
    for prov, views in votes_by_provider.items():
        llm[prov] = {}
        for view, lab_map in views.items():
            rows = []
            for a in anchor:
                lab = lab_map.get(a["response_id"])
                if lab is None:
                    continue
                rows.append({**a, "pred": lab, "score": float(lab)})
            llm[prov][view] = binary_metrics(rows)
    out["m2_m3_llm"] = llm

    # ---- aggregate statistics for M1 (mean seed) + LLM views
    for key, preds_source in [("m1", m1), ("llm_qwen", llm.get("qwen")), ("llm_deepseek", llm.get("deepseek"))]:
        if not preds_source:
            continue
        # build eval rows per view
        by_view = {}
        for view in ["q_only", "y_only", "q_y", "wrong_q_y"]:
            if key == "m1":
                rows = local[view][0]["rows"] if local.get(view) else []
            else:
                rows = []
                lab_map = preds_source.get(view, {})
                for a in anchor:
                    lab = lab_map.get(a["response_id"])
                    if lab is None:
                        continue
                    rows.append({**a, "pred": lab, "score": float(lab)})
            by_view[view] = rows
        if by_view.get("q_y"):
            res = aggregate_results(anchor, by_view, iterations=int(cfg["e1_v4"]["bootstrap_iterations"]), seed=int(cfg["experiment"]["seed"]))
            out[f"stats_{key}"] = res
    write_json(out_dir / "E1_V4_STATS.json", out)
    return {"status": "STATS_DONE", "keys": sorted(out)}


def phase_c_replay(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.c_replay import c_report
    from frauddistill.e1_final_v4.detectors import NeuralJointDetector
    out_dir = rel(cfg["data"]["output_dir"])
    a_rows = read_jsonl(rel(cfg["data"]["v32_dir"]) / "E1_V32_REAL_POOL.jsonl")
    # E1-C independence: exclude every canonical case that entered the B panel
    # (guide section 10.2: case/family-level, not row-level).
    b_rows = read_jsonl(out_dir / "E1_V4_PANEL_ALL.jsonl")
    b_cases = {r.get("canonical_case_id") for r in b_rows if r.get("canonical_case_id")}
    n_before = len(a_rows)
    pos_before = sum(1 for r in a_rows if int(r.get("gold_central", 0) or 0) == 1)
    a_rows = [r for r in a_rows if r.get("canonical_case_id") not in b_cases]
    n_after = len(a_rows)
    pos_after = sum(1 for r in a_rows if int(r.get("gold_central", 0) or 0) == 1)
    seeds = cfg["e1_v4"]["seeds"]
    predictors = {}
    for mode in ["q_y", "y_only"]:
        per_seed = []
        for seed in seeds:
            model_dir = out_dir / "models" / f"{mode}_seed{seed}"
            if not (model_dir / "meta.json").exists():
                continue
            det = NeuralJointDetector(mode, model_name=cfg["e1_v4"]["neural"]["model_name"], max_length=int(cfg["e1_v4"]["neural"]["max_length"]), seed=seed)
            det.model = type(det.model).from_pretrained(str(model_dir))
            det.tokenizer = type(det.tokenizer).from_pretrained(str(model_dir))
            meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
            thr = float(meta.get("threshold", 0.5))
            per_seed.append((seed, lambda rows, d=det: d.predict_proba(rows), thr))
        predictors[mode] = per_seed
    result = c_report(cfg, out_dir, a_rows, predictors)
    result["exclusion"] = {"b_cases": len(b_cases), "a_rows_before": n_before, "a_positives_before": pos_before,
                           "a_rows_after_exclusion": n_after, "a_positives_after_exclusion": pos_after,
                           "note": "C uses only canonical cases that never entered the B panel (case/family-level independence; guide 10.2)."}
    return {"status": "C_DONE", "aggregate": result["aggregate"], "exclusion": result["exclusion"]}


def phase_final_report(cfg: dict[str, Any], args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.reporting import build_report
    out_dir = rel(cfg["data"]["output_dir"])
    rep = build_report(cfg, out_dir)
    return {"status": "REPORT_READY", "report": rep}


PHASES = {
    "p0": phase_p0,
    "a-reconcile": phase_a_reconcile,
    "b-build-tasks": phase_b_build_tasks,
    "b-gen-y": phase_b_gen_y,
    "b-gen-qbenign": phase_b_gen_qbenign,
    "b-gold": phase_b_gold,
    "b-adjudicate": phase_b_adjudicate,
    "b-assemble": phase_b_assemble,
    "b-split-freeze": phase_b_split_freezer,
    "b-train": phase_b_train,
    "b-anchor-local": phase_b_anchor_local,
    "b-anchor-llm": phase_b_anchor_llm,
    "b-stats": phase_b_stats,
    "c-replay": phase_c_replay,
    "final-report": phase_final_report,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--mode", default="all", choices=["all", "lr", "m1"])
    args = parser.parse_args()
    cfg = load_config()
    t0 = time.time()
    result = PHASES[args.phase](cfg, args)
    result["wall_seconds"] = round(time.time() - t0, 2)
    print(json.dumps({"phase": args.phase, **result}, ensure_ascii=False, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
