# -*- coding: utf-8 -*-
"""E1 v4 final balance: replace B3 zh generated-defensive negatives with real
v32 natural safe responses (provenance balance) and swap 43 B3 SD positives for
unused generated positives (provenance balance). Zero new generation; only gold.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.api_executor import execute_json_tasks
from frauddistill.e1_final_v3.io import norm, read_json, read_jsonl, sha_text, write_json, write_jsonl
from frauddistill.e1_final_v4.gold import central_label, gold_judge_task, parse_vote
from frauddistill.e1_final_v4.panel import _row_for_gold

CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_final_triad_v4.yaml"
OUT_DIR = ROOT / "data" / "prepared" / "e1_final_triad_v4"
V32_POOL = ROOT / "data" / "prepared" / "e1_final_triad_v32" / "E1_V32_REAL_POOL.jsonl"


def load_cfg():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def git_clean() -> bool:
    import subprocess
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip() == ""
    except Exception:
        return False


def limits_for(cfg):
    b = cfg["budget"]
    return {
        "hard_stop_total_cny": float(b.get("hard_stop_total_cny", 0) or 0),
        "qwen_hard_stop_cny": float(b.get("qwen_hard_stop_cny", 0) or 0),
        "deepseek_hard_stop_cny": float(b.get("deepseek_hard_stop_cny", 0) or 0),
    }


def concurrency_for(cfg):
    a = cfg["api"]
    return {
        "qwen": int(a["effective_qwen_concurrency"]),
        "deepseek": int(a["effective_deepseek_concurrency"]),
        "adjudicator": int(a["effective_adjudicator_concurrency"]),
    }


def _vote_index():
    out = {}
    for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_VOTES.jsonl"):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _adj_index():
    out = {}
    for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl"):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _resolved(rid, votes, adj):
    vv = votes.get(rid) or {}
    a, b = central_label(vv.get("judge_a")), central_label(vv.get("judge_b"))
    if a is not None and a == b and not (vv.get("judge_a", {}).get("uncertain") or vv.get("judge_b", {}).get("uncertain")):
        return a, "double_agree"
    av = central_label((adj.get(rid) or {}).get("adjudicator"))
    if av is not None:
        return av, "adjudicated"
    if a is not None and b is not None and a == b:
        return a, "double_agree_uncertain"
    return None, "unresolved"


def build_assignments(cfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(OUT_DIR / "E1_V4_PANEL_ALL.jsonl")
    panel_rids = {r["response_id"] for r in rows}
    panel_q = {norm(r["q_private"]) for r in rows}

    # ---- v32 real negatives for B3 zh generated_defensive rows
    v32_q = collections.defaultdict(list)
    for r in read_jsonl(V32_POOL):
        if r.get("language") == "zh" and int(r.get("gold_central", -1)) == 0:
            v32_q[norm(r.get("q_private", ""))].append(r)
    used_y = set()
    neg_assign = []
    for r in sorted([x for x in rows if x["stratum"].startswith("b3") and x["language"] == "zh" and x["provenance"] == "generated_defensive" and int(x["gold_central"]) == 0],
                    key=lambda x: x["response_id"]):
        cands = sorted(v32_q.get(norm(r["q_private"]), []), key=lambda x: len(x.get("y_private", "")))
        chosen = None
        for c in cands:
            y = (c.get("y_private") or "").strip()
            if y and norm(y) not in used_y and len(y) >= 400:
                chosen = y
                break
        if chosen is None:
            continue
        used_y.add(norm(chosen))
        rid = r["response_id"].removesuffix("-NEG2") + "-NEG3" if r["response_id"].endswith("-NEG2") else r["response_id"] + "-NEG3"
        neg_assign.append({
            **r, "response_id": rid, "old_response_id": r["response_id"], "y_private": chosen,
            "provenance": "real_target_v32", "source": "v32_real_zh",
        })

    # ---- swap 43 B3 SD positives for unused generated positives
    votes = _vote_index()
    adj = _adj_index()
    m = read_json(OUT_DIR / "E1_V4_TASK_MANIFEST.json")
    pos_assign = []
    for k, it in enumerate(m["b3_gen_qs"]):
        rid = f"E1-V4-B3-PAIR-{k:04d}-POS"
        if rid in panel_rids:
            continue
        lab, _ = _resolved(rid, votes, adj)
        if lab != 1:
            continue
        q = it["q_private"]
        gen_y = {}
        for r in read_jsonl(OUT_DIR / "E1_V4_GEN_Y_RESULTS.jsonl"):
            if r.get("response_id") == it["response_id"] and r.get("status") == "ok":
                gen_y = r
                break
        y = (gen_y.get("text") or "").strip()
        if not y:
            continue
        pos_assign.append({
            "response_id": rid, "q_private": q, "y_private": y, "language": it["language"],
            "canonical_case_id": it["canonical_case_id"], "stratum": "b3_context_stable_natural",
            "family_id": it["canonical_case_id"], "pair_id": f"B3-{it['canonical_case_id']}",
            "provenance": "generated_y", "source": "b3_gen_unused",
        })
        if len(pos_assign) >= 43:
            break

    write_jsonl(OUT_DIR / "E1_V4_NEG_V3_ASSIGNMENTS.jsonl", neg_assign)
    write_jsonl(OUT_DIR / "E1_V4_POS_V2_ASSIGNMENTS.jsonl", pos_assign)
    return neg_assign, pos_assign, {"neg": len(neg_assign), "pos": len(pos_assign)}


def phase_build(cfg, args) -> dict[str, Any]:
    neg, pos, stats = build_assignments(cfg)
    tasks = []
    for a in neg + pos:
        row = _row_for_gold(a["response_id"], a["q_private"], a["y_private"], a["language"], a["canonical_case_id"],
                            a["stratum"], a["family_id"], a["pair_id"], a["provenance"])
        for judge in ["judge_a", "judge_b"]:
            tasks.append(gold_judge_task(row, judge, cfg, "E1-v4-gold-final-balance"))
    write_jsonl(OUT_DIR / "E1_V4_GOLD_TASKS_FINAL_BALANCE.jsonl", tasks)
    return {"status": "BUILT", "stats": stats, "n_tasks": len(tasks)}


def phase_gold(cfg, args) -> dict[str, Any]:
    tasks = read_jsonl(OUT_DIR / "E1_V4_GOLD_TASKS_FINAL_BALANCE.jsonl")
    result = execute_json_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GOLD_VOTES.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def phase_adjudicate(cfg, args) -> dict[str, Any]:
    from frauddistill.e1_final_v4.gold import adjudication_task, needs_adjudication
    votes = _vote_index()
    existing_adj = {r["response_id"] for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl") if r.get("status") == "ok"}
    rids = {a["response_id"] for a in read_jsonl(OUT_DIR / "E1_V4_NEG_V3_ASSIGNMENTS.jsonl")} | \
           {a["response_id"] for a in read_jsonl(OUT_DIR / "E1_V4_POS_V2_ASSIGNMENTS.jsonl")}
    content = {}
    for path in ["E1_V4_NEG_V3_ASSIGNMENTS.jsonl", "E1_V4_POS_V2_ASSIGNMENTS.jsonl"]:
        for a in read_jsonl(OUT_DIR / path):
            content[a["response_id"]] = {"q_private": a["q_private"], "y_private": a["y_private"]}
    tasks = []
    for rid in sorted(rids):
        if rid in existing_adj:
            continue
        vv = votes.get(rid) or {}
        if needs_adjudication(vv.get("judge_a"), vv.get("judge_b")):
            tasks.append(adjudication_task(content[rid], vv.get("judge_a"), vv.get("judge_b"), cfg, "E1-v4-adjudicate-final-balance"))
    result = execute_json_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider={"qwen": concurrency_for(cfg)["adjudicator"]},
        pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def phase_apply(cfg, args) -> dict[str, Any]:
    rows = read_jsonl(OUT_DIR / "E1_V4_PANEL_ALL.jsonl")
    votes = _vote_index()
    adj = _adj_index()
    neg_assign = read_jsonl(OUT_DIR / "E1_V4_NEG_V3_ASSIGNMENTS.jsonl")
    pos_assign = read_jsonl(OUT_DIR / "E1_V4_POS_V2_ASSIGNMENTS.jsonl")

    neg_ok = []
    for a in neg_assign:
        lab, m = _resolved(a["response_id"], votes, adj)
        if lab == 0:
            neg_ok.append(a)
    pos_ok = []
    for a in pos_assign:
        lab, m = _resolved(a["response_id"], votes, adj)
        if lab == 1:
            pos_ok.append(a)
    print(f"[apply] neg_ok={len(neg_ok)}/{len(neg_assign)} pos_ok={len(pos_ok)}/{len(pos_assign)}", flush=True)

    neg_old_ids = {a["old_response_id"] for a in neg_ok}
    pos_ok_rids = {a["response_id"] for a in pos_ok}
    out = []
    for r in rows:
        if r["response_id"] in neg_old_ids:
            a = next(x for x in neg_ok if x["old_response_id"] == r["response_id"])
            out.append({**r, "response_id": a["response_id"], "y_private": a["y_private"], "provenance": "real_target_v32",
                        "gold_method": "double_agree"})
            continue
        if r["provenance"] == "source_derived_open_control" and r["stratum"].startswith("b3") and int(r["gold_central"]) == 1 and pos_ok:
            # replace an SD pos with an unused generated pos (in assignment order)
            a = pos_ok.pop(0)
            out.append({**r, "response_id": a["response_id"], "q_private": a["q_private"], "y_private": a["y_private"],
                        "canonical_case_id": a["canonical_case_id"], "family_id": a["family_id"], "pair_id": a["pair_id"],
                        "provenance": "generated_y", "gold_method": "double_agree"})
            continue
        out.append(r)

    counts = collections.Counter(r["stratum"] for r in out)
    labels = collections.Counter(int(r["gold_central"]) for r in out)
    langs = collections.Counter(r["language"] for r in out)
    prov = collections.Counter(r["provenance"] for r in out)
    audit = {"n_rows": len(out), "by_stratum": dict(counts), "by_label": dict(labels), "by_language": dict(langs),
             "by_provenance": dict(prov), "neg_v3_ok": len(neg_ok), "pos_v2_ok": len(pos_ok)}
    write_jsonl(OUT_DIR / "E1_V4_PANEL_ALL.jsonl", out)
    write_json(OUT_DIR / "E1_V4_PANEL_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return {"status": "APPLIED", "audit": audit}


PHASES = {"build": phase_build, "gold": phase_gold, "adjudicate": phase_adjudicate, "apply": phase_apply}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_cfg()
    t0 = time.time()
    result = PHASES[args.phase](cfg, args)
    result["wall_seconds"] = round(time.time() - t0, 2)
    print(json.dumps({"phase": args.phase, **result}, ensure_ascii=False, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
