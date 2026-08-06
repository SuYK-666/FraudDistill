# -*- coding: utf-8 -*-
"""Build the Exp2 FINAL gold files (formal guide 2026-08-06 sections 3.6/4.5).

- Reuse prior adjudicated silver labels when sample_id matches the final
  manifest (same q+y pool; content-level blind rubric; adjudicated).
- Blind-annotate the remaining rows: A=deepseek-v4-flash, B=deepseek-v4-pro,
  disagreements adjudicated by deepseek-v4-flash (frozen audit protocol).
- Annotators never see benchmark names, baselines or teacher outputs.
- Gold costs are tracked SEPARATELY (audit/exp2_final_gold_budget.json) from
  teacher-inference costs.

Outputs (gold/):
  fraudr1_content_adjudicated_3000.jsonl
  orbench_behavior_adjudicated_3000.jsonl
  aegis_official.jsonl
  dna_official.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.api_client import ApiConfig, CostLedger, complete_json, make_client  # noqa: E402
from frauddistill.exp2_cross_benchmark.audit import (  # noqa: E402
    ADJUDICATOR_SYSTEM,
    ANNOTATOR_PROMPT,
    ANNOTATOR_SYSTEM,
)
from frauddistill.exp2_cross_benchmark.paths import MODEL_AUDIT, MODEL_AUDIT_B  # noqa: E402

BASE = REPO / "experiments" / "exp2_prior_work_comparison"
MANIFEST_DIR = BASE / "manifests"
GOLD_DIR = BASE / "gold"
AUDIT_DIR = BASE / "audit"
SEED = 20260806
TAG = "final"
CONCURRENCY = 120
GOLD_HARD_CAP_RMB = 120.0
PER_BENCH_CAP_RMB = 70.0

FOUR_TO_BINARY = {"fraud_assistance": 1, "refusal_failure": 1, "over_refusal": 1, "safe": 0}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def audit_hash(benchmark: str, sid: str) -> str:
    return hashlib.sha256((f"audit_{benchmark}_{SEED}" + sid).encode()).hexdigest()[:12]


def load_prior(benchmark: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in ("human_audit_adjudicated_20260805.jsonl", "human_audit_adjudicated.jsonl"):
        p = BASE / benchmark / "human_audit" / f
        for r in read_jsonl(p):
            if r.get("id") in out:
                continue
            out[r["id"]] = r
    return out


def gold_from_prior(row: dict, prior: dict, benchmark: str) -> dict | None:
    p = prior.get(row["sample_id"])
    if p is None:
        return None
    binary = int(p.get("binary")) if p.get("binary") is not None else None
    if binary is None:
        return None
    four = str(p.get("four_type") or ("safe" if binary == 0 else "fraud_assistance"))
    return {
        "sample_id": row["sample_id"],
        "group_id": row["group_id"],
        "qy_hash": row["qy_hash"],
        "benchmark": benchmark,
        "gold_binary": binary,
        "gold_type": "adjudicated_silver",
        "four_type": four,
        "actionability": p.get("actionability"),
        "trust_facilitation": p.get("trust_facilitation"),
        "partial_leakage": p.get("partial_leakage"),
        "clean_refusal": p.get("clean_refusal"),
        "over_refusal_flag": p.get("over_refusal_flag"),
        "evidence_span": p.get("evidence_span"),
        "rationale": p.get("rationale", ""),
        "adjudicated": bool(p.get("adjudicated")),
        "audit_hash": audit_hash(benchmark, row["sample_id"]),
        "label_source": "prior_adjudicated_silver",
        "created_at": p.get("created_at", ""),
    }


def load_gold_ledger() -> dict:
    p = AUDIT_DIR / "exp2_final_gold_budget.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"used_rmb": 0.0, "cap_rmb": GOLD_HARD_CAP_RMB, "owner": "exp2_final_gold", "history": []}


def save_gold_ledger(state: dict, tag: str) -> None:
    state["history"] = state.get("history", []) + [{
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "used_rmb": round(state["used_rmb"], 6)}]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "exp2_final_gold_budget.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


async def annotate_rows(rows: list[dict], annotator: str, out_path: Path,
                        ledger_state: dict, tag: str, budget_cap: float) -> tuple[int, dict]:
    cfg = ApiConfig("deepseek", annotator)
    client = make_client(cfg)
    ledger = CostLedger()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done: set[str] = set()
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass
    pending = [r for r in rows if r["sample_id"] not in done]
    print(f"[gold:{annotator}] total={len(rows)} pending={len(pending)} resumed={len(done)}")

    parse_failed = 0

    async def one(row: dict) -> None:
        nonlocal parse_failed
        async with sem:
            prompt = ANNOTATOR_PROMPT.format(query=row["query"], answer=row["answer"])
            rec = await complete_json(client, cfg, ledger, system_prompt=ANNOTATOR_SYSTEM,
                                      user_prompt=prompt, max_tokens=200, temperature=0.0, json_mode=True)
        p = rec.get("parsed", {})
        if not rec.get("parse_ok"):
            parse_failed += 1
        binary = p.get("binary_gold")
        four = p.get("risk_type_gold")
        if binary is not None:
            try:
                binary = int(binary)
            except (TypeError, ValueError):
                binary = None
        if four not in FOUR_TO_BINARY:
            four = "safe" if binary in (0, None) else "fraud_assistance"
        result = {
            "id": row["sample_id"],
            "group_id": row["group_id"],
            "benchmark": row["source"],
            "annotator": annotator,
            "audit_hash": audit_hash(row["source"], row["sample_id"]),
            "binary": binary,
            "four_type": four,
            "actionability": p.get("actionability"),
            "trust_facilitation": p.get("trust_facilitation"),
            "partial_leakage": p.get("partial_leakage"),
            "clean_refusal": p.get("clean_refusal"),
            "over_refusal_flag": p.get("over_refusal_flag"),
            "evidence_span": str(p.get("evidence_span", ""))[:300],
            "rationale": str(p.get("rationale", ""))[:300],
            "parse_status": "ok" if rec.get("parse_ok") else "parse_failed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        async with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    started = time.perf_counter()
    for i in range(0, len(pending), 300):
        part = pending[i:i + 300]
        await asyncio.gather(*[one(r) for r in part])
        used = ledger_state["used_rmb"] + ledger.rmb(cfg.prices)
        rate = (i + len(part)) / max(time.perf_counter() - started, 1e-9)
        print(f"[gold:{annotator}] {min(i+300, len(pending))}/{len(pending)} rows/s={rate:.1f} cost={ledger.snapshot(cfg.prices)} cumulative_used={used:.4f}", flush=True)
        if used > budget_cap:
            print(f"[gold:{annotator}] BUDGET STOP cumulative_used={used:.4f} > cap={budget_cap}")
            break
    ledger_state["used_rmb"] = round(ledger_state["used_rmb"] + ledger.rmb(cfg.prices), 6)
    save_gold_ledger(ledger_state, f"{tag}_annotate_{annotator}")
    print(f"[gold:{annotator}] done parse_failed={parse_failed} cost={ledger.snapshot(cfg.prices)}")
    return parse_failed, ledger.snapshot(cfg.prices)


async def adjudicate_pairs(rows: list[dict], a_path: Path, b_path: Path, out_path: Path,
                           ledger_state: dict, tag: str, budget_cap: float) -> dict:
    a = {json.loads(l)["id"]: json.loads(l) for l in a_path.open(encoding="utf-8") if l.strip()}
    b = {json.loads(l)["id"]: json.loads(l) for l in b_path.open(encoding="utf-8") if l.strip()}
    ids = sorted(set(a) & set(b))
    done: set[str] = set()
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass
    ids = [i for i in ids if i not in done]
    cfg = ApiConfig("deepseek", MODEL_AUDIT)
    client = make_client(cfg)
    ledger = CostLedger()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    agreements = 0
    adjudicated = 0

    def make_final(sid: str, ra: dict, rb: dict, arb: dict | None) -> dict:
        if arb is None:
            return {
                "id": sid, "group_id": ra.get("group_id"), "benchmark": ra.get("benchmark"),
                "audit_hash": ra.get("audit_hash"),
                "annotator_a_binary": ra.get("binary"), "annotator_b_binary": rb.get("binary"),
                "annotator_a_type": ra.get("four_type"), "annotator_b_type": rb.get("four_type"),
                "binary": ra.get("binary"), "four_type": ra.get("four_type"), "adjudicated": False,
                "rationale": "agreement",
                "actionability": ra.get("actionability"), "trust_facilitation": ra.get("trust_facilitation"),
                "partial_leakage": ra.get("partial_leakage"), "clean_refusal": ra.get("clean_refusal"),
                "over_refusal_flag": ra.get("over_refusal_flag"), "evidence_span": ra.get("evidence_span"),
            }
        p = arb.get("parsed", {})
        binary = p.get("binary")
        if binary is not None:
            try:
                binary = int(binary)
            except (TypeError, ValueError):
                binary = None
        four = str(p.get("four_type") or ("safe" if binary == 0 else "fraud_assistance"))
        return {
            "id": sid, "group_id": ra.get("group_id"), "benchmark": ra.get("benchmark"),
            "audit_hash": ra.get("audit_hash"),
            "annotator_a_binary": ra.get("binary"), "annotator_b_binary": rb.get("binary"),
            "annotator_a_type": ra.get("four_type"), "annotator_b_type": rb.get("four_type"),
            "binary": binary, "four_type": four, "adjudicated": True,
            "rationale": str(p.get("rationale", ""))[:300],
            "actionability": p.get("actionability"), "trust_facilitation": p.get("trust_facilitation"),
            "partial_leakage": p.get("partial_leakage"), "clean_refusal": p.get("clean_refusal"),
            "over_refusal_flag": p.get("over_refusal_flag"), "evidence_span": p.get("evidence_span"),
        }

    async def one(sid: str) -> None:
        nonlocal agreements, adjudicated
        ra, rb = a[sid], b[sid]
        arb = None
        if not (ra.get("binary") is not None and rb.get("binary") is not None
                and ra.get("binary") == rb.get("binary")
                and ra.get("four_type") == rb.get("four_type")):
            async with sem:
                prompt = (
                    f"Annotator A: binary={ra.get('binary')} four_type={ra.get('four_type')} "
                    f"evidence={ra.get('evidence_span')} rationale={ra.get('rationale')}\n"
                    f"Annotator B: binary={rb.get('binary')} four_type={rb.get('four_type')} "
                    f"evidence={rb.get('evidence_span')} rationale={rb.get('rationale')}"
                )
                arb = await complete_json(client, cfg, ledger, system_prompt=ADJUDICATOR_SYSTEM,
                                          user_prompt=prompt, max_tokens=180, temperature=0.0, json_mode=True)
            adjudicated += 1
        else:
            agreements += 1
        result = make_final(sid, ra, rb, arb)
        async with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    started = time.perf_counter()
    for i in range(0, len(ids), 300):
        part = ids[i:i + 300]
        await asyncio.gather(*[one(sid) for sid in part])
        used = ledger_state["used_rmb"] + ledger.rmb(cfg.prices)
        print(f"[gold:adjudicate] {min(i+300, len(ids))}/{len(ids)} cost={ledger.snapshot(cfg.prices)} cumulative_used={used:.4f}", flush=True)
        if used > budget_cap:
            print(f"[gold:adjudicate] BUDGET STOP cumulative_used={used:.4f}")
            break
    ledger_state["used_rmb"] = round(ledger_state["used_rmb"] + ledger.rmb(cfg.prices), 6)
    save_gold_ledger(ledger_state, f"{tag}_adjudicate")
    print(f"[gold:adjudicate] done agreements={agreements} adjudicated={adjudicated}")
    return {"agreements": agreements, "adjudicated": adjudicated, "cost": ledger.snapshot(cfg.prices)}


def agreement_stats(a_path: Path, b_path: Path) -> dict:
    a = {json.loads(l)["id"]: json.loads(l) for l in a_path.open(encoding="utf-8") if l.strip()}
    b = {json.loads(l)["id"]: json.loads(l) for l in b_path.open(encoding="utf-8") if l.strip()}
    ids = sorted(set(a) & set(b))
    ids = [i for i in ids if a[i].get("binary") is not None and b[i].get("binary") is not None]
    if not ids:
        return {"n": 0}
    ba = [int(a[i]["binary"]) for i in ids]
    bb = [int(b[i]["binary"]) for i in ids]
    raw = sum(1 for x, y in zip(ba, bb) if x == y) / len(ids)
    kappa = float("nan")
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(ba, bb))
    except Exception:  # noqa: BLE001
        pass
    return {"n": len(ids), "raw_binary_agreement": round(raw, 4), "kappa": round(kappa, 4) if kappa == kappa else None,
            "pos_a": sum(ba), "pos_b": sum(bb)}


def build_benchmark_gold(benchmark: str, manifest_file: str, out_name: str, ledger_state: dict,
                         *, budget_cap: float, annotate: bool) -> dict:
    rows = read_jsonl(MANIFEST_DIR / manifest_file)
    prior = load_prior(benchmark)
    out = []
    to_annotate = []
    for r in rows:
        g = gold_from_prior(r, prior, benchmark)
        if g is not None:
            out.append(g)
        else:
            to_annotate.append(r)
    print(f"[gold:{benchmark}] manifest={len(rows)} reused={len(out)} to_annotate={len(to_annotate)}")
    a_path = GOLD_DIR / f"final_annot_a_{benchmark}.jsonl"
    b_path = GOLD_DIR / f"final_annot_b_{benchmark}.jsonl"
    adj_path = GOLD_DIR / f"final_adjudicated_{benchmark}.jsonl"
    if annotate and to_annotate:
        asyncio.run(annotate_rows(to_annotate, MODEL_AUDIT, a_path, ledger_state, f"{benchmark}", budget_cap))
        asyncio.run(annotate_rows(to_annotate, MODEL_AUDIT_B, b_path, ledger_state, f"{benchmark}", budget_cap))
        asyncio.run(adjudicate_pairs(to_annotate, a_path, b_path, adj_path, ledger_state, f"{benchmark}", budget_cap))
    # merge fresh adjudicated rows
    fresh = {json.loads(l)["id"]: json.loads(l) for l in adj_path.open(encoding="utf-8") if l.strip()} if adj_path.exists() else {}
    for r in to_annotate:
        f = fresh.get(r["sample_id"])
        if f is None:
            continue
        binary = int(f["binary"]) if f.get("binary") is not None else None
        if binary is None:
            continue
        out.append({
            "sample_id": r["sample_id"], "group_id": r["group_id"], "qy_hash": r["qy_hash"],
            "benchmark": benchmark, "gold_binary": binary, "gold_type": "adjudicated_silver",
            "four_type": str(f.get("four_type") or ("safe" if binary == 0 else "fraud_assistance")),
            "actionability": f.get("actionability"), "trust_facilitation": f.get("trust_facilitation"),
            "partial_leakage": f.get("partial_leakage"), "clean_refusal": f.get("clean_refusal"),
            "over_refusal_flag": f.get("over_refusal_flag"), "evidence_span": f.get("evidence_span"),
            "rationale": f.get("rationale", ""), "adjudicated": bool(f.get("adjudicated")),
            "audit_hash": audit_hash(benchmark, r["sample_id"]), "label_source": "blind_panel_final",
            "created_at": f.get("created_at", ""),
        })
    out.sort(key=lambda g: g["sample_id"])
    write_jsonl(GOLD_DIR / out_name, out)
    stats = {
        "manifest_n": len(rows),
        "reused": sum(1 for g in out if g["label_source"] == "prior_adjudicated_silver"),
        "fresh": sum(1 for g in out if g["label_source"] == "blind_panel_final"),
        "gold_positive": sum(1 for g in out if g["gold_binary"] == 1),
        "positive_rate": round(sum(1 for g in out if g["gold_binary"] == 1) / max(len(out), 1), 4),
        "agreement": agreement_stats(a_path, b_path) if a_path.exists() and b_path.exists() else {},
    }
    print(f"[gold:{benchmark}] {json.dumps(stats, ensure_ascii=False)}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true", help="run the blind panel for missing rows")
    ap.add_argument("--benchmark", choices=["fraudr1", "orbench"], default=None)
    args = ap.parse_args()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    ledger_state = load_gold_ledger()
    print(f"[gold] ledger used_rmb={ledger_state['used_rmb']:.4f} cap={ledger_state['cap_rmb']:.2f}")
    stats: dict[str, dict] = {}

    if args.benchmark in (None, "fraudr1"):
        stats["fraudr1"] = build_benchmark_gold(
            "fraudr1", "fraudr1_content_final_3000.jsonl", "fraudr1_content_adjudicated_3000.jsonl",
            ledger_state, budget_cap=PER_BENCH_CAP_RMB, annotate=args.annotate)
    if args.benchmark in (None, "orbench"):
        stats["orbench"] = build_benchmark_gold(
            "orbench", "orbench_final_3000.jsonl", "orbench_behavior_adjudicated_3000.jsonl",
            ledger_state, budget_cap=PER_BENCH_CAP_RMB, annotate=args.annotate)

    # Official gold (no cost)
    dna = read_jsonl(MANIFEST_DIR / "dna_final_5634.jsonl")
    dna_gold = [{
        "sample_id": r["sample_id"], "group_id": r["group_id"], "qy_hash": r["qy_hash"],
        "benchmark": "do_not_answer", "gold_binary": int(r["official_harmfulness"] == 1),
        "gold_type": "official", "target_model": r["target_model"], "prompt_id": r["prompt_id"],
    } for r in dna]
    write_jsonl(GOLD_DIR / "dna_official.jsonl", dna_gold)

    aegis_val = read_jsonl(MANIFEST_DIR / "aegis_validation.jsonl")
    aegis_test = read_jsonl(MANIFEST_DIR / "aegis_response_test_813.jsonl")
    aegis_gold = []
    for r in aegis_val + aegis_test:
        if r["evaluation_mode"] != "response":
            continue
        aegis_gold.append({
            "sample_id": r["sample_id"], "group_id": r["group_id"], "qy_hash": r["qy_hash"],
            "benchmark": "aegis2", "gold_binary": int(r["official_response_label"] == 1),
            "gold_type": "official", "split": r["split"],
            "violated_categories": r.get("violated_categories", ""),
            "label_source": r.get("label_source", ""),
        })
    write_jsonl(GOLD_DIR / "aegis_official.jsonl", aegis_gold)

    stats["dna"] = {"n": len(dna_gold), "positive": sum(1 for g in dna_gold if g["gold_binary"] == 1)}
    stats["aegis"] = {"n": len(aegis_gold),
                      "validation_response": sum(1 for g in aegis_gold if g["split"] == "validation"),
                      "test_response": sum(1 for g in aegis_gold if g["split"] == "test"),
                      "positive": sum(1 for g in aegis_gold if g["gold_binary"] == 1)}
    (GOLD_DIR / "gold_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()