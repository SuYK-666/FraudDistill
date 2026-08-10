# -*- coding: utf-8 -*-
"""v4 panel assembly: build new-gold tasks from generation results; assemble 6000-row panel."""

from __future__ import annotations

import collections
import json
import random
from typing import Any

from frauddistill.e1_final_v3.io import norm, read_json, read_jsonl, sha_text, write_json, write_jsonl
from frauddistill.e1_final_v4.gold import central_label, parse_vote
from frauddistill.e1_final_v4.panel import _gold_task, _row_for_gold, benign_topic_prompt, build_v32_gold_index, sd_pair_canonical


def _gen_index(results_path) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in read_jsonl(results_path):
        rid = r.get("response_id")
        text = (r.get("text") or "").strip()
        if r.get("status") == "ok" and rid and text:
            out[rid] = text
    return out


def _renderer(lang: str, cfg: dict[str, Any]) -> str:
    return cfg["e1_v4"]["q_renderer"][lang]


def _aegis_refusals(aegis_path) -> list[dict[str, Any]]:
    with open(aegis_path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for r in d:
        t = (r.get("response") or "").strip()
        if len(t) >= 30 and r.get("response_label") == "safe":
            out.append({"y": t, "prompt": r.get("prompt", "")})
    return out


def build_qbenign_tasks(cfg: dict[str, Any], out_dir, pilot_n: int | None = None) -> int:
    manifest = read_json(out_dir / "E1_V4_TASK_MANIFEST.json")
    gen_y = _gen_index(out_dir / "E1_V4_GEN_Y_RESULTS.jsonl")
    tasks = []
    pool = manifest["b1_q_pool"]
    if pilot_n is not None:
        n = min(pilot_n, len(pool))
        idxs = sorted({round(i * (len(pool) - 1) / max(1, n - 1)) for i in range(n)})
        pool = [pool[i] for i in idxs]
    for item in pool:
        rid = item["response_id"]
        if not gen_y.get(rid):
            continue
        m = cfg["models"]["gen_qwen"]
        tasks.append({
            "response_id": f"{rid}-QBENIGN",
            "task_kind": "b1_qbenign",
            "target_provider": m["provider"],
            "requested_target_model": m["model"],
            "extra_body": m.get("extra_body", {}),
            "q_private": benign_topic_prompt(item["q_private"], item["language"]),
            "temperature": cfg["generation"]["temperature"],
            "top_p": cfg["generation"]["top_p"],
            "max_tokens": 768,
            "timeout_seconds": cfg["generation"]["timeout_seconds"],
            "phase": "E1-v4-gen-qbenign",
            "status": "PENDING_API",
        })
    write_jsonl(out_dir / "E1_V4_GEN_QBENIGN_TASKS.jsonl", tasks)
    return len(tasks)


def build_gold_tasks(cfg: dict[str, Any], out_dir, pilot_n: int | None = None) -> dict[str, int]:
    manifest = read_json(out_dir / "E1_V4_TASK_MANIFEST.json")
    gen_y = _gen_index(out_dir / "E1_V4_GEN_Y_RESULTS.jsonl")
    gen_qb = _gen_index(out_dir / "E1_V4_GEN_QBENIGN_RESULTS.jsonl")
    tasks: list[dict[str, Any]] = []
    counts: dict[str, int] = collections.Counter()

    ref_by_q: dict[str, str] = {}
    for r in read_jsonl(out_dir / "E1_V4_GEN_REFUSAL_RESULTS.jsonl"):
        pq = r.get("pair_q")
        t = (r.get("text") or "").strip()
        if r.get("status") == "ok" and pq and t:
            ref_by_q[pq] = t

    # ---- B1
    b1_ready = []
    for i, item in enumerate(manifest["b1_q_pool"]):
        y = gen_y.get(item["response_id"])
        qb = gen_qb.get(item["response_id"] + "-QBENIGN")
        if not y or not qb:
            continue
        lang = item["language"]
        b1_ready.append({
            "idx": i, "lang": lang, "y": y,
            "q_scam": item["q_private"], "q_benign": _renderer(lang, cfg) + qb.strip(),
            "case": item["canonical_case_id"], "cat": item.get("fraud_category", ""),
        })
    if pilot_n is not None:
        b1_ready = b1_ready[:pilot_n]
    for it in b1_ready:
        for side, q, prov in [
            ("SCAM", it["q_scam"], "generated_y_counterfactual_qreal"),
            ("BENIGN", it["q_benign"], "generated_y_generated_q"),
        ]:
            row = _row_for_gold(
                f"E1-V4-B1-PAIR-{it['idx']:04d}-{side}", q, it["y"], it["lang"], it["case"],
                "b1_context_critical_y_matched", f"B1-{it['case']}", f"B1-{it['idx']:04d}", prov,
                {"fraud_category": it["cat"]})
            for judge in ["judge_a", "judge_b"]:
                tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b1"))
        counts["b1_pairs"] += 1

    # ---- B1 real matched pairs (priority 1): re-judged with v4 gold
    for rp in manifest.get("b1_real_pairs", []):
        row = _row_for_gold(
            f"E1-V4-REALPAIR-{rp['idx']:04d}-SCAM", rp["q_scam"], rp["y"], rp["language"], rp["case_scam"],
            "b1_context_critical_y_matched", f"B1-REAL-{rp['idx']:04d}", f"B1R-{rp['idx']:04d}", "real_matched_v32")
        for judge in ["judge_a", "judge_b"]:
            tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b1-real"))
        row = _row_for_gold(
            f"E1-V4-REALPAIR-{rp['idx']:04d}-BENIGN", rp["q_benign"], rp["y"], rp["language"], rp["case_benign"],
            "b1_context_critical_y_matched", f"B1-REAL-{rp['idx']:04d}", f"B1R-{rp['idx']:04d}", "real_matched_v32")
        for judge in ["judge_a", "judge_b"]:
            tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b1-real"))
        counts["b1_real_pairs"] += 1

    # ---- B2
    aegis = _aegis_refusals(cfg["data"]["aegis_refusals"])
    rng = random.Random(int(cfg["experiment"]["seed"]))
    rng.shuffle(aegis)
    aegis_iter = iter(aegis)
    b2_items = []
    for item in manifest["b2_qs"]:
        b2_items.append({"q": item["q_private"], "lang": item["language"], "case": item["canonical_case_id"], "pos_sd": True, "pos_rid": None})
    for item in manifest["b2_gen_qs"]:
        b2_items.append({"q": item["q_private"], "lang": item["language"], "case": item["canonical_case_id"], "pos_sd": False, "pos_rid": item["response_id"]})
    if pilot_n is not None:
        b2_items = b2_items[:pilot_n]
    for k, item in enumerate(b2_items):
        if item["lang"] == "zh":
            y_neg = ref_by_q.get(item["q"])
            prov_neg = "generated_refusal"
        else:
            a = next(aegis_iter, None)
            y_neg = a["y"] if a else None
            prov_neg = "aegis_refusal"
        if y_neg:
            row = _row_for_gold(
                f"E1-V4-B2-PAIR-{k:04d}-NEG", item["q"], y_neg, item["lang"], item["case"],
                "b2_response_critical_q_matched", f"B2-{item['case']}", f"B2-{k:04d}", prov_neg)
            for judge in ["judge_a", "judge_b"]:
                tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b2-neg"))
            counts["b2_neg"] += 1
        if not item["pos_sd"]:
            y_pos = gen_y.get(item["pos_rid"])
            if y_pos:
                row = _row_for_gold(
                    f"E1-V4-B2-PAIR-{k:04d}-POS", item["q"], y_pos, item["lang"], item["case"],
                    "b2_response_critical_q_matched", f"B2-{item['case']}", f"B2-{k:04d}", "generated_y")
                for judge in ["judge_a", "judge_b"]:
                    tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b2-pos"))
                counts["b2_pos_gen"] += 1

    # ---- B3 AEGIS (en) + zh refusals + generated pos
    for k, item in enumerate(manifest["b3_aegis_qs"]):
        a = next(aegis_iter, None)
        if not a:
            break
        row = _row_for_gold(
            f"E1-V4-B3-AEGIS-{k:04d}", item["q_private"], a["y"], "en", item["canonical_case_id"],
            "b3_context_stable_natural", item["canonical_case_id"], f"B3-{item['canonical_case_id']}", "aegis_refusal")
        for judge in ["judge_a", "judge_b"]:
            tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b3-aegis"))
        counts["b3_aegis"] += 1
    for k, item in enumerate(manifest["b3_zh_qs"]):
        y_neg = ref_by_q.get(item["q_private"])
        if not y_neg:
            continue
        row = _row_for_gold(
            f"E1-V4-B3-ZHREF-{k:04d}", item["q_private"], y_neg, "zh", item["canonical_case_id"],
            "b3_context_stable_natural", item["canonical_case_id"], f"B3-{item['canonical_case_id']}", "generated_refusal")
        for judge in ["judge_a", "judge_b"]:
            tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b3-zhref"))
        counts["b3_zhref"] += 1
    for k, item in enumerate(manifest["b3_gen_qs"]):
        y = gen_y.get(item["response_id"])
        if not y:
            continue
        row = _row_for_gold(
            f"E1-V4-B3-PAIR-{k:04d}-POS", item["q_private"], y, item["language"], item["canonical_case_id"],
            "b3_context_stable_natural", item["canonical_case_id"], f"B3-{item['canonical_case_id']}", "generated_y")
        for judge in ["judge_a", "judge_b"]:
            tasks.append(_gold_task(row, judge, cfg, "E1-v4-gold-b3-pos"))
        counts["b3_pos_gen"] += 1

    write_jsonl(out_dir / "E1_V4_GOLD_TASKS_NEW.jsonl", tasks)
    write_json(out_dir / "E1_V4_GOLD_TASK_COUNTS.json", dict(counts))
    return dict(counts)


# ---------------------------------------------------------------- panel assembly


def _vote_index(votes_path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(votes_path):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _adj_index(adj_path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(adj_path):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _y_by_rid(votes_path) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in read_jsonl(votes_path):
        rid = r.get("response_id")
        y = r.get("y_private")
        if not y:
            jp = r.get("judge_prompt") or ""
            marker = "[TARGET RESPONSE]\n"
            i = jp.find(marker)
            if i >= 0:
                y = jp[i + len(marker):]
                j = y.find("\n\nDEFINITIONS")
                if j >= 0:
                    y = y[:j]
                y = y.strip()
        if rid and y and rid not in out:
            out[rid] = y
    return out


def _qy_from_prompt(prompt: str) -> tuple[str, str]:
    q = y = ""
    if not prompt:
        return q, y
    qm = "[USER QUERY]\n"
    ym = "[TARGET RESPONSE]\n"
    dm = "\n\nDEFINITIONS"
    i = prompt.find(qm)
    if i >= 0:
        j = prompt.find(ym, i)
        if j >= 0:
            q = prompt[i + len(qm):j].strip()
            k = prompt.find(dm, j)
            y = prompt[j + len(ym):k if k >= 0 else None].strip() if k >= 0 else prompt[j + len(ym):].strip()
    return q, y


def build_content_vote_index(votes_path, adj_path) -> dict[str, dict[str, dict[str, Any]]]:
    """Content-keyed (q,y) vote index: lets identical (q,y) rows share gold votes."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for path in [votes_path, adj_path]:
        for r in read_jsonl(path):
            q = r.get("q_private") or ""
            y = r.get("y_private") or ""
            if not q or not y:
                q, y = _qy_from_prompt(r.get("judge_prompt") or "")
            v = parse_vote(r)
            if v is None or not q or not y:
                continue
            key = sha_text(norm(q) + "\x00" + norm(y))
            out.setdefault(key, {})[r.get("judge")] = v
    return out


def q_key_of(q: str) -> str:
    return sha_text(norm(q))


def resolved_from_vv(vv: dict[str, dict[str, Any]]) -> tuple[int | None, str]:
    a = central_label(vv.get("judge_a"))
    b = central_label(vv.get("judge_b"))
    if a is not None and a == b and not (vv.get("judge_a", {}).get("uncertain") or vv.get("judge_b", {}).get("uncertain")):
        return a, "double_agree"
    av = central_label(vv.get("adjudicator"))
    if av is not None:
        return av, "adjudicated"
    if a is not None and b is not None and a == b:
        return a, "double_agree_uncertain"
    return None, "unresolved"


def resolved_label(rid: str, votes: dict[str, dict[str, Any]], adj: dict[str, dict[str, Any]]) -> tuple[int | None, str]:
    vv = votes.get(rid) or {}
    a, b = central_label(vv.get("judge_a")), central_label(vv.get("judge_b"))
    if a is not None and a == b and not (vv.get("judge_a", {}).get("uncertain") or vv.get("judge_b", {}).get("uncertain")):
        return a, "double_agree"
    avv = adj.get(rid) or {}
    av = central_label(avv.get("adjudicator"))
    if av is not None:
        return av, "adjudicated"
    if a is not None and b is not None and a == b:
        return a, "double_agree_uncertain"
    return None, "unresolved"


def assemble_panel(cfg: dict[str, Any], out_dir) -> dict[str, Any]:
    manifest = read_json(out_dir / "E1_V4_TASK_MANIFEST.json")
    votes = _vote_index(out_dir / "E1_V4_GOLD_VOTES.jsonl")
    adj = _adj_index(out_dir / "E1_V4_GOLD_ADJUDICATION.jsonl")
    content_index = build_content_vote_index(out_dir / "E1_V4_GOLD_VOTES.jsonl", out_dir / "E1_V4_GOLD_ADJUDICATION.jsonl")
    q_index: dict[str, dict[str, dict[str, Any]]] = {}
    for r in read_jsonl(out_dir / "E1_V4_GOLD_VOTES.jsonl"):
        q = r.get("q_private") or ""
        if not q:
            q, _ = _qy_from_prompt(r.get("judge_prompt") or "")
        v = parse_vote(r)
        if v is None or not q:
            continue
        q_index.setdefault(q_key_of(q), {})[r.get("judge")] = v
    for r in read_jsonl(out_dir / "E1_V4_GOLD_ADJUDICATION.jsonl"):
        q = r.get("q_private") or ""
        if not q:
            q, _ = _qy_from_prompt(r.get("judge_prompt") or "")
        v = parse_vote(r)
        if v is None or not q:
            continue
        q_index.setdefault(q_key_of(q), {})[r.get("judge")] = v
    gen_y = _gen_index(out_dir / "E1_V4_GEN_Y_RESULTS.jsonl")
    gen_qb = _gen_index(out_dir / "E1_V4_GEN_QBENIGN_RESULTS.jsonl")
    y_by_rid = _y_by_rid(out_dir / "E1_V4_GOLD_VOTES.jsonl")

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    def add(rid, q, y, lang, case, stratum, family, pair, prov, label, method, extra=None):
        row = {
            "response_id": rid, "q_private": q, "y_private": y, "language": lang,
            "canonical_case_id": case, "stratum": stratum, "family_id": family, "pair_id": pair,
            "provenance": prov, "gold_central": label, "gold_method": method,
            "gold_status": "KNOWN", "source_dataset": "E1-FINAL-TRIAD-v4",
        }
        if extra:
            row.update(extra)
        rows.append(row)

    # ---- B1
    b1_kept = 0
    # real matched pairs first (priority 1)
    for rp in manifest.get("b1_real_pairs", []):
        if b1_kept >= 1000:
            break
        scam_lab, scam_m = resolved_label(f"E1-V4-REALPAIR-{rp['idx']:04d}-SCAM", votes, adj)
        ben_lab, ben_m = resolved_label(f"E1-V4-REALPAIR-{rp['idx']:04d}-BENIGN", votes, adj)
        if scam_lab == 1 and ben_lab == 0:
            pair = f"B1R-{rp['idx']:04d}"
            add(f"E1-V4-REALPAIR-{rp['idx']:04d}-SCAM", rp["q_scam"], rp["y"], rp["language"], rp["case_scam"],
                "b1_context_critical_y_matched", f"B1-REAL-{rp['idx']:04d}", pair, "real_matched_v32", 1, scam_m)
            add(f"E1-V4-REALPAIR-{rp['idx']:04d}-BENIGN", rp["q_benign"], rp["y"], rp["language"], rp["case_benign"],
                "b1_context_critical_y_matched", f"B1-REAL-{rp['idx']:04d}", pair, "real_matched_v32", 0, ben_m)
            b1_kept += 1
        else:
            dropped.append({"kind": "b1_real", "idx": rp["idx"], "scam": scam_lab, "benign": ben_lab})

    for i, item in enumerate(manifest["b1_q_pool"]):
        if b1_kept >= 1000:
            break
        y = gen_y.get(item["response_id"])
        if not y:
            continue
        scam_lab, scam_m = resolved_label(f"E1-V4-B1-PAIR-{i:04d}-SCAM", votes, adj)
        ben_lab, ben_m = resolved_label(f"E1-V4-B1-PAIR-{i:04d}-BENIGN", votes, adj)
        if scam_lab == 1 and ben_lab == 0:
            case = item["canonical_case_id"]
            pair = f"B1-{i:04d}"
            qb = gen_qb.get(item["response_id"] + "-QBENIGN")
            q_benign = _renderer(item["language"], cfg) + qb.strip() if qb else ""
            add(f"E1-V4-B1-PAIR-{i:04d}-SCAM", item["q_private"], y, item["language"], case,
                "b1_context_critical_y_matched", f"B1-{case}", pair, "generated_y_counterfactual_qreal", 1, scam_m,
                {"fraud_category": item.get("fraud_category", "")})
            add(f"E1-V4-B1-PAIR-{i:04d}-BENIGN", q_benign, y, item["language"], case,
                "b1_context_critical_y_matched", f"B1-{case}", pair, "generated_y_generated_q", 0, ben_m,
                {"fraud_category": item.get("fraud_category", "")})
            b1_kept += 1
        else:
            dropped.append({"kind": "b1", "idx": i, "scam": scam_lab, "benign": ben_lab})

    # ---- B2
    # replicate build_gold_tasks refusal assignment (zh: generated refusal; en: aegis shuffle)
    ref_by_q: dict[str, str] = {}
    for r in read_jsonl(out_dir / "E1_V4_GEN_REFUSAL_RESULTS.jsonl"):
        pq = r.get("pair_q")
        t = (r.get("text") or "").strip()
        if r.get("status") == "ok" and pq and t:
            ref_by_q[pq] = t
    aegis_list = _aegis_refusals(cfg["data"]["aegis_refusals"])
    rng2 = random.Random(int(cfg["experiment"]["seed"]))
    rng2.shuffle(aegis_list)
    aegis_iter2 = iter(aegis_list)
    b2_neg_y_by_k: dict[int, str] = {}
    for k, item in enumerate(manifest["b2_qs"] + manifest["b2_gen_qs"]):
        if item["language"] == "zh":
            y_neg = ref_by_q.get(item["q_private"])
        else:
            a = next(aegis_iter2, None)
            y_neg = a["y"] if a else None
        if y_neg:
            b2_neg_y_by_k[k] = y_neg
    b2_kept = 0
    for k, item in enumerate(manifest["b2_qs"] + manifest["b2_gen_qs"]):
        if b2_kept >= 1000:
            break
        q = item["q_private"]
        case = item["canonical_case_id"]
        pair = f"B2-{k:04d}"
        if item.get("from_sd"):
            pos_lab, pos_m = 1, "reused_v32_gold"
            pos_y, pos_rid = item["y_private"], f"SD-{item['content_key'][:16]}"
        else:
            pos_lab, pos_m = resolved_label(f"E1-V4-B2-PAIR-{k:04d}-POS", votes, adj)
            pos_y = gen_y.get(item["response_id"])
            pos_rid = f"E1-V4-B2-PAIR-{k:04d}-POS"
        neg_rid = f"E1-V4-B2-PAIR-{k:04d}-NEG"
        y_neg = y_by_rid.get(neg_rid)
        if neg_rid in votes:
            neg_lab, neg_m = resolved_label(neg_rid, votes, adj)
        elif y_neg:
            neg_lab, neg_m = resolved_from_vv(content_index.get(sha_text(norm(q) + "\x00" + norm(y_neg))) or {})
        else:
            # recover the intended refusal y exactly as build_gold_tasks assigns it
            y_neg = b2_neg_y_by_k.get(k)
            if y_neg:
                neg_lab, neg_m = resolved_from_vv(content_index.get(sha_text(norm(q) + "\x00" + norm(y_neg))) or {})
            else:
                neg_lab, neg_m = None, "no_votes"
        if pos_lab == 1 and neg_lab == 0 and pos_y and y_neg:
            add(pos_rid, q, pos_y, item["language"], case, "b2_response_critical_q_matched", f"B2-{case}", pair,
                "source_derived_open_control" if item.get("from_sd") else "generated_y", 1, pos_m)
            add(f"E1-V4-B2-PAIR-{k:04d}-NEG", q, y_neg, item["language"], case, "b2_response_critical_q_matched", f"B2-{case}", pair,
                "aegis_refusal" if item["language"] != "zh" else "generated_refusal", 0, neg_m)
            b2_kept += 1
        else:
            dropped.append({"kind": "b2", "k": k, "pos": pos_lab, "neg": neg_lab})

    # ---- B3 pos
    b3_pos = 0
    for it in manifest["b3_sd_pos"]:
        if b3_pos >= 1000:
            break
        add(f"SD-{it['content_key'][:16]}", it["q_private"], it["y_private"], it["language"], it["canonical_case_id"],
            "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "source_derived_open_control", 1, "reused_v32_gold")
        b3_pos += 1
    for k, it in enumerate(manifest["salvage"]):
        if b3_pos >= 1000:
            break
        lab, m = resolved_label(f"E1-V4-SALVAGE-{k:04d}", votes, adj)
        if lab == 1:
            add(f"E1-V4-SALVAGE-{k:04d}", it["q_private"], it["y_private"], it["language"], it["canonical_case_id"],
                "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "source_derived_open_control", 1, m,
                {"salvage_status": it["label_status"]})
            b3_pos += 1
    for k, it in enumerate(manifest["b3_gen_qs"]):
        if b3_pos >= 1000:
            break
        y = gen_y.get(it["response_id"])
        lab, m = resolved_label(f"E1-V4-B3-PAIR-{k:04d}-POS", votes, adj)
        if lab == 1 and y:
            add(f"E1-V4-B3-PAIR-{k:04d}-POS", it["q_private"], y, it["language"], it["canonical_case_id"],
                "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "generated_y", 1, m)
            b3_pos += 1

    # ---- B3 neg
    b3_neg = 0
    for it in manifest["b3_sd_neg"]:
        if b3_neg >= 1000:
            break
        add(f"SD-{it['content_key'][:16]}", it["q_private"], it["y_private"], it["language"], it["canonical_case_id"],
            "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "source_derived_open_control", 0, "reused_v32_gold")
        b3_neg += 1
    for k, it in enumerate(manifest["salvage"]):
        if b3_neg >= 1000:
            break
        lab, m = resolved_label(f"E1-V4-SALVAGE-{k:04d}", votes, adj)
        if lab == 0:
            add(f"E1-V4-SALVAGE-{k:04d}", it["q_private"], it["y_private"], it["language"], it["canonical_case_id"],
                "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "source_derived_open_control", 0, m,
                {"salvage_status": it["label_status"]})
            b3_neg += 1
    for k, it in enumerate(manifest["b3_aegis_qs"]):
        if b3_neg >= 1000:
            break
        aegis_rid = f"E1-V4-B3-AEGIS-{k:04d}"
        y_neg = y_by_rid.get(aegis_rid)
        if aegis_rid in votes:
            lab, m = resolved_label(aegis_rid, votes, adj)
        elif y_neg:
            lab, m = resolved_from_vv(content_index.get(sha_text(norm(it["q_private"]) + "\x00" + norm(y_neg))) or {})
        elif q_index.get(q_key_of(it["q_private"])):
            lab, m = resolved_from_vv(q_index.get(q_key_of(it["q_private"])) or {})
        else:
            lab, m = None, "no_votes"
        if lab == 0 and y_neg:
            add(f"E1-V4-B3-AEGIS-{k:04d}", it["q_private"], y_neg, "en", it["canonical_case_id"],
                "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "aegis_refusal", 0, m)
            b3_neg += 1
    for k, it in enumerate(manifest["b3_zh_qs"]):
        if b3_neg >= 1000:
            break
        zhref_rid = f"E1-V4-B3-ZHREF-{k:04d}"
        y_neg = y_by_rid.get(zhref_rid)
        if zhref_rid in votes:
            lab, m = resolved_label(zhref_rid, votes, adj)
        elif y_neg:
            lab, m = resolved_from_vv(content_index.get(sha_text(norm(it["q_private"]) + "\x00" + norm(y_neg))) or {})
        elif q_index.get(q_key_of(it["q_private"])):
            lab, m = resolved_from_vv(q_index.get(q_key_of(it["q_private"])) or {})
        else:
            lab, m = None, "no_votes"
        if lab == 0 and y_neg:
            add(f"E1-V4-B3-ZHREF-{k:04d}", it["q_private"], y_neg, "zh", it["canonical_case_id"],
                "b3_context_stable_natural", it["canonical_case_id"], f"B3-{it['canonical_case_id']}", "generated_refusal", 0, m)
            b3_neg += 1

    counts = collections.Counter(r["stratum"] for r in rows)
    audit = {
        "n_rows": len(rows),
        "by_stratum": dict(counts),
        "by_label": dict(collections.Counter(r["gold_central"] for r in rows)),
        "by_language": dict(collections.Counter(r["language"] for r in rows)),
        "by_provenance": dict(collections.Counter(r["provenance"] for r in rows)),
        "b1_kept": b1_kept, "b2_kept": b2_kept, "b3_pos": b3_pos, "b3_neg": b3_neg,
        "dropped_preview": dropped[:50],
        "n_dropped": len(dropped),
    }
    write_jsonl(out_dir / "E1_V4_PANEL_ALL.jsonl", rows)
    write_json(out_dir / "E1_V4_PANEL_AUDIT.json", audit)
    return audit
