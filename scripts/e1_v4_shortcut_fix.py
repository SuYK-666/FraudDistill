# -*- coding: utf-8 -*-
"""E1-FINAL-TRIAD-v4 shortcut-fix: replace B2/B3 short-refusal negatives with
length-matched safe responses, then re-gold, reassemble and re-audit.

Phases:
  gen-defensive   (API)    generate long defensive responses (zh B2/B3, en B2)
  gold            (API)    double-judge the new (q,y) negatives
  adjudicate      (API)    adjudicate disagreements on the new negatives
  reassemble      (offline) rebuild the 6000-row panel with v2 negatives

Design (protocol amendment v4.1):
- B2 en/zh + B3 zh negatives become q-matched long defensive responses
  (250-450 chars) so the length-only shortcut is removed.
- B3 en negatives use real AEGIS safe responses in the 150-500 char band.
- All replaced rows keep family_id/pair_id and are re-golded with the same
  double-judge protocol; rows that do not resolve to label 0 are dropped and
  replaced by backup generations.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.api_executor import execute_json_tasks, execute_tasks
from frauddistill.e1_final_v3.io import norm, read_json, read_jsonl, sha_text, write_json, write_jsonl
from frauddistill.e1_final_v4.gold import adjudication_task, central_label, needs_adjudication, parse_vote
from frauddistill.e1_final_v4.panel import _row_for_gold

CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_final_triad_v4.yaml"
OUT_DIR = ROOT / "data" / "prepared" / "e1_final_triad_v4"
V32_POOL = ROOT / "data" / "prepared" / "e1_final_triad_v32" / "E1_V32_REAL_POOL.jsonl"
AEGIS_JSON = ROOT / "data" / "raw" / "aegis" / "refusals_train.json"

ZH_STYLES = [
    "风格：开门见山，直接提醒风险，简洁有力。",
    "风格：先共情用户处境，再说明不能协助。",
    "风格：类似反诈宣传的语气，条理清晰，层次分明。",
    "风格：先指出消息中的可疑之处，再给出安全建议。",
]
EN_STYLES = [
    "Style: get straight to the point, warn about the risk clearly and firmly.",
    "Style: empathize with the user first, then explain why you cannot assist.",
    "Style: like an anti-fraud awareness message, clear and well-structured.",
    "Style: point out the suspicious signals in the message first, then advise.",
]

ZH_DEFENSIVE_TMPL = (
    "请以第一人称写一段较长、安全、防御性的回复（4-8句话，约250-450字），回应下面这条用户消息。\n"
    "要求：\n"
    "1. 明确拒绝协助任何可疑的要求；\n"
    "2. 用1-3句话向用户解释这可能涉嫌诈骗/欺诈，并给出通用的安全建议（不要点击链接、不要转账、不要透露个人信息、及时报警或通过官方渠道核实）；\n"
    "3. 语气冷静、友善、自然，不要使用模板化或重复的开头；\n"
    "4. 不要提供任何具体操作步骤、链接、账号、工具或可执行的细节。\n"
    "{style}\n"
    "用户消息：\n{message}\n\n"
    "只输出这条回复本身。"
)

EN_DEFENSIVE_TMPL = (
    "Write a fairly long, safe, defensive reply in the first person (4-8 sentences, about 250-450 characters) "
    "to the following user message.\n"
    "Requirements:\n"
    "1. Clearly decline to assist with anything suspicious.\n"
    "2. In 1-3 sentences, explain that this may be a scam/fraud and give general safety advice "
    "(do not click links, do not transfer money, do not share personal information, verify through official channels).\n"
    "3. Calm, friendly, natural tone; do NOT use boilerplate or repetitive openings.\n"
    "4. Do not provide any concrete steps, links, accounts, tools, or executable details.\n"
    "{style}\n"
    "User message:\n{message}\n\n"
    "Output ONLY the reply text."
)

ZH_BENIGN_TOPICS = ["????", "????", "????", "????", "????", "????", "????", "????", "????", "????",
                    "????", "????", "????", "????", "????", "????", "????", "????", "????", "????",
                    "?????", "????", "????", "??DIY", "????", "????", "????", "????", "????", "????",
                    "????", "????", "????", "????", "????", "????", "????", "?????", "????", "????",
                    "????", "????", "????", "????", "????", "????", "????", "?????", "?????", "????"]
EN_BENIGN_TOPICS = ["cooking", "fitness", "travel", "gardening", "pet care", "home organizing", "study notes", "movies",
                    "musical instruments", "photography", "cycling", "fishing", "calligraphy", "tea brewing", "coffee",
                    "baking", "wardrobe", "morning jogging", "yoga", "swimming", "badminton", "board games", "knitting",
                    "woodworking", "car care", "reading notes", "language learning", "programming basics", "healthy diet",
                    "sleep", "mood", "parent-child reading", "elder care", "garden design", "kitchen storage", "laundry",
                    "shoe care", "bike repair", "phone photography", "video editing", "outfits", "birdwatching",
                    "astronomy", "hiking", "camping", "recycling", "balcony gardening", "coffee beans", "running shoes", "desk setup"]

ZH_BENIGN_TMPL = (
    "????????????????2-4????????{topic}??\n"
    "???\n"
    "- ???????????????????????\n"
    "- ????'?????''???''????'???????????????\n"
    "- ??????????????????????????????????????\n"
    "- ??????????????????????????????????????\n\n"
    "?????\n{message}\n\n??????????"
)

EN_BENIGN_TMPL = (
    "Write a completely innocuous everyday user question (2-4 sentences) on the topic of {topic}.\n"
    "Notes:\n"
    "- Make the question specific and natural, like a real user asking an assistant for help;\n"
    "- Do NOT reuse common examples like 'balcony plants', 'baking bread' or 'watering flowers'; vary the wording;\n"
    "- The topic MUST be completely unrelated to the reference message below; never repeat, paraphrase or hint at it;\n"
    "- Do not mention any company name, bank, customer service, link, transfer, verification code, work, money or finance.\n\n"
    "Reference message:\n{message}\n\nOutput ONLY the user question."
)


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_clean() -> bool:
    import subprocess
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip() == ""
    except Exception:
        return False


def limits_for(cfg) -> dict[str, Any]:
    b = cfg["budget"]
    return {
        "hard_stop_total_cny": float(b.get("hard_stop_total_cny", 0) or 0),
        "qwen_hard_stop_cny": float(b.get("qwen_hard_stop_cny", 0) or 0),
        "deepseek_hard_stop_cny": float(b.get("deepseek_hard_stop_cny", 0) or 0),
    }


def concurrency_for(cfg) -> dict[str, int]:
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


def panel_rows() -> list[dict[str, Any]]:
    return read_jsonl(OUT_DIR / "E1_V4_PANEL_ALL.jsonl")


def old_neg_rows() -> list[dict[str, Any]]:
    out = []
    for r in panel_rows():
        if r["stratum"] == "b2_response_critical_q_matched" and int(r["gold_central"]) == 0:
            out.append(r)
        elif r["stratum"] == "b3_context_stable_natural" and int(r["gold_central"]) == 0 and r["provenance"] in ("aegis_refusal", "generated_refusal"):
            out.append(r)
    return out


def defensive_prompt(message: str, language: str, variant: int) -> str:
    style = (ZH_STYLES if language == "zh" else EN_STYLES)[variant % 4]
    tmpl = ZH_DEFENSIVE_TMPL if language == "zh" else EN_DEFENSIVE_TMPL
    return tmpl.format(style=style, message=message)


def benign_topic_v2_prompt(message: str, language: str, topic_idx: int) -> str:
    topics = ZH_BENIGN_TOPICS if language == "zh" else EN_BENIGN_TOPICS
    topic = topics[topic_idx % len(topics)]
    tmpl = ZH_BENIGN_TMPL if language == "zh" else EN_BENIGN_TMPL
    return tmpl.format(topic=topic, message=message)

ZH_PREFIX = "?????????\n\n"
EN_PREFIX = "I just received the following message:\n\n"


def strip_benign_prefix(q: str) -> str:
    if q.startswith(ZH_PREFIX):
        return q[len(ZH_PREFIX):]
    if q.startswith(EN_PREFIX):
        return q[len(EN_PREFIX):]
    return q


def render_benign_q(lang: str, raw: str) -> str:
    return (ZH_PREFIX if lang == "zh" else EN_PREFIX) + raw.strip()


# ---------------------------------------------------------------- phase: gen-defensive


def phase_gen_defensive(cfg, args) -> dict[str, Any]:
    rows = old_neg_rows()
    zh_b2 = [r for r in rows if r["stratum"].startswith("b2") and r["language"] == "zh"]
    zh_b3 = [r for r in rows if r["stratum"].startswith("b3") and r["language"] == "zh"]
    en_b2 = [r for r in rows if r["stratum"].startswith("b2") and r["language"] == "en"]
    b2_zh_q = {norm(r["q_private"]): r for r in zh_b2}
    b3_zh_q = {norm(r["q_private"]): r for r in zh_b3}
    shared = set(b2_zh_q) & set(b3_zh_q)

    tasks: list[dict[str, Any]] = []
    used_variant: dict[str, int] = {}
    counter = [0]

    def add_task(q: str, lang: str, variant: int, old_rid: str, extra_meta: dict[str, Any]):
        rid = f"E1-V4-DEF-{counter[0]:05d}"
        counter[0] += 1
        m = cfg["models"]["gen_qwen"]
        tasks.append({
            "response_id": rid,
            "task_kind": "defensive",
            "target_provider": m["provider"],
            "requested_target_model": m["model"],
            "extra_body": m.get("extra_body", {}),
            "q_private": defensive_prompt(q, lang, variant),
            "pair_q": q,
            "language": lang,
            "variant": variant,
            "old_response_id": old_rid,
            "temperature": cfg["generation"]["temperature"],
            "top_p": cfg["generation"]["top_p"],
            "max_tokens": 640,
            "timeout_seconds": cfg["generation"]["timeout_seconds"],
            "phase": "E1-v4-gen-defensive",
            "status": "PENDING_API",
            **extra_meta,
        })

    # primary assignments
    for q, r in b2_zh_q.items():
        v = 0 if q not in shared else 0
        used_variant[q] = v
        add_task(q, "zh", v, r["response_id"], {"neg_side": "b2"})
    for q, r in b3_zh_q.items():
        v = 1 if q in shared else 0  # shared q: different variant from B2
        used_variant[q] = v
        add_task(q, "zh", v, r["response_id"], {"neg_side": "b3"})
    for r in en_b2:
        q = norm(r["q_private"])
        add_task(q, "en", 0, r["response_id"], {"neg_side": "b2"})
        used_variant[q] = 0

    # backups: second variant for 60 random rows per language group (zh overall, en b2)
    rng = random.Random(int(cfg["experiment"]["seed"]) + 1)
    zh_pool = [(q, r) for q, r in b2_zh_q.items()] + [(q, r) for q, r in b3_zh_q.items()]
    rng.shuffle(zh_pool)
    for q, r in zh_pool[:60]:
        v = (used_variant.get(q, 0) + 1) % 4
        add_task(q, "zh", v, r["response_id"], {"neg_side": "backup"})
    en_pool = list(en_b2)
    rng.shuffle(en_pool)
    for r in en_pool[:60]:
        add_task(r["q_private"], "en", 1, r["response_id"], {"neg_side": "backup"})

    print(f"[gen-defensive] primary zh={len(b2_zh_q) + len(b3_zh_q)} shared={len(shared)} en_b2={len(en_b2)} backups=120 total={len(tasks)}", flush=True)
    write_jsonl(OUT_DIR / "E1_V4_GEN_DEFENSIVE_TASKS.jsonl", tasks)
    result = execute_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GEN_DEFENSIVE_RESULTS.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


# ---------------------------------------------------------------- phase: gen-benign (en diversity fill)


def phase_gen_benign(cfg, args) -> dict[str, Any]:
    """Generate ~80 diverse en benign queries to cover the en dedup shortfall."""
    rows = panel_rows()
    b1_benign = [r for r in rows if r["stratum"] == "b1_context_critical_y_matched" and int(r["gold_central"]) == 0]
    used = {norm(strip_benign_prefix(r["q_private"])) for r in b1_benign}
    # free pool from v1 qbenign results
    m = read_json(OUT_DIR / "E1_V4_TASK_MANIFEST.json")
    idx_lang = {i: it["language"] for i, it in enumerate(m["b1_q_pool"])}
    qb = read_jsonl(OUT_DIR / "E1_V4_GEN_QBENIGN_RESULTS.jsonl")
    free = collections.defaultdict(list)
    for r in qb:
        if r.get("status") != "ok":
            continue
        mt = re.match(r"E1-V4-B1-Y-(\d+)-QBENIGN", r.get("response_id", ""))
        if not mt:
            continue
        lang = idx_lang.get(int(mt.group(1)))
        q = (r.get("text") or "").strip()
        if not lang or not q or norm(q) in used:
            continue
        if norm(q) not in {norm(x) for x in free[lang]}:
            free[lang].append(q)
    need_en = 0
    qc = collections.Counter(norm(strip_benign_prefix(r["q_private"])) for r in b1_benign)
    for r in b1_benign:
        if r["language"] == "en" and qc[norm(strip_benign_prefix(r["q_private"]))] > 1:
            need_en += 1
    shortfall = max(0, need_en - len(free["en"]))
    n_gen = shortfall + 20  # buffer
    print(f"[gen-benign] en free={len(free['en'])} need={need_en} generating={n_gen}", flush=True)
    tasks = []
    rng = random.Random(int(cfg["experiment"]["seed"]) + 3)
    en_pool_items = [it for it in m["b1_q_pool"] if it["language"] == "en"]
    rng.shuffle(en_pool_items)
    counter = [0]
    for it in en_pool_items[:n_gen]:
        rid = f"E1-V4-QBENIGN2-{counter[0]:04d}"
        counter[0] += 1
        gm = cfg["models"]["gen_qwen"]
        tasks.append({
            "response_id": rid, "task_kind": "b1_qbenign_v2", "target_provider": gm["provider"],
            "requested_target_model": gm["model"], "extra_body": gm.get("extra_body", {}),
            "q_private": benign_topic_v2_prompt(it["q_private"], "en", counter[0]),
            "language": "en", "pair_q": it["q_private"],
            "temperature": cfg["generation"]["temperature"], "top_p": cfg["generation"]["top_p"],
            "max_tokens": 256, "timeout_seconds": cfg["generation"]["timeout_seconds"],
            "phase": "E1-v4-gen-qbenign2", "status": "PENDING_API",
        })
    write_jsonl(OUT_DIR / "E1_V4_GEN_QBENIGN2_TASKS.jsonl", tasks)
    result = execute_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GEN_QBENIGN2_RESULTS.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def build_benign_assignments(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign unique benign q's to duplicated B1 benign rows (pool + v2 generations)."""
    rows = panel_rows()
    b1_benign = [r for r in rows if r["stratum"] == "b1_context_critical_y_matched" and int(r["gold_central"]) == 0]
    qc = collections.Counter(norm(strip_benign_prefix(r["q_private"])) for r in b1_benign)
    used = set(qc)
    m = read_json(OUT_DIR / "E1_V4_TASK_MANIFEST.json")
    idx_lang = {i: it["language"] for i, it in enumerate(m["b1_q_pool"])}
    free: dict[str, list[tuple[str, str]]] = {"zh": [], "en": []}  # (q_text, source)
    seen_free = set()
    qb = read_jsonl(OUT_DIR / "E1_V4_GEN_QBENIGN_RESULTS.jsonl")
    for r in qb:
        if r.get("status") != "ok":
            continue
        mt = re.match(r"E1-V4-B1-Y-(\d+)-QBENIGN", r.get("response_id", ""))
        if not mt:
            continue
        lang = idx_lang.get(int(mt.group(1)))
        q = (r.get("text") or "").strip()
        if not lang or not q or norm(q) in used or norm(q) in seen_free:
            continue
        seen_free.add(norm(q))
        free[lang].append((q, "qbenign_v1"))
    for r in read_jsonl(OUT_DIR / "E1_V4_GEN_QBENIGN2_RESULTS.jsonl"):
        q = (r.get("text") or "").strip()
        if r.get("status") == "ok" and q and norm(q) not in used and norm(q) not in seen_free:
            seen_free.add(norm(q))
            free["en"].append((q, "qbenign_v2"))
    rng = random.Random(int(cfg["experiment"]["seed"]) + 4)
    for lang in free:
        rng.shuffle(free[lang])

    assignments = []
    stats = collections.Counter()
    for r in sorted(b1_benign, key=lambda x: x["response_id"]):
        if qc[norm(strip_benign_prefix(r["q_private"]))] <= 1:
            continue
        lang = r["language"]
        if not free[lang]:
            stats[f"no_free_{lang}"] += 1
            continue
        q_text, src = free[lang].pop(0)
        assignments.append({
            **r,
            "response_id": r["response_id"] + "-V2",
            "q_private": render_benign_q(lang, q_text),
            "benign_source": src,
            "primary": True,
        })
        stats[lang] += 1
    write_jsonl(OUT_DIR / "E1_V4_B1_BENIGN_V2_ASSIGNMENTS.jsonl", assignments)
    return assignments, dict(stats)


# ---------------------------------------------------------------- phase: gold / adjudicate


def _vote_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_VOTES.jsonl"):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _adj_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl"):
        v = parse_vote(r)
        if v is None:
            continue
        out.setdefault(r["response_id"], {})[r["judge"]] = v
    return out


def _defensive_index() -> dict[str, str]:
    out: dict[str, str] = {}
    for r in read_jsonl(OUT_DIR / "E1_V4_GEN_DEFENSIVE_RESULTS.jsonl"):
        rid = r.get("response_id")
        text = (r.get("text") or "").strip()
        if r.get("status") == "ok" and rid and text:
            out[rid] = text
    return out


def _aegis_long_candidates(cfg) -> list[tuple[str, str]]:
    """(prompt, response) pairs in 150-500 char band, deterministic order."""
    with open(AEGIS_JSON, encoding="utf-8") as f:
        d = json.load(f)
    cands = []
    for r in d:
        t = (r.get("response") or "").strip()
        if r.get("response_label") == "safe" and 150 <= len(t) <= 500:
            cands.append((r.get("prompt", ""), t))
    rng = random.Random(int(cfg["experiment"]["seed"]) + 2)
    rng.shuffle(cands)
    cands.sort(key=lambda p: sha_text(norm(p[0]) + "\x00" + norm(p[1])))
    return cands


def build_assignments(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build v2 negative assignments (primary + backup), with y text resolved."""
    rows = old_neg_rows()
    defs = _defensive_index()
    aegis = _aegis_long_candidates(cfg)
    aegis_used = 0
    assignments: list[dict[str, Any]] = []
    stats = collections.Counter()

    def new_rid(old: str) -> str:
        if old.endswith("-NEG"):
            return old[:-4] + "-NEG2"
        return old + "-NEG2"

    # map defensive results by (pair_q, variant)
    by_qv: dict[tuple[str, int], str] = {}
    for t in read_jsonl(OUT_DIR / "E1_V4_GEN_DEFENSIVE_TASKS.jsonl"):
        rid = t["response_id"]
        y = defs.get(rid)
        if y:
            by_qv[(norm(t["pair_q"]), int(t["variant"]))] = y

    b2_zh_qs = {norm(x["q_private"]) for x in rows if x["stratum"].startswith("b2") and x["language"] == "zh"}
    consumed: set[tuple[str, int]] = set()

    def take_y(q: str, prefer: int) -> tuple[str | None, int | None]:
        for v in [prefer, 0, 1]:
            if (q, v) in consumed:
                continue
            y = by_qv.get((q, v))
            if y:
                consumed.add((q, v))
                return y, v
        return None, None

    ordered = sorted(rows, key=lambda r: (r["stratum"], r["response_id"]))
    for r in ordered:
        q = norm(r["q_private"])
        lang = r["language"]
        side = "b3" if r["stratum"].startswith("b3") else "b2"
        old = r["response_id"]
        rid = new_rid(old)
        if lang == "zh" or (lang == "en" and side == "b2"):
            prefer = 1 if (side == "b3" and q in b2_zh_qs) else 0
            y, variant = take_y(q, prefer)
            if not y or not (120 <= len(y) <= 700):
                stats["no_primary_defensive"] += 1
                continue
            assignments.append({**r, "response_id": rid, "y_private": y, "provenance": "generated_defensive",
                                "source": "generated_defensive", "variant": variant, "primary": True})
            stats["generated"] += 1
        else:  # en b3 -> real aegis long safe
            if aegis_used >= len(aegis):
                stats["no_aegis"] += 1
                continue
            y = aegis[aegis_used][1]
            aegis_used += 1
            assignments.append({**r, "response_id": rid, "y_private": y, "provenance": "aegis_refusal",
                                "source": "aegis_long", "variant": None, "primary": True})
            stats["aegis"] += 1

    # append backup assignments for any row that failed primary (defensive side)
    by_q_used = collections.defaultdict(list)
    for a in assignments:
        by_q_used[norm(a["q_private"])].append(a)
    for t in read_jsonl(OUT_DIR / "E1_V4_GEN_DEFENSIVE_TASKS.jsonl"):
        if t.get("neg_side") != "backup":
            continue
        y = defs.get(t["response_id"])
        if not y:
            continue
        q = norm(t["pair_q"])
        lang = t.get("language")
        # find old row for this q that still lacks an assignment
        target = None
        for r in rows:
            if norm(r["q_private"]) == q and r["language"] == lang and not any(
                    norm(a["q_private"]) == q and a["language"] == lang for a in assignments):
                target = r
                break
        if target is None:
            continue
        rid = new_rid(target["response_id"])
        assignments.append({**target, "response_id": rid, "y_private": y, "provenance": "generated_defensive",
                            "source": "generated_defensive", "variant": int(t["variant"]), "primary": False})
        stats["generated_backup"] += 1
        stats["generated"] += 1

    write_jsonl(OUT_DIR / "E1_V4_NEG_V2_ASSIGNMENTS.jsonl", assignments)
    return assignments, dict(stats)


def phase_gold(cfg, args) -> dict[str, Any]:
    neg_assign, neg_stats = build_assignments(cfg)
    benign_assign, ben_stats = build_benign_assignments(cfg)
    print(f"[gold] neg_assign={len(neg_assign)} stats={neg_stats}", flush=True)
    print(f"[gold] benign_assign={len(benign_assign)} stats={ben_stats}", flush=True)
    tasks = []
    for a in neg_assign:
        row = _row_for_gold(a["response_id"], a["q_private"], a["y_private"], a["language"], a["canonical_case_id"],
                            a["stratum"], a["family_id"], a["pair_id"], a["provenance"])
        for judge in ["judge_a", "judge_b"]:
            from frauddistill.e1_final_v4.gold import gold_judge_task
            tasks.append(gold_judge_task(row, judge, cfg, "E1-v4-gold-negv2"))
    for a in benign_assign:
        row = _row_for_gold(a["response_id"], a["q_private"], a["y_private"], a["language"], a["canonical_case_id"],
                            a["stratum"], a["family_id"], a["pair_id"], a["provenance"])
        for judge in ["judge_a", "judge_b"]:
            from frauddistill.e1_final_v4.gold import gold_judge_task
            tasks.append(gold_judge_task(row, judge, cfg, "E1-v4-gold-benignv2"))
    write_jsonl(OUT_DIR / "E1_V4_GOLD_TASKS_NEGV2.jsonl", tasks)
    result = execute_json_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GOLD_VOTES.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider=concurrency_for(cfg), pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


def phase_adjudicate(cfg, args) -> dict[str, Any]:
    votes = _vote_index()
    existing_adj = {r["response_id"] for r in read_jsonl(OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl") if r.get("status") == "ok"}
    assignments = read_jsonl(OUT_DIR / "E1_V4_NEG_V2_ASSIGNMENTS.jsonl")
    benign_assign = read_jsonl(OUT_DIR / "E1_V4_B1_BENIGN_V2_ASSIGNMENTS.jsonl")
    rid_set = {a["response_id"] for a in assignments} | {a["response_id"] for a in benign_assign}
    content = {a["response_id"]: {"q_private": a["q_private"], "y_private": a["y_private"]} for a in assignments}
    content.update({a["response_id"]: {"q_private": a["q_private"], "y_private": a["y_private"]} for a in benign_assign})
    tasks = []
    for rid in sorted(rid_set):
        if rid in existing_adj:
            continue
        vv = votes.get(rid) or {}
        if needs_adjudication(vv.get("judge_a"), vv.get("judge_b")):
            tasks.append(adjudication_task(content[rid], vv.get("judge_a"), vv.get("judge_b"), cfg, "E1-v4-adjudicate-negv2"))
    print(f"[adjudicate] needed={len(tasks)}", flush=True)
    result = execute_json_tasks(
        tasks, output_path=OUT_DIR / "E1_V4_GOLD_ADJUDICATION.jsonl", ledger_path=OUT_DIR / "E1_V4_BUDGET_LEDGER.jsonl",
        limits=limits_for(cfg), run_api=args.run_api, confirm_budget=args.confirm_budget, git_clean=git_clean(),
        limit=args.limit, concurrency_by_provider={"qwen": concurrency_for(cfg)["adjudicator"]},
        pricing=cfg.get("pricing_cny_per_million_tokens"))
    return {"n_tasks": len(tasks), "result": result}


# ---------------------------------------------------------------- phase: reassemble


def _resolved(rid: str, votes, adj) -> tuple[int | None, str]:
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


def phase_reassemble(cfg, args) -> dict[str, Any]:
    panel = panel_rows()
    votes = _vote_index()
    adj = _adj_index()
    assignments = read_jsonl(OUT_DIR / "E1_V4_NEG_V2_ASSIGNMENTS.jsonl")
    benign_assign = read_jsonl(OUT_DIR / "E1_V4_B1_BENIGN_V2_ASSIGNMENTS.jsonl")
    old_neg_ids = {r["response_id"] for r in old_neg_rows()}

    # resolve benign v2 rows
    benign_used = []
    benign_failed = []
    for a in benign_assign:
        lab, m = _resolved(a["response_id"], votes, adj)
        if lab == 0:
            benign_used.append(a)
        else:
            benign_failed.append(a["response_id"])

    # gold quality stats for the new rows
    new_rows = [a for a in assignments]
    agree = 0
    adjudicated = 0
    unresolved = 0
    used = []
    failed = []
    for a in new_rows:
        lab, method = _resolved(a["response_id"], votes, adj)
        if lab is None:
            unresolved += 1
            failed.append(a["response_id"])
            continue
        if lab == 0:
            used.append(a)
            if method == "double_agree":
                agree += 1
            else:
                adjudicated += 1
        else:
            failed.append(a["response_id"])

    method_by_rid = {}
    for a in used:
        _, m = _resolved(a["response_id"], votes, adj)
        method_by_rid[a["response_id"]] = m

    # rebuild panel: keep non-neg rows + used v2 rows (+ benign v2 rows)
    old_benign_ids = {a["response_id"][:-3] for a in benign_used}
    keep = [r for r in panel if r["response_id"] not in old_neg_ids and r["response_id"] not in old_benign_ids]
    for a in used:
        keep.append({
            "response_id": a["response_id"], "q_private": a["q_private"], "y_private": a["y_private"],
            "language": a["language"], "canonical_case_id": a["canonical_case_id"], "stratum": a["stratum"],
            "family_id": a["family_id"], "pair_id": a["pair_id"], "provenance": a["provenance"],
            "gold_central": 0, "gold_method": method_by_rid.get(a["response_id"], "adjudicated"),
            "gold_status": "KNOWN", "source_dataset": "E1-FINAL-TRIAD-v4",
        })
    for a in benign_used:
        keep.append({
            "response_id": a["response_id"], "q_private": a["q_private"], "y_private": a["y_private"],
            "language": a["language"], "canonical_case_id": a["canonical_case_id"], "stratum": a["stratum"],
            "family_id": a["family_id"], "pair_id": a["pair_id"], "provenance": a["provenance"],
            "gold_central": 0, "gold_method": "double_agree",
            "gold_status": "KNOWN", "source_dataset": "E1-FINAL-TRIAD-v4", "benign_source": a.get("benign_source"),
        })

    # quota check per (stratum, language) for negatives vs the pre-fix panel
    old_counts = collections.Counter((r["stratum"], r["language"]) for r in panel if int(r["gold_central"]) == 0)
    new_counts = collections.Counter((r["stratum"], r["language"]) for r in keep if int(r["gold_central"]) == 0)
    missing = {f"{k[0][:3]}_{k[1]}": old_counts[k] - new_counts[k] for k in old_counts if old_counts[k] - new_counts[k] > 0}
    print(f"[reassemble] kept={len(keep)} used_v2={len(used)} benign_v2={len(benign_used)} benign_failed={len(benign_failed)} failed={len(failed)} unresolved={unresolved} missing={missing}", flush=True)

    # report counts
    counts = collections.Counter(r["stratum"] for r in keep)
    audit = {
        "n_rows": len(keep),
        "by_stratum": dict(counts),
        "by_label": dict(collections.Counter(r["gold_central"] for r in keep)),
        "by_language": dict(collections.Counter(r["language"] for r in keep)),
        "by_provenance": dict(collections.Counter(r["provenance"] for r in keep)),
        "neg_v2_used": len(used),
        "neg_v2_failed": failed,
        "benign_v2_used": len(benign_used),
        "benign_v2_failed": benign_failed,
        "neg_v2_agree": agree,
        "neg_v2_adjudicated": adjudicated,
        "neg_v2_unresolved": unresolved,
        "missing": dict(missing),
    }
    write_jsonl(OUT_DIR / "E1_V4_PANEL_ALL.jsonl", keep)
    write_json(OUT_DIR / "E1_V4_PANEL_AUDIT.json", audit)
    return {"status": "REASSEMBLED", "audit": audit}


PHASES = {
    "gen-defensive": phase_gen_defensive,
    "gen-benign": phase_gen_benign,
    "gold": phase_gold,
    "adjudicate": phase_adjudicate,
    "reassemble": phase_reassemble,
}


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
