# -*- coding: utf-8 -*-
"""Build the Exp2 FINAL full-size manifests (formal-guide 2026-08-06, sections 2-6).

Outputs (experiments/exp2_prior_work_comparison/manifests/):
  fraudr1_content_final_3000.jsonl   3,000 content rows (all positive candidates + balanced negatives)
  orbench_final_3000.jsonl           3,000 full core pool (hard 1,000 / regular 1,400 / toxic 600)
  dna_final_5634.jsonl               5,634 full pool
  aegis_validation.jsonl             1,445 official validation (687 response-labeled rows flagged)
  aegis_response_test_813.jsonl        813 official response test rows
  aegis_prompt_test_1151.jsonl       1,151 prompt-only test rows (appendix only)

Positive-candidate signals for Fraud-R1 (guide 3.5): Official Judge positive,
old T6 teacher positive, prior adjudicated audit positive. Candidate signals
are used ONLY to enrich the positive share; final gold comes from the blind
audit pipeline (gold/).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "experiments" / "exp2_prior_work_comparison"
MANIFEST_DIR = BASE / "manifests"
RAW_AEGIS = REPO / "data" / "raw" / "aegis"
SEED = 20260806

FRAUD_FAMILIES = [
    "Fraudulent Services", "Impersonation", "Phishing Scams",
    "Fake Job Postings", "Online Relationships",
]


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def qy_hash(query: str, answer: str) -> str:
    payload = normalize_text(query) + "\0" + normalize_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_exposed_ids() -> set[str]:
    """dev300 + all pilot manifests (rows the frozen final 3,000 must not overlap)."""
    exposed: set[str] = set()
    files = [
        BASE / "archive" / "legacy_20260803" / "_dev_manifest" / "fraudr1_diag_dev300.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "final_pilot_manifest.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "skill_gate_manifest.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "repair_pilot.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "boundary_pilot.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "boundary_smoke.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "final_pilot_smoke.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "final_pilot_half.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "skill_gate_main.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "skill_gate_smoke.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "skill_gate_diagnostic.jsonl",
        BASE / "archive" / "prefinal_20260806" / "pilot" / "retry_manifest.jsonl",
    ]
    for p in files:
        for r in read_jsonl(p):
            sid = str(r.get("sample_id") or r.get("id") or "")
            if sid.startswith("fraudr1_") or sid.startswith("orbench_"):
                exposed.add(sid)
    return exposed


def build_fraudr1() -> list[dict]:
    full = [r for r in read_jsonl(MANIFEST_DIR / "full_manifest.jsonl") if r["source"] == "fraudr1"]
    exposed = load_exposed_ids()
    pool = [r for r in full if r["sample_id"] not in exposed]
    # Guide 3.7: unique qy_hash per manifest row; drop pool-level qy duplicates
    # (2 base/levelup twin cases share identical q+y).
    seen_qy: set[str] = set()
    deduped = []
    for r in pool:
        if r["qy_hash"] in seen_qy:
            continue
        seen_qy.add(r["qy_hash"])
        deduped.append(r)
    pool = deduped
    print(f"[fraud] pool={len(pool)} exposed={len(full)-len(pool)}")

    judge = {str(r["id"]): r for r in read_jsonl(BASE / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl")}
    t6 = {str(r["id"]): r for r in read_jsonl(BASE / "archive" / "prefinal_20260806" / "teacher_predictions_t6" / "fraudr1_t6_predictions.jsonl")}
    aud = {r["id"]: r for r in read_jsonl(BASE / "fraudr1" / "human_audit" / "human_audit_adjudicated_20260805.jsonl")}

    pos_ids: set[str] = set()
    for r in pool:
        sid = r["sample_id"]
        if judge.get(sid, {}).get("prediction_binary") == 1:
            pos_ids.add(sid)
        if t6.get(sid, {}).get("prediction_binary") == 1:
            pos_ids.add(sid)
        if sid in aud and aud[sid].get("binary") == 1:
            pos_ids.add(sid)
    neg_ids = [r["sample_id"] for r in pool if r["sample_id"] not in pos_ids]
    print(f"[fraud] positive candidates={len(pos_ids)} negatives available={len(neg_ids)}")

    rng = random.Random(SEED)
    by = {r["sample_id"]: r for r in pool}
    pos = sorted(pos_ids)
    rng.shuffle(pos)

    # Deficit-driven quota fill (guide 3.1/3.2 priorities: family ~600 each,
    # language zh=en=1500, scenario assistant=roleplay=1500).
    family_lang_scen: dict[tuple, list[str]] = defaultdict(list)
    for sid in neg_ids:
        r = by[sid]
        family_lang_scen[(r["official_category"], r["language"],
                          (r.get("metadata") or {}).get("fraudr1_scenario"))].append(sid)
    for v in family_lang_scen.values():
        rng.shuffle(v)

    need_total = 3000 - len(pos)
    fam_need = {f: 600 - sum(1 for s in pos if by[s]["official_category"] == f) for f in FRAUD_FAMILIES}
    lang_need = {"Chinese": 1500 - sum(1 for s in pos if by[s]["language"] == "Chinese"),
                 "English": 1500 - sum(1 for s in pos if by[s]["language"] == "English")}
    scen_need = {"assistant": 1500 - sum(1 for s in pos if by[s]["metadata"]["fraudr1_scenario"] == "assistant"),
                 "roleplay": 1500 - sum(1 for s in pos if by[s]["metadata"]["fraudr1_scenario"] == "roleplay")}
    assert need_total == lang_need["Chinese"] + lang_need["English"] == scen_need["assistant"] + scen_need["roleplay"]
    assert sum(fam_need.values()) == need_total

    cells = sorted((f, l, s) for f in FRAUD_FAMILIES for l in ("Chinese", "English") for s in ("assistant", "roleplay"))
    quotas: dict[tuple, int] = {c: 0 for c in cells}
    remain = {c: list(family_lang_scen.get(c, [])) for c in cells}
    for _ in range(need_total):
        best, best_score = None, -1
        for c in cells:
            if not remain[c]:
                continue
            f, l, s = c
            if fam_need[f] <= 0 or lang_need[l] <= 0 or scen_need[s] <= 0:
                continue
            score = fam_need[f] + lang_need[l] + scen_need[s]
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        quotas[best] += 1
        fam_need[best[0]] -= 1
        lang_need[best[1]] -= 1
        scen_need[best[2]] -= 1
        remain[best].pop(0)
    neg_sel: list[str] = []
    for c, q in quotas.items():
        neg_sel += family_lang_scen.get(c, [])[:q]
    if len(neg_sel) < need_total:
        chosen = set(neg_sel)
        for sid in neg_ids:
            if len(neg_sel) >= need_total:
                break
            if sid not in chosen:
                neg_sel.append(sid)
                chosen.add(sid)
    print(f"[fraud] selected negatives={len(neg_sel)} (need {need_total})")
    selected = pos + neg_sel
    assert len(selected) == 3000, f"selected {len(selected)} != 3000"

    out = []
    for sid in selected:
        r = by[sid]
        md = r.get("metadata") or {}
        out.append({
            "sample_id": sid,
            "group_id": r["group_id"],
            "qy_hash": r["qy_hash"],
            "source": "fraudr1",
            "fraud_family": r["official_category"],
            "language": r["language"],
            "scenario": md.get("fraudr1_scenario", ""),
            "variant": md.get("fraudr1_variant", ""),
            "query": r["query"],
            "answer": r["answer"],
            "gold_binary": None,
            "gold_type": "adjudicated_silver",
            "label_source": "",
            "previous_dev_exposure": False,
            "positive_candidate": sid in pos_ids,
            "target_model": r.get("target_model", ""),
            "overlap_exp3": bool(r.get("overlap_exp3")),
        })
    return out


def build_orbench() -> list[dict]:
    full = [r for r in read_jsonl(MANIFEST_DIR / "full_manifest.jsonl") if r["source"] == "orbench"]
    strata = Counter(r.get("official_gold_type") for r in full)
    print(f"[orbench] pool={len(full)} strata={dict(strata)}")
    out = []
    for r in full:
        out.append({
            "sample_id": r["sample_id"],
            "group_id": r["group_id"],
            "qy_hash": r["qy_hash"],
            "source": "orbench",
            "request_gold_type": r.get("official_gold_type", ""),
            "language": r.get("language", ""),
            "query": r["query"],
            "answer": r["answer"],
            "gold_binary": None,
            "gold_type": "adjudicated_silver",
            "label_source": "",
            "previous_dev_exposure": False,
            "target_model": r.get("target_model", ""),
            "overlap_exp3": bool(r.get("overlap_exp3")),
        })
    return out


def build_dna() -> list[dict]:
    full = [r for r in read_jsonl(MANIFEST_DIR / "full_manifest.jsonl") if r["source"] == "do_not_answer"]
    print(f"[dna] pool={len(full)}")
    out = []
    for r in full:
        out.append({
            "sample_id": r["sample_id"],
            "group_id": r["group_id"],
            "qy_hash": r["qy_hash"],
            "source": "do_not_answer",
            "query": r["query"],
            "answer": r["answer"],
            "official_harmfulness": r.get("official_gold_binary"),
            "target_model": r.get("target_model", ""),
            "prompt_id": r.get("original_id", ""),
            "overlap_exp3": bool(r.get("overlap_exp3")),
        })
    return out


def build_aegis() -> tuple[list[dict], list[dict], list[dict]]:
    def rows_from(raw: list[dict], split: str, id_suffix: str = "") -> list[dict]:
        out = []
        for r in raw:
            sid = f"aegis_{r['id']}{id_suffix}"
            resp = (r.get("response") or "").strip()
            rl = r.get("response_label")
            has_resp = bool(resp) and rl not in (None, "")
            rec = {
                "sample_id": sid,
                "group_id": sid,
                "qy_hash": qy_hash(str(r.get("prompt", "")), resp),
                "source": "aegis2",
                "query": str(r.get("prompt", "")),
                "answer": resp,
                "evaluation_mode": "response" if has_resp else "prompt_only",
                "official_response_label": None if not has_resp else (1 if str(rl).lower().startswith("unsafe") else 0),
                "official_prompt_label": r.get("prompt_label"),
                "violated_categories": r.get("violated_categories") or "",
                "label_source": r.get("response_label_source", ""),
                "split": split,
            }
            out.append(rec)
        return out

    val = rows_from(json.loads((RAW_AEGIS / "validation.json").read_text(encoding="utf-8")), "validation")
    # refusals_validation.json holds ALTERNATE (refused) responses for prompts
    # that also appear in validation.json; the raw file contains 46 internal
    # (id,response) duplicates and every id collides with validation.json ids,
    # so refusals get a "_refusal" suffix and internal dups are dropped
    # (sample_id is the (prompt,response) unit).
    ref_raw = json.loads((RAW_AEGIS / "refusals_validation.json").read_text(encoding="utf-8"))
    seen_ref: set[str] = set()
    ref_unique = []
    for r in ref_raw:
        h = qy_hash(str(r.get("prompt", "")), str(r.get("response", "") or ""))
        if h in seen_ref:
            continue
        seen_ref.add(h)
        ref_unique.append(r)
    ref_val = rows_from(ref_unique, "validation", id_suffix="_refusal")
    test = rows_from(json.loads((RAW_AEGIS / "test.json").read_text(encoding="utf-8")), "test")
    validation = val + ref_val
    assert len({r["sample_id"] for r in validation}) == len(validation), "validation sample_id collision"
    # NOTE: the official validation split contains 43 interactions with
    # identical (q,y) content but distinct ids; sample_id is the unit.
    print(f"[aegis] validation={len(validation)} (response-labeled={sum(1 for r in validation if r['evaluation_mode']=='response')})")
    print(f"[aegis] test={len(test)} response={sum(1 for r in test if r['evaluation_mode']=='response')} prompt={sum(1 for r in test if r['evaluation_mode']=='prompt_only')}")
    test_resp = [r for r in test if r["evaluation_mode"] == "response"]
    test_prompt = [r for r in test if r["evaluation_mode"] == "prompt_only"]
    return validation, test_resp, test_prompt


def main() -> None:
    fraud = build_fraudr1()
    orbench = build_orbench()
    dna = build_dna()
    aegis_val, aegis_test_resp, aegis_test_prompt = build_aegis()

    write_jsonl(MANIFEST_DIR / "fraudr1_content_final_3000.jsonl", fraud)
    write_jsonl(MANIFEST_DIR / "orbench_final_3000.jsonl", orbench)
    write_jsonl(MANIFEST_DIR / "dna_final_5634.jsonl", dna)
    write_jsonl(MANIFEST_DIR / "aegis_validation.jsonl", aegis_val)
    write_jsonl(MANIFEST_DIR / "aegis_response_test_813.jsonl", aegis_test_resp)
    write_jsonl(MANIFEST_DIR / "aegis_prompt_test_1151.jsonl", aegis_test_prompt)

    # ---- assertions (guide 3.7 + technical gate) ----
    assert len(fraud) == 3000
    assert len({r["sample_id"] for r in fraud}) == 3000
    assert len({r["qy_hash"] for r in fraud}) == 3000
    assert not any(r["previous_dev_exposure"] for r in fraud)
    assert set(r["fraud_family"] for r in fraud) == set(FRAUD_FAMILIES)
    assert len(orbench) == 3000
    assert len(dna) == 5634
    assert len(aegis_val) == 1399  # 1,245 validation + 154 unique refusals (deduped)
    assert len(aegis_test_resp) == 813
    assert len(aegis_test_prompt) == 1151
    assert len({r["sample_id"] for r in orbench}) == 3000
    assert len({r["sample_id"] for r in dna}) == 5634
    assert len({r["sample_id"] for r in aegis_test_resp}) == 813

    summary = {
        "seed": SEED,
        "fraudr1": {
            "n": len(fraud),
            "n_pos_candidates": sum(1 for r in fraud if r["positive_candidate"]),
            "positive_rate_candidates": round(sum(1 for r in fraud if r["positive_candidate"]) / 3000, 4),
            "family": dict(Counter(r["fraud_family"] for r in fraud)),
            "language": dict(Counter(r["language"] for r in fraud)),
            "scenario": dict(Counter(r["scenario"] for r in fraud)),
            "variant": dict(Counter(r["variant"] for r in fraud)),
            "family_lang": {f"{k[0]}|{k[1]}": v for k, v in Counter((r["fraud_family"], r["language"]) for r in fraud).items()},
        },
        "orbench": {
            "n": len(orbench),
            "strata": dict(Counter(r["request_gold_type"] for r in orbench)),
        },
        "dna": {"n": len(dna), "positive_candidates": sum(1 for r in dna if r.get("official_harmfulness") == 1)},
        "aegis_validation": {"n": len(aegis_val), "response_labeled": sum(1 for r in aegis_val if r["evaluation_mode"] == "response"),
                              "note": "1,245 validation + 154 unique refusals (46 raw dup rows dropped)"},
        "aegis_test_response": {"n": len(aegis_test_resp), "unsafe": sum(1 for r in aegis_test_resp if r["official_response_label"] == 1)},
        "aegis_test_prompt": {"n": len(aegis_test_prompt)},
        "main_table_total": 3000 + 3000 + 5634 + 813,
    }
    (MANIFEST_DIR / "exp2_final_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()