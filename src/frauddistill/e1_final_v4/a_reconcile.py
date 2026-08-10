# -*- coding: utf-8 -*-
"""E1-A reconciliation & prevalence statistics (offline, no API)."""
from __future__ import annotations

import collections
import json
import math
from typing import Any

import numpy as np

from frauddistill.e1_final_v3.io import read_jsonl, write_json


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(rows: list[dict[str, Any]], rng: np.random.Generator | None = None, iters: int = 10000) -> dict[str, Any]:
    rng = rng or np.random.default_rng(20260810)
    cases = collections.defaultdict(list)
    for r in rows:
        cases[str(r.get("canonical_case_id") or r.get("response_id"))].append(int(r.get("gold_central", 0) or 0))
    case_ids = list(cases)
    rates = []
    for _ in range(iters):
        chosen = rng.choice(case_ids, size=len(case_ids), replace=True)
        k = sum(sum(cases[c]) for c in chosen)
        n = sum(len(cases[c]) for c in chosen)
        rates.append(k / n if n else 0.0)
    rates = np.asarray(rates)
    return {
        "mean": float(rates.mean()),
        "ci95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))],
        "iterations": iters,
    }


def mcnemar(a: int, b: int) -> dict[str, Any]:
    """Paired 2x2: a = pos/neg, b = neg/pos."""
    if a + b == 0:
        return {"n_discordant": 0, "p": 1.0, "note": "no discordant pairs"}
    # exact binomial two-sided
    from scipy.stats import binomtest
    p = binomtest(min(a, b), a + b, 0.5, alternative="two-sided").pvalue
    return {"n_discordant": a + b, "a": a, "b": b, "p": float(p)}


def reconcile_a(v32_real_pool_path, out_dir) -> dict[str, Any]:
    rows = read_jsonl(v32_real_pool_path)
    n = len(rows)
    # integrity
    rid_dup = [k for k, v in collections.Counter(r.get("response_id") for r in rows).items() if v > 1]
    gold_missing = sum(1 for r in rows if r.get("gold_central") is None)
    judge_missing = sum(1 for r in rows if r.get("judge_a_positive") is None or r.get("judge_b_positive") is None)
    valid_schema = sum(1 for r in rows if isinstance(r.get("gold_central"), int) and r.get("response_id"))
    agree = sum(1 for r in rows if r.get("judge_a_positive") is not None and r.get("judge_b_positive") is not None and bool(r.get("judge_a_positive")) == bool(r.get("judge_b_positive")))
    n_judged = sum(1 for r in rows if r.get("judge_a_positive") is not None and r.get("judge_b_positive") is not None)

    central_pos = sum(1 for r in rows if int(r.get("gold_central", -1)) == 1)
    lower_pos = sum(1 for r in rows if int(r.get("gold_lower", -1)) == 1)
    upper_pos = sum(1 for r in rows if int(r.get("gold_upper", -1)) == 1)
    ja_pos = sum(1 for r in rows if bool(r.get("judge_a_positive")))
    jb_pos = sum(1 for r in rows if bool(r.get("judge_b_positive")))
    known = sum(1 for r in rows if str(r.get("gold_status", "")).startswith("KNOWN"))
    reused = sum(1 for r in rows if r.get("gold_method") == "reused_v31_A_frozen")
    adjudicated = sum(1 for r in rows if r.get("gold_method") == "adjudicated")
    double_agree = sum(1 for r in rows if r.get("gold_method") == "double_agree")
    unresolved = sum(1 for r in rows if r.get("gold_method") in (None, "") and not str(r.get("gold_status", "")).startswith("KNOWN"))

    # strata
    strata = {}
    for key in ["target_provider", "scenario", "language", "fraud_category"]:
        groups = collections.defaultdict(list)
        for r in rows:
            groups[str(r.get(key) or "unknown")].append(r)
        strata[key] = {
            g: {
                "n": len(sub),
                "positive": sum(1 for x in sub if int(x.get("gold_central", -1)) == 1),
                "rate": round(sum(1 for x in sub if int(x.get("gold_central", -1)) == 1) / max(1, len(sub)), 6),
            }
            for g, sub in sorted(groups.items())
        }

    # paired McNemar qwen vs deepseek (same canonical_case_id, both providers)
    by_case = collections.defaultdict(dict)
    for r in rows:
        by_case[r["canonical_case_id"]][r["target_provider"]] = r
    disc_pairs = 0
    a_side = 0
    b_side = 0
    for cid, provs in by_case.items():
        if "qwen" in provs and "deepseek" in provs:
            lq = int(provs["qwen"].get("gold_central", -1))
            ld = int(provs["deepseek"].get("gold_central", -1))
            if lq >= 0 and ld >= 0:
                if lq == 1 and ld == 0:
                    a_side += 1
                elif lq == 0 and ld == 1:
                    b_side += 1
    mcn = mcnemar(a_side, b_side)

    w = wilson_ci(central_pos, n)
    cb = cluster_bootstrap(rows)
    result = {
        "registry_rows": n,
        "unique_response_ids": len(set(r.get("response_id") for r in rows)),
        "duplicate_response_ids": len(rid_dup),
        "gold_completion": round(known / n, 6),
        "gold_missing": gold_missing,
        "judge_vote_missing": judge_missing,
        "valid_schema_rate": round(valid_schema / n, 6),
        "judge_agreement": round(agree / max(1, n_judged), 6),
        "judged_rows": n_judged,
        "central_positives": central_pos,
        "lower_positives": lower_pos,
        "upper_positives": upper_pos,
        "judge_a_positives": ja_pos,
        "judge_b_positives": jb_pos,
        "gold_method": {"reused": reused, "adjudicated": adjudicated, "double_agree": double_agree},
        "central_rate": round(central_pos / n, 6),
        "wilson_ci95": [round(w[0], 6), round(w[1], 6)],
        "cluster_bootstrap": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cb.items()},
        "strata": strata,
        "paired_mcnemar_qwen_vs_deepseek": mcn,
        "reconciliation": {
            "v32_central_11_source": "E1_V32_REAL_POOL.jsonl gold_central=1 (frozen v3.2 canonical registry)",
            "old_28_source": "v3.1 E1_FINAL_C_PREDICTIONS gold filter used upper-bound/reused labels (26 unique response_ids x 2 eval modes)",
            "paper_canonical": "gold_central (material_central consensus), 11/7500",
            "note": "差异来自 endpoint 口径 (central vs upper/reused) 与 C-eval 双模式计数, 不是响应集差异",
        },
    }
    write_json(out_dir / "E1_V4_A_RECONCILIATION.json", result)
    return result
