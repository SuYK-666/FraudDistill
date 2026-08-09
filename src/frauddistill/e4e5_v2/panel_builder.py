# -*- coding: utf-8 -*-
"""Panel builder: U2 candidates (Aegis/PKU official labels), U1/U3 assembly,
family-level calibration/test split, pilot extraction, manifest hashes."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .exposure_registry import ExposureRegistry
from .gold_mapper import assign_gold, rule_label
from .schemas import canonical_row, manifest_sha256, read_jsonl, write_jsonl


# --------------------------------------------------------------------------
# U2: Aegis validation (official response_label) + PKU-SafeRLHF (official
# response-level safe labels). G0 gold mapping.
# --------------------------------------------------------------------------
PKU_FRAUD_ADJACENT = {"Cybercrime", "Economic Crime", "White-Collar Crime", "Mental Manipulation"}


def aegis_candidates(aegis_validation: Path) -> list[dict]:
    with open(aegis_validation, encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        rl = r.get("response_label")
        if rl not in ("safe", "unsafe"):
            continue
        q = str(r.get("prompt") or "").strip()
        y = str(r.get("response") or "").strip()
        if not q or not y:
            continue
        cats = str(r.get("violated_categories") or "")
        gtype = "safe" if rl == "safe" else ("fraud_assistance" if any(
            c in cats for c in ("Financial", "Fraud", "Criminal Planning")) else "refusal_failure")
        out.append(canonical_row(
            id_="u2_aegis_" + str(r["id"])[:12],
            q=q, y=y,
            gold_label=rl,
            gold_type=gtype,
            source="aegis",
            source_version="validation-v1",
            language="en",
            target_model="aegis-response-pool",
            primary_shift="U2_source",
            family_id="u2_aegis_fam_" + str(r["id"])[:12],
            template_id="u2_aegis_tpl_" + str(r["id"])[:12],
            gold_source="official_response_label",
            license="cc-by-4.0",
            fraud_category="general_harm",
            metadata={"original_id": r["id"], "violated_categories": cats,
                      "prompt_label": r.get("prompt_label"),
                      "response_label_source": r.get("response_label_source")},
        ))
    return out


def pku_candidates(pku_files: list[Path]) -> list[dict]:
    out = []
    for fp in pku_files:
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            q = str(r.get("prompt") or "").strip()
            if not q:
                continue
            for k in ("0", "1"):
                y = str(r.get("response_" + k) or "").strip()
                safe = r.get("is_response_" + k + "_safe")
                if not y or safe is None:
                    continue
                hc = r.get("response_" + k + "_harm_category") or {}
                fraud_adj = any(hc.get(c) for c in PKU_FRAUD_ADJACENT)
                gtype = "safe" if safe else ("fraud_assistance" if fraud_adj else "refusal_failure")
                fam = "u2_pku_fam_" + r.get("response_0_sha256", r.get("prompt", ""))[:12]
                out.append(canonical_row(
                    id_="u2_pku_" + (r.get("response_" + k + "_sha256") or "")[:16],
                    q=q, y=y,
                    gold_label="safe" if safe else "unsafe",
                    gold_type=gtype,
                    source="pku_saferlhf",
                    source_version=fp.stem,
                    language="en",
                    target_model=str(r.get("response_" + k + "_source") or ""),
                    primary_shift="U2_source",
                    family_id=fam,
                    template_id=fam,
                    gold_source="official_response_label",
                    license="cc-by-nc-4.0",
                    fraud_category="general_harm",
                    metadata={"prompt_source": r.get("prompt_source"),
                              "harm_categories": hc, "severity": r.get("response_" + k + "_severity_level")},
                ))
    return out


# --------------------------------------------------------------------------
# U1/U3: generated rows -> provisional gold via G1 rules (G2 later)
# --------------------------------------------------------------------------
def generated_candidates(gen_rows: list[dict], shift: str, source: str,
                         target_model: str) -> list[dict]:
    out = []
    for r in gen_rows:
        q = str(r.get("user_query") or "").strip()
        y = str(r.get("target_model_answer") or "").strip()
        if not q or not y:
            continue
        fam = str(r.get("family_id") or r.get("prompt_id") or "")
        out.append(canonical_row(
            id_=str(r["id"]),
            q=q, y=y,
            gold_label=None,
            gold_type=None,
            source=source,
            source_version="gen_v1",
            language=str(r.get("language") or "zh"),
            target_model=target_model,
            primary_shift=shift,
            family_id=fam,
            template_id=fam,
            gold_source="provisional_g1",
            fraud_category=str(r.get("fraud_category") or ""),
            metadata={"perspective": r.get("perspective", ""), "seed": r.get("seed"),
                      "mode": r.get("mode", "chat"), "prompt_id": r.get("prompt_id"),
                      "gen_status": r.get("generation_status")},
        ))
    return out


def apply_g1(rows: list[dict]) -> list[dict]:
    for r in rows:
        if r.get("gold_label"):
            continue
        res = rule_label(r["user_query"], r["target_model_answer"])
        r["gold_label"] = res["gold_label"]
        r["gold_type"] = res["gold_type"]
        r["metadata"]["g1"] = {k: res[k] for k in ("method", "refusal_hits", "action_hits") if k in res}
        r["gold_source"] = "rule_g1" if res["gold_label"] else "ambiguous_g2"
    return rows


# --------------------------------------------------------------------------
# Panel split (family-level) + pilot extraction
# --------------------------------------------------------------------------
def split_panel(candidates: list[dict], cfg: dict, seed: int = 20260809) -> dict:
    """Family-level split: calibration vs test per (shift, cell, label).

    Returns {pilot, calibration, test} rows. Families are never shared
    across splits and never across cells. Quotas from cfg['panel']['shifts'].
    Surplus families (unused by formal splits) go to pilot only if the family
    was never used in calibration/test.
    """
    rng = random.Random(seed)
    fams: dict[tuple, list[dict]] = defaultdict(list)
    for r in candidates:
        cell = r["source"] if r["primary_shift"] != "U1_category" else r["fraud_category"]
        fams[(r["primary_shift"], cell)].append(r)
    cal_rows, test_rows = [], []
    pilots = []
    used_fams: set[str] = set()
    for (shift, cell), rows in fams.items():
        by_fam: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_fam[r["family_id"]].append(r)
        fam_list = list(by_fam.keys())
        rng.shuffle(fam_list)

        def both_labels(f):
            return any(r["gold_label"] == "unsafe" for r in by_fam[f]) and any(r["gold_label"] == "safe" for r in by_fam[f])

        fam_list.sort(key=lambda f: (0 if both_labels(f) else 1, f))
        q = cfg["panel"]["shifts"][shift]
        need_cal = {"safe": q["calibration_safe"], "unsafe": q["calibration_unsafe"]}
        need_test = {"safe": q["test_safe"], "unsafe": q["test_unsafe"]}
        surplus: dict[str, list[dict]] = {}
        for fam in fam_list:
            frows = by_fam[fam]
            counts = Counter(r["gold_label"] for r in frows)
            cand_splits = []
            for split_name, need in (("cal", need_cal), ("test", need_test)):
                if all(need.get(lbl, 0) >= counts.get(lbl, 0) for lbl in ("safe", "unsafe")):
                    cand_splits.append((split_name, sum(need.get(l, 0) for l in ("safe", "unsafe"))))
            if not cand_splits:
                surplus[fam] = frows
                continue
            split_name = min(cand_splits, key=lambda x: x[1])[0]
            need = need_cal if split_name == "cal" else need_test
            for lbl in ("safe", "unsafe"):
                take = [r for r in frows if r["gold_label"] == lbl][: need[lbl]]
                if split_name == "cal":
                    cal_rows.extend(take)
                else:
                    test_rows.extend(take)
                need[lbl] -= len(take)
            used_fams.add(fam)
        # fill remaining needs from SAME-CELL surplus only (family-atomic)
        for split_name, need in (("cal", need_cal), ("test", need_test)):
            while sum(need.values()) > 0:
                fit_fam = None
                for f, frows in surplus.items():
                    counts = Counter(r["gold_label"] for r in frows)
                    if all(need.get(lbl, 0) >= counts.get(lbl, 0) for lbl in ("safe", "unsafe")):
                        fit_fam = f
                        break
                if fit_fam is None:
                    break
                frows = surplus.pop(fit_fam)
                for lbl in ("safe", "unsafe"):
                    take = [r for r in frows if r["gold_label"] == lbl][: need[lbl]]
                    if split_name == "cal":
                        cal_rows.extend(take)
                    else:
                        test_rows.extend(take)
                    need[lbl] -= len(take)
                used_fams.add(fit_fam)
        # leftover surplus of this cell -> pilot if family never used formally
        for f, frows in surplus.items():
            if f not in used_fams:
                pilots.extend(frows)
    return {"pilot": pilots, "calibration": cal_rows, "test": test_rows}


def check_quotas(split: dict, cfg: dict) -> dict:
    report = {}
    for name in ("calibration", "test"):
        rows = split[name]
        c = Counter((r["primary_shift"], r["source"] if r["primary_shift"] != "U1_category" else r["fraud_category"], r["gold_label"]) for r in rows)
        report[name] = {str(k): v for k, v in sorted(c.items())}
    return report


def exposure_audit_rows(registry: ExposureRegistry, rows: list[dict]) -> dict:
    bad = []
    fam_bad = []
    for r in rows:
        q = str(r.get("user_query") or "")
        y = str(r.get("target_model_answer") or "")
        ex = registry.exact_overlap(q, y)
        fov = registry.family_overlap(str(r.get("family_id") or ""))
        if ex["qy_exact"] or ex["q_exact"] or ex["y_exact"] or fov:
            bad.append({"id": r["id"], **ex, "family_overlap": fov})
        if registry.family_overlap(str(r.get("family_id") or "")):
            fam_bad.append(r["id"])
    return {"n": len(rows), "n_failed": len(bad), "passed": len(bad) == 0, "failures": bad[:100]}


def write_manifests(split: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("pilot", "calibration", "frozen_test"):
        rows = split.get(name, [])
        if name == "frozen_test":
            rows = split.get("test", [])
        p = out_dir / f"{name}.jsonl"
        write_jsonl(p, rows)
        hashes[name] = {"n": len(rows), "sha256": manifest_sha256(rows)}
    (out_dir / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashes
