from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from frauddistill.utils.io import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[3]
FRAUD_TERMS = (
    "fraud", "scam", "phishing", "impersonat", "credential", "password", "otp", "bank transfer",
    "wire transfer", "money mule", "launder", "fake job", "romance scam", "investment scam",
    "deception", "deceive", "payment diversion", "account takeover", "验证码", "诈骗", "钓鱼",
    "冒充", "转账", "洗钱", "跑分", "虚假招聘", "刷单", "套现", "银行卡", "验证码", "骗",
)
REFUSAL_TERMS = ("cannot", "can't", "sorry", "refuse", "illegal", "safe", "report", "prevent", "warning", "抱歉", "不能", "违法", "举报", "防范", "提醒")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(seed: int, *parts: Any) -> str:
    return sha_text(":".join([str(seed), *[str(p) for p in parts]]))


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fraud_scope_filter(row: dict) -> bool:
    meta = row.get("metadata") or {}
    category = " ".join(str(v) for v in [meta.get("category"), meta.get("violated_categories"), meta.get("harm_category"), meta.get("harm_category_union"), row.get("risk_type")])
    text = f"{row.get('user_query','')} {row.get('target_model_answer','')} {category}".lower()
    if not any(term.lower() in text for term in FRAUD_TERMS):
        return False
    if row.get("gold_label") == "unsafe":
        return any(term.lower() in text for term in FRAUD_TERMS)
    if row.get("gold_label") == "safe":
        return any(term.lower() in text for term in FRAUD_TERMS) and (
            any(term.lower() in str(row.get("target_model_answer", "")).lower() for term in REFUSAL_TERMS)
            or any(term.lower() in str(row.get("user_query", "")).lower() for term in FRAUD_TERMS)
        )
    return False


def normalize_public_row(row: dict, source: str) -> dict | None:
    label = str(row.get("gold_label") or row.get("exp1_label") or row.get("pair_fraud_label") or "").lower()
    if label not in {"safe", "unsafe"}:
        return None
    q = normalize_ws(row.get("user_query") or row.get("prompt"))
    y = normalize_ws(row.get("target_model_answer") or row.get("response"))
    if not q or not y or y.lower() in {"none", "null", "redacted"}:
        return None
    meta = dict(row.get("metadata") or {})
    raw_id = str(row.get("id") or row.get("row_uid") or sha_text(q + "\n" + y)[:24])
    risk = secondary_type(label, q, y)
    item = {
        "id": f"r4_public_{source}_{raw_id}_{sha_text(q + chr(10) + y)[:12]}",
        "source": source,
        "language": row.get("language") or meta.get("language") or "English",
        "user_query": q,
        "target_model_answer": y,
        "gold_label": label,
        "gold_risk_type": risk,
        "label_provenance": row.get("label_provenance") or meta.get("response_label_source") or "official_response_or_interaction_label",
        "source_prompt_id": row.get("source_prompt_id") or meta.get("source_prompt_id") or sha_text(q)[:24],
        "metadata": meta | {"source_original_id": raw_id, "source_label_field": response_label_field(row)},
    }
    if not fraud_scope_filter(item):
        return None
    item["canonical_group_id"] = f"natural_{source}_{sha_text(item['source_prompt_id'])[:18]}"
    return item


def secondary_type(label: str, q: str, y: str) -> str:
    ylow = y.lower()
    if label == "unsafe":
        if any(term in ylow for term in ("yes", "sure", "当然", "可以", "here is", "message", "script")):
            return "unsafe_fraud_enablement"
        return "unsafe_defense_failure"
    if any(term.lower() in ylow for term in REFUSAL_TERMS):
        return "safe_refusal"
    if any(term.lower() in ylow for term in ("detect", "report", "prevent", "warning", "识别", "举报", "防范")):
        return "safe_detection_or_prevention"
    return "safe_benign"


def response_label_field(row: dict) -> str:
    meta = row.get("metadata") or {}
    for key in ("response_label", "is_response_safe", "is_safe", "response_harm_label"):
        if key in row or key in meta:
            return key
    return "unknown"


def load_public_sources(config: dict) -> tuple[dict[str, list[dict]], dict]:
    audit = {"sources": {}, "failures": [], "source_files": {}}
    sources = {
        "PKU-SafeRLHF": load_pku_rows(config, audit),
        "BeaverTails": load_beaver_rows(config, audit),
        "Aegis": load_aegis_rows(audit),
    }
    normalized = {}
    for source, rows in sources.items():
        out = []
        seen = set()
        for row in rows:
            item = normalize_public_row(row, source)
            if not item:
                continue
            key = sha_text(item["user_query"] + "\n" + item["target_model_answer"])
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        normalized[source] = out
        labels = Counter(row["gold_label"] for row in out)
        audit["sources"][source] = {"rows": len(out), "safe": labels.get("safe", 0), "unsafe": labels.get("unsafe", 0)}
    return normalized, audit


def load_pku_rows(config: dict, audit: dict) -> list[dict]:
    rows = []
    try:
        from datasets import load_dataset

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        for split in ("train", "test"):
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, revision=config["data"]["public_revisions"].get("pku_saferlhf", "main"))
            audit["source_files"][f"PKU-SafeRLHF:{split}"] = {"dataset": "PKU-Alignment/PKU-SafeRLHF", "revision": config["data"]["public_revisions"].get("pku_saferlhf", "main"), "rows_raw": len(ds)}
            for idx, item in enumerate(ds):
                prompt = normalize_ws(item.get("prompt"))
                if not prompt:
                    continue
                for ridx in (0, 1):
                    answer = normalize_ws(item.get(f"response_{ridx}"))
                    if not answer:
                        continue
                    safe = bool(item.get(f"is_response_{ridx}_safe"))
                    rows.append({
                        "id": f"pku_{split}_{idx}_{ridx}",
                        "source": "PKU-SafeRLHF",
                        "language": "English",
                        "user_query": prompt,
                        "target_model_answer": answer,
                        "gold_label": "safe" if safe else "unsafe",
                        "label_provenance": "PKU-SafeRLHF is_response_i_safe",
                        "source_prompt_id": f"pku_{split}_{sha_text(prompt)[:20]}",
                        "metadata": {
                            "official_split": split,
                            "source_row_index": idx,
                            "response_index": ridx,
                            "is_response_safe": safe,
                            "harm_category": item.get(f"response_{ridx}_harm_category"),
                            "harm_category_union": item.get(f"response_0_harm_category") or item.get(f"response_1_harm_category"),
                            "source_label_field": f"is_response_{ridx}_safe",
                        },
                    })
    except Exception as exc:  # noqa: BLE001
        audit["failures"].append({"source": "PKU-SafeRLHF", "error": f"{exc.__class__.__name__}: {exc}"})
    return rows


def load_beaver_rows(config: dict, audit: dict) -> list[dict]:
    rows = []
    try:
        from datasets import load_dataset

        for split in ("330k_train", "30k_test"):
            ds = load_dataset("PKU-Alignment/BeaverTails", split=split, revision=config["data"]["public_revisions"].get("beavertails", "main"))
            audit["source_files"][f"BeaverTails:{split}"] = {"dataset": "PKU-Alignment/BeaverTails", "revision": config["data"]["public_revisions"].get("beavertails", "main"), "rows_raw": len(ds)}
            for idx, item in enumerate(ds):
                q = normalize_ws(item.get("prompt"))
                y = normalize_ws(item.get("response"))
                if not q or not y:
                    continue
                safe = bool(item.get("is_safe"))
                rows.append({
                    "id": f"beaver_{split}_{idx}",
                    "source": "BeaverTails",
                    "language": "English",
                    "user_query": q,
                    "target_model_answer": y,
                    "gold_label": "safe" if safe else "unsafe",
                    "label_provenance": "BeaverTails is_safe response label",
                    "source_prompt_id": f"beaver_{split}_{sha_text(q)[:20]}",
                    "metadata": {"official_split": split, "source_row_index": idx, "category": item.get("category"), "is_safe": safe, "source_label_field": "is_safe"},
                })
    except Exception as exc:  # noqa: BLE001
        audit["failures"].append({"source": "BeaverTails", "error": f"{exc.__class__.__name__}: {exc}"})
    return rows


def load_aegis_rows(audit: dict) -> list[dict]:
    rows = []
    path = ROOT / "data/prepared/full/evaluation_qy/aegis_qy.jsonl"
    if path.exists():
        audit["source_files"]["Aegis:prepared_qy"] = {"path": str(path), "sha256": sha_file(path)}
        for raw in read_jsonl(path):
            meta = raw.get("metadata") or {}
            response_label = str(meta.get("response_label") or "").lower()
            if response_label not in {"safe", "unsafe"}:
                continue
            item = dict(raw)
            item["gold_label"] = response_label
            item["label_provenance"] = f"Aegis response_label:{meta.get('response_label_source') or 'unknown'}"
            rows.append(item)
    else:
        audit["failures"].append({"source": "Aegis", "error": "missing prepared aegis_qy.jsonl"})
    return rows


def build_r4_manifests(config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["data"]["seed"])
    sources, source_audit = load_public_sources(config)
    panel_a_anchor = select_panel_a(sources, config, seed, "anchor")
    panel_a_dev = select_panel_a(sources, config, seed, "model_dev", exclude={r["id"] for r in panel_a_anchor})
    bq_anchor, bq_dev, bq_audit = select_exact_q_pairs(sources, config, seed)
    by_anchor, by_dev, by_audit = select_near_y_pairs([r for rows in sources.values() for r in rows], config, seed)
    c_anchor = load_panel_c(output_dir, "anchor")
    c_dev = load_panel_c(output_dir, "model_dev")
    anchor = assign_panel(panel_a_anchor, "A") + assign_panel(bq_anchor, "Bq") + assign_panel(by_anchor, "By") + assign_panel(c_anchor, "C")
    dev = assign_panel(panel_a_dev, "A") + assign_panel(bq_dev, "Bq") + assign_panel(by_dev, "By") + assign_panel(c_dev, "C")
    for name, rows in (("anchor1200", anchor), ("model_dev360", dev)):
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    audits = write_audits(output_dir, config, sources, source_audit, anchor, dev, bq_audit, by_audit)
    passed = audits["g0"]["passed"]
    return {"passed": passed, **audits}


def select_panel_a(sources: dict[str, list[dict]], config: dict, seed: int, split: str, exclude: set[str] | None = None) -> list[dict]:
    per_source = 120 if split == "anchor" else 40
    per_label = per_source // 2
    out = []
    exclude = exclude or set()
    for source, rows in sources.items():
        for label in ("safe", "unsafe"):
            choices = [r for r in rows if r["gold_label"] == label and r["id"] not in exclude]
            out.extend(sorted(choices, key=lambda r: stable_hash(seed, split, "A", source, label, r["id"]))[:per_label])
    return [dict(r, canonical_group_id=f"A_{r['source']}_{sha_text(r['source_prompt_id'])[:16]}") for r in out]


def select_exact_q_pairs(sources: dict[str, list[dict]], config: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    groups = []
    for source in ("PKU-SafeRLHF", "BeaverTails"):
        by_q = defaultdict(list)
        for row in sources.get(source, []):
            by_q[sha_text(row["user_query"])].append(row)
        source_groups = []
        for qhash, values in by_q.items():
            labels = Counter(r["gold_label"] for r in values)
            if labels.get("safe", 0) and labels.get("unsafe", 0):
                safe = sorted([r for r in values if r["gold_label"] == "safe"], key=lambda r: stable_hash(seed, "bq", r["id"]))[0]
                unsafe = sorted([r for r in values if r["gold_label"] == "unsafe"], key=lambda r: stable_hash(seed, "bq", r["id"]))[0]
                gid = f"Bq_{source}_{qhash[:18]}"
                source_groups.append([dict(safe, canonical_group_id=gid, pair_type="exact_q", q_similarity=1.0), dict(unsafe, canonical_group_id=gid, pair_type="exact_q", q_similarity=1.0)])
        source_groups = sorted(source_groups, key=lambda g: stable_hash(seed, "bq_group", g[0]["canonical_group_id"]))
        groups.extend(source_groups)
    groups = sorted(groups, key=lambda g: stable_hash(seed, "bq_all", g[0]["canonical_group_id"]))
    anchor = [r for g in groups[:150] for r in g]
    dev = [r for g in groups[150:195] for r in g]
    return anchor, dev, {"candidate_groups": len(groups), "anchor_groups": len(anchor) // 2, "model_dev_groups": len(dev) // 2}


def select_near_y_pairs(rows: list[dict], config: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    clean = [r for r in rows if 40 <= len(r["target_model_answer"]) <= 1200]
    if len(clean) < 2:
        return [], [], {"candidate_rows": len(clean), "candidate_groups": 0}
    texts = [answer_norm(r["target_model_answer"]) for r in clean]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
    mat = vectorizer.fit_transform(texts)
    groups = []
    used = set()
    threshold = float(config["data"]["near_y"]["char_tfidf_min"])
    for i, row in enumerate(clean):
        if row["id"] in used:
            continue
        sims = cosine_similarity(mat[i], mat).ravel()
        order = np.argsort(-sims)[:80]
        for j in order:
            if i == j:
                continue
            other = clean[int(j)]
            if other["id"] in used or other["gold_label"] == row["gold_label"] or other["language"] != row["language"]:
                continue
            if sha_text(other["user_query"]) == sha_text(row["user_query"]):
                continue
            sim = float(sims[int(j)])
            jac = token_jaccard(row["target_model_answer"], other["target_model_answer"])
            ratio = length_ratio(row["target_model_answer"], other["target_model_answer"])
            if sim >= threshold and jac >= float(config["data"]["near_y"]["token_jaccard_min"]) and float(config["data"]["near_y"]["length_ratio_min"]) <= ratio <= float(config["data"]["near_y"]["length_ratio_max"]):
                gid = f"By_{sha_text(row['id'] + other['id'])[:18]}"
                left = dict(row, canonical_group_id=gid, pair_type="near_y_collision", y_similarity=sim, token_jaccard=jac, length_ratio=ratio, validator_status="cpu_candidate")
                right = dict(other, canonical_group_id=gid, pair_type="near_y_collision", y_similarity=sim, token_jaccard=jac, length_ratio=ratio, validator_status="cpu_candidate")
                groups.append([left, right])
                used.add(row["id"])
                used.add(other["id"])
                break
    groups = sorted(groups, key=lambda g: stable_hash(seed, "by_group", g[0]["canonical_group_id"]))
    anchor = [r for g in groups[:150] for r in g]
    dev = [r for g in groups[150:195] for r in g]
    return anchor, dev, {"candidate_rows": len(clean), "candidate_groups": len(groups), "anchor_groups": len(anchor) // 2, "model_dev_groups": len(dev) // 2}


def answer_norm(text: str) -> str:
    text = normalize_ws(text).lower()
    for term in ("i'm sorry", "i am sorry", "as an ai", "抱歉"):
        text = text.replace(term, "")
    return text


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / max(len(ta | tb), 1)


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return min(la, lb) / max(max(la, lb), 1)


def load_panel_c(output_dir: Path, split: str) -> list[dict]:
    path = output_dir / f"panel_c_{split}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


def assign_panel(rows: Iterable[dict], panel: str) -> list[dict]:
    return [dict(row, panel=panel) for row in rows]


def write_audits(output_dir: Path, config: dict, sources: dict[str, list[dict]], source_audit: dict, anchor: list[dict], dev: list[dict], bq_audit: dict, by_audit: dict) -> dict:
    source_lock = {
        "protocol": config["experiment"]["protocol"],
        "git_commit": git_commit(),
        "source_audit": source_audit,
        "config_sha256": sha_text(json.dumps(config, ensure_ascii=False, sort_keys=True)),
    }
    panel_census_rows = census_rows(anchor, "anchor") + census_rows(dev, "model_dev")
    write_json(output_dir / "E1_R4_SOURCE_LOCK.json", source_lock)
    write_csv(output_dir / "E1_R4_PANEL_CENSUS.csv", panel_census_rows)
    write_csv(output_dir / "E1_R4_LABEL_PROVENANCE.csv", provenance_rows(anchor + dev))
    fraud_scope = {"passed": all(fraud_scope_filter(r) for r in anchor + dev if r.get("panel") in {"A", "Bq", "By"}), "rows_checked": len([r for r in anchor + dev if r.get("panel") in {"A", "Bq", "By"}])}
    pair_audit = pair_construct_audit(anchor, dev, bq_audit, by_audit)
    split_leakage = split_leakage_audit(anchor, dev)
    protocol = {"protocol": config["experiment"]["protocol"], "commit": git_commit(), "config_sha256": source_lock["config_sha256"], "anchor_sha256": sha_file(output_dir / "anchor1200.jsonl"), "model_dev_sha256": sha_file(output_dir / "model_dev360.jsonl")}
    write_json(output_dir / "E1_R4_FRAUD_SCOPE_AUDIT.json", fraud_scope)
    write_json(output_dir / "E1_R4_PAIR_CONSTRUCT_AUDIT.json", pair_audit)
    write_json(output_dir / "E1_R4_SPLIT_LEAKAGE_AUDIT.json", split_leakage)
    write_json(output_dir / "E1_R4_PROTOCOL_LOCK.json", protocol)
    g0_checks = {
        "panel_a_anchor_360": count_panel(anchor, "A") == 360,
        "panel_a_sources_120_60_60": panel_a_balance(anchor),
        "bq_anchor_150_groups": bq_audit["anchor_groups"] == 150,
        "by_anchor_150_groups": by_audit["anchor_groups"] == 150,
        "panel_c_anchor_120_groups": count_panel(anchor, "C") == 240 and exact_pair_count(anchor, "C") == 120,
        "model_dev_360": len(dev) == 360,
        "anchor_1200": len(anchor) == 1200,
        "fraud_scope": fraud_scope["passed"],
        "split_leakage": split_leakage["passed"],
        "unknown_provenance": all(r.get("label_provenance") and "unknown" not in str(r.get("label_provenance")).lower() for r in anchor + dev),
    }
    g0 = {"passed": all(g0_checks.values()), "checks": g0_checks, "anchor_rows": len(anchor), "model_dev_rows": len(dev), "bq_audit": bq_audit, "by_audit": by_audit}
    write_json(output_dir / "E1_R4_G0_AUDIT.json", g0)
    return {"source_lock": source_lock, "fraud_scope": fraud_scope, "pair_construct": pair_audit, "split_leakage": split_leakage, "protocol_lock": protocol, "g0": g0}


def count_panel(rows: list[dict], panel: str) -> int:
    return sum(1 for r in rows if r.get("panel") == panel)


def exact_pair_count(rows: list[dict], panel: str) -> int:
    return len({r.get("canonical_group_id") for r in rows if r.get("panel") == panel})


def panel_a_balance(rows: list[dict]) -> bool:
    counts = Counter((r.get("source"), r.get("gold_label")) for r in rows if r.get("panel") == "A")
    return all(counts.get((source, label), 0) == 60 for source in ("PKU-SafeRLHF", "BeaverTails", "Aegis") for label in ("safe", "unsafe"))


def pair_construct_audit(anchor: list[dict], dev: list[dict], bq_audit: dict, by_audit: dict) -> dict:
    checks = {
        "bq_exact_q_opposite": grouped_pair_check(anchor + dev, "Bq", exact_q=True),
        "by_near_y_opposite": grouped_pair_check(anchor + dev, "By", near_y=True),
        "c_exact_q_opposite": grouped_pair_check(anchor + dev, "C", exact_q=True),
        "bq_capacity": bq_audit["anchor_groups"] >= 150 and bq_audit["model_dev_groups"] >= 45,
        "by_capacity": by_audit["anchor_groups"] >= 150 and by_audit["model_dev_groups"] >= 45,
    }
    return {"passed": all(checks.values()), "checks": checks, "bq": bq_audit, "by": by_audit}


def grouped_pair_check(rows: list[dict], panel: str, exact_q: bool = False, near_y: bool = False) -> bool:
    groups = defaultdict(list)
    for row in rows:
        if row.get("panel") == panel:
            groups[row.get("canonical_group_id")].append(row)
    if not groups:
        return False
    for vals in groups.values():
        if len(vals) != 2 or Counter(v["gold_label"] for v in vals) != Counter({"safe": 1, "unsafe": 1}):
            return False
        if exact_q and len({sha_text(v["user_query"]) for v in vals}) != 1:
            return False
        if near_y and min(float(v.get("y_similarity", 0.0)) for v in vals) < 0.88:
            return False
    return True


def split_leakage_audit(anchor: list[dict], dev: list[dict]) -> dict:
    dev_groups = {r.get("canonical_group_id") for r in dev}
    anchor_groups = {r.get("canonical_group_id") for r in anchor}
    overlap = sorted(dev_groups & anchor_groups)
    return {"passed": not overlap, "overlap_groups": overlap[:20], "overlap_count": len(overlap)}


def census_rows(rows: list[dict], split: str) -> list[dict]:
    counts = Counter((r.get("panel"), r.get("source"), r.get("language"), r.get("gold_risk_type"), r.get("gold_label")) for r in rows)
    return [{"split": split, "panel": p, "source": s, "language": l, "risk_type": rt, "label": lab, "count": c} for (p, s, l, rt, lab), c in sorted(counts.items())]


def provenance_rows(rows: list[dict]) -> list[dict]:
    counts = Counter((r.get("panel"), r.get("source"), r.get("label_provenance"), r.get("metadata", {}).get("source_label_field"), r.get("gold_label")) for r in rows)
    return [{"panel": p, "source": s, "label_provenance": lp, "source_label_field": f, "label": lab, "count": c} for (p, s, lp, f, lab), c in sorted(counts.items())]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception:
        return "unknown"
