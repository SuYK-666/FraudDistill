"""Replay A0-A6 from frozen predictions and nested audit outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .common import (REPO, binary_label, binary_metrics, family_bootstrap_delta,
                     load_score_map, read_json, read_jsonl, resolve_repo, row_key, sha256_file,
                     write_json, write_jsonl)

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
PREP_DIR = SUPP_DIR / "prepared"
RESULT_DIR = SUPP_DIR / "results" / "ablation"
TEST_PATH = PREP_DIR / "raw" / "relation_test.jsonl"
AUDIT_RESULTS = SUPP_DIR / "audit" / "audit_results.jsonl"
CONFIG = REPO / "configs/supplementary/lighteval_ablation.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _valid_label(row: dict) -> str | None:
    label = row.get("audit_label")
    return label if label in {"safe", "unsafe"} else None


def _audit_cost(row: dict) -> tuple[float, float, float]:
    """Return incremental API cost, cached original cost, and online latency."""
    status = row.get("audit_status")
    cost = float(row.get("estimated_cost_cny", 0.0) or 0.0)
    if status in {"ok", "new_live_call"}:
        return cost, 0.0, float(row.get("latency_ms", 0.0) or 0.0)
    if "cache" in str(status):
        return 0.0, float(row.get("cached_original_cost_cny", cost) or 0.0), 0.0
    return 0.0, 0.0, 0.0


def evaluate_ablation(a0_predictions: Path | None = None, bootstrap_replicates: int = 10000,
                      seed: int = 20260811, allow_missing_audit: bool = True,
                      audit_results_path: Path | None = None,
                      result_dir: Path | None = None) -> dict:
    cfg = _config()
    result_dir = Path(result_dir) if result_dir is not None else RESULT_DIR
    lock_path = PREP_DIR / "asset_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("run scripts/supp_prepare.py before evaluating the ablation")
    lock = read_json(lock_path)
    if lock["configs"]["ablation"]["sha256"] != sha256_file(CONFIG):
        raise ValueError("ablation config changed after preparation; rerun scripts/supp_prepare.py")
    if lock["prepared"]["raw_files"]["relation_test"]["sha256"] != sha256_file(TEST_PATH):
        raise ValueError("frozen E6 test manifest changed after preparation")
    test_rows = read_jsonl(TEST_PATH)
    preds = load_score_map(resolve_repo(cfg["paths"]["student_predictions"]))
    selection = read_jsonl(PREP_DIR / "audit_selection_manifest.jsonl")
    audit_path = Path(audit_results_path) if audit_results_path is not None else AUDIT_RESULTS
    audit_rows = read_jsonl(audit_path)
    audit_by_id = {row["id"]: row for row in audit_rows}
    source_t = float(cfg["threshold"]["source"])
    target_t = float(cfg["threshold"]["target"])
    missing = [row_key(row) for row in test_rows if row_key(row) not in preds]
    if missing:
        raise ValueError(f"missing trained-student predictions: {len(missing)} rows")
    if not allow_missing_audit and any(row["id"] not in audit_by_id for row in selection):
        raise FileNotFoundError("full 76-item audit is incomplete; run scripts/supp_audit.py --execute")

    y = np.asarray([binary_label(row) for row in test_rows], dtype=int)
    families = [str(row.get("family_id") or row_key(row)) for row in test_rows]
    keys = [row_key(row) for row in test_rows]
    scores = np.asarray([float(preds[key]["risk_score"]) for key in keys], dtype=float)
    rank_by_id = {row["id"]: int(row["rank"]) for row in selection}
    audit_ranked = sorted(selection, key=lambda row: int(row["rank"]))
    for result in audit_rows:
        if result["id"] not in rank_by_id or int(result.get("rank", -1)) != rank_by_id[result["id"]]:
            raise ValueError(f"audit result rank/ID mismatch: {result.get('id')}")
    prefixes = cfg["audit"]["rates"]
    budgets = {rate: min(len(audit_ranked), math.floor(float(rate) * len(test_rows))) for rate in prefixes}

    policies: dict[str, dict] = {}
    for item in cfg["ablations"]:
        policy_id = item["id"]
        if policy_id == "A0":
            if a0_predictions is None or not Path(a0_predictions).exists():
                policies[policy_id] = {
                    "id": policy_id, "label": item["label"], "status": "pending_a0_predictions",
                    "training": False, "calibration": False, "audit_rate": 0.0,
                }
                continue
            a0 = load_score_map(a0_predictions)
            missing_a0 = [key for key in keys if key not in a0]
            if missing_a0:
                raise ValueError(f"A0 predictions missing {len(missing_a0)} test rows")
            final = np.asarray([1 if a0[key].get("label") == "unsafe" else 0 for key in keys], dtype=int)
            final_scores = np.asarray([float(a0[key].get("risk_score", 0.0)) for key in keys])
            m = binary_metrics(y, final_scores, final, threshold=0.5)
            policies[policy_id] = {
                "id": policy_id, "label": item["label"], "status": "ready",
                "training": False, "calibration": False, "audit_rate": 0.0,
                "threshold": 0.5, "audited": 0, "valid_audits": 0,
                "metrics": m, "final_pred": final.tolist(),
                "corrections": [], "incremental_api_cost_cny": 0.0,
                "cached_original_cost_cny": 0.0, "online_audit_latency_ms": 0.0,
            }
            continue

        threshold = target_t if item["calibration"] else source_t
        auto = (scores >= threshold).astype(int)
        final = auto.copy()
        rate = float(item["audit_rate"])
        expected_prefix = audit_ranked[:budgets[rate]] if rate > 0 else []
        prefix_complete = all(row["id"] in audit_by_id for row in expected_prefix)
        audited = 0
        correction_rows = []
        incremental_cost = cached_cost = online_latency = 0.0
        if rate > 0:
            k = budgets[rate]
            for selection_row in audit_ranked[:k]:
                task_id = selection_row["id"]
                idx = keys.index(task_id)
                audited += 1
                result = audit_by_id.get(task_id)
                if result is None:
                    correction_rows.append({"id": task_id, "status": "missing", "auto": int(auto[idx]),
                                            "final": int(auto[idx]), "gold": int(y[idx])})
                    continue
                inc, cache_orig, latency = _audit_cost(result)
                incremental_cost += inc
                cached_cost += cache_orig
                online_latency += latency
                label = _valid_label(result)
                if label is None:
                    final[idx] = auto[idx]
                    correction_rows.append({"id": task_id, "status": result.get("audit_status", "invalid"),
                                            "auto": int(auto[idx]), "final": int(auto[idx]),
                                            "gold": int(y[idx])})
                    continue
                audit_pred = 1 if label == "unsafe" else 0
                final[idx] = audit_pred
                correction_rows.append({
                    "id": task_id, "status": "valid", "audit_rank": int(result.get("rank", rank_by_id[task_id])),
                    "auto": int(auto[idx]), "final": audit_pred, "gold": int(y[idx]),
                    "fixed_error": bool(auto[idx] != y[idx] and audit_pred == y[idx]),
                    "introduced_error": bool(auto[idx] == y[idx] and audit_pred != y[idx]),
                    "unchanged": bool(auto[idx] == audit_pred),
                })

        m = binary_metrics(y, scores, final, threshold=threshold)
        valid = sum(_valid_label(audit_by_id.get(row["id"], {})) is not None
                    for row in audit_ranked[:audited])
        invalid = audited - valid
        fixed = sum(bool(row.get("fixed_error")) for row in correction_rows)
        introduced = sum(bool(row.get("introduced_error")) for row in correction_rows)
        policies[policy_id] = {
            "id": policy_id, "label": item["label"],
            "status": "ready" if prefix_complete else "pending_audit",
            "training": bool(item["training"]), "calibration": bool(item["calibration"]),
            "audit_rate": rate, "audit_budget_k": audited, "audited": audited,
            "valid_audits": valid, "invalid_audits": invalid,
            "threshold": threshold, "metrics": m, "final_pred": final.tolist(),
            "corrections": correction_rows,
            "fixed_errors": fixed, "introduced_errors": introduced,
            "net_corrections": fixed - introduced,
            "incremental_api_cost_cny": round(incremental_cost, 8),
            "cached_original_cost_cny": round(cached_cost, 8),
            "online_audit_latency_ms": round(online_latency, 2),
            "avg_online_latency_ms": round(online_latency / max(valid, 1), 2),
        }

    if "A2" in policies and policies["A2"].get("status") == "ready":
        a2_mf1 = policies["A2"]["metrics"]["macro_f1"]
        for policy in policies.values():
            if policy.get("status") != "ready":
                continue
            policy["delta_macro_f1_vs_A2"] = round(policy["metrics"]["macro_f1"] - a2_mf1, 6)
            if "final_pred" not in policy:
                policy["final_pred"] = (scores >= float(policy["threshold"])).astype(int).tolist()

    bootstrap = {}
    ready = {key: val for key, val in policies.items()
             if val.get("status") == "ready" and "final_pred" in val}
    if "A2" in ready:
        baseline = np.asarray(ready["A2"]["final_pred"], dtype=int)
        for policy_id in ("A1", "A3", "A4", "A5", "A6"):
            if policy_id not in ready:
                continue
            candidate = np.asarray(ready[policy_id]["final_pred"], dtype=int)
            bootstrap[f"{policy_id}_vs_A2"] = family_bootstrap_delta(
                y, candidate, baseline, families,
                replicates=bootstrap_replicates, seed=seed,
            )
    if "A0" in ready and "A1" in ready:
        bootstrap["A1_vs_A0"] = family_bootstrap_delta(
            y, np.asarray(ready["A1"]["final_pred"], dtype=int),
            np.asarray(ready["A0"]["final_pred"], dtype=int), families,
            replicates=bootstrap_replicates, seed=seed,
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_dir / "final_predictions_by_policy.jsonl", [
        {"id": key, "gold": int(y[i]), "family_id": families[i],
         **{policy_id: int(policy["final_pred"][i]) for policy_id, policy in ready.items()}}
        for i, key in enumerate(keys)
    ])
    summary = {
        "protocol": cfg["protocol"],
        "n": len(test_rows),
        "source_threshold": source_t,
        "target_threshold": target_t,
        "budgets": budgets,
        "policies": policies,
        "bootstrap": bootstrap,
        "notes": [
            "Audit rank is frozen once by |score-target_threshold|, then stable composite id.",
            "Every audit budget is a strict prefix of the 76-item list.",
            "Invalid or missing reviewer outputs retain the automatic calibrated decision.",
            "Cache hits incur zero incremental API cost and zero online review latency; original cache cost is reported separately.",
        ],
    }
    write_json(result_dir / "ablation_results.json", summary)
    write_json(result_dir / "bootstrap.json", bootstrap)
    rows = [
        "| Policy | Training | Calibration | Audit | MF1 | Recall | FPR | ΔMF1 vs A2 | Fixed | Introduced | Net | API cost CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cfg["ablations"]:
        policy = policies[item["id"]]
        if policy.get("status") != "ready":
            rows.append(f"| {item['id']} | {'Y' if item['training'] else '-'} | "
                        f"{'Y' if item['calibration'] else '-'} | {item['audit_rate']:.0%} | "
                        "NOT RUN | - | - | - | - | - | - | - |")
            continue
        m = policy["metrics"]
        rows.append("| {id} | {train} | {cal} | {rate:.0%} | {mf1:.4f} | {rec:.4f} | {fpr:.4f} | "
                    "{delta:+.4f} | {fixed} | {intro} | {net:+d} | {cost:.4f} |".format(
                        id=policy["id"], train="Y" if policy["training"] else "-",
                        cal="Y" if policy["calibration"] else "-", rate=policy["audit_rate"],
                        mf1=m["macro_f1"], rec=m["recall"] or 0.0, fpr=m["fpr"] or 0.0,
                        delta=policy.get("delta_macro_f1_vs_A2", 0.0),
                        fixed=policy.get("fixed_errors", 0), intro=policy.get("introduced_errors", 0),
                        net=policy.get("net_corrections", 0),
                        cost=policy.get("incremental_api_cost_cny", 0.0),
                    ))
    (result_dir / "ablation_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary





def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-predictions", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--require-audit", action="store_true")
    parser.add_argument("--audit-results", default=None)
    parser.add_argument("--result-dir", default=None)
    args = parser.parse_args(argv)
    summary = evaluate_ablation(Path(args.a0_predictions) if args.a0_predictions else None,
                                args.bootstrap_replicates, args.seed,
                                allow_missing_audit=not args.require_audit,
                                audit_results_path=Path(args.audit_results) if args.audit_results else None,
                                result_dir=Path(args.result_dir) if args.result_dir else None)
    print(json.dumps({
        "status": "READY",
        "output": str((Path(args.result_dir) if args.result_dir else RESULT_DIR) / "ablation_results.json"),
        "policy_status": {k: v.get("status") for k, v in summary["policies"].items()},
        "budgets": summary["budgets"],
        "bootstrap": sorted(summary["bootstrap"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
