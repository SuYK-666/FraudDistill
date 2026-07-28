from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_ccfa.relation_manifest import write_relation_manifests_v6r2


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6r2.yaml"
TAXONOMY_PATH = ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 Relation-Gate v6r2")
    parser.add_argument("--stage", choices=["g0", "smoke", "pilot", "formal"], required=True)
    parser.add_argument("--manifest_dir", default="data/prepared/e1_relation_gate_v6r2")
    parser.add_argument("--output_dir", default="outputs/e1_relation_gate_v6r2")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.stage != "g0":
        payload = {"decision": "E1_V6R2_STAGE_LOCKED", "stage": args.stage, "reason": "non-G0 stages are locked until G0r2 PASS is present"}
        out = ROOT / args.output_dir / args.stage
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "E1_V6R2_DECISION.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    census = write_relation_manifests_v6r2(ROOT / args.manifest_dir, config, TAXONOMY_PATH, int(config["data"]["seed"]))
    out = ROOT / args.output_dir / "g0"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": "E1_V6R2_G0R2_PASS" if census.get("passed") else "E1_V6R2_STOP",
        "stage": "g0",
        "git_commit": git_commit(),
        "git_status": git_status(),
        "census": census,
    }
    write_json(out / "E1_V6R2_DECISION.json", payload)
    write_g0_report(out / "E1_V6R2_REPORT_CN.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    if not census.get("passed"):
        raise SystemExit(2)


def write_g0_report(path: Path, payload: dict) -> None:
    census = payload["census"]
    r2 = census.get("r2_audit", {})
    false_checks = [key for key, value in census.get("checks", {}).items() if not value]
    lines = [
        "# FraudDistill E1 v6r2 G0r2 报告",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- Git commit：`{payload['git_commit']}`",
        "",
        "## 数据容量",
        "",
        f"- R1 groups：{census.get('r1_groups')}",
        f"- R2 selected groups：{census.get('r2_groups')}",
        f"- R2 max matching groups：{r2.get('max_matching_groups')}",
        f"- R3 rows：{census.get('r3_rows')}",
        "",
        "## R2 审计",
        "",
        f"- edge_count：{r2.get('edge_count')}",
        f"- q SMD：{r2.get('q_selector_smd')}",
        f"- y SMD：{r2.get('y_selector_smd')}",
        f"- length SMD：{r2.get('log_answer_length_smd')}",
        f"- refusal gap：{r2.get('refusal_gap')}",
        f"- independent q AUROC：{r2.get('independent_q_probe_auc')}",
        f"- independent y AUROC：{r2.get('independent_y_probe_auc')}",
        f"- largest row source：{r2.get('largest_source_rate')}",
        f"- largest source-pair：{r2.get('largest_source_pair_rate')}",
        f"- cross-source group rate：{r2.get('cross_source_group_rate')}",
        f"- third-source share：{r2.get('third_source_share')}",
        "",
        "## 未通过 Checks",
        "",
    ]
    lines.extend([f"- `{check}`" for check in false_checks] or ["- 无"])
    lines.extend(["", "## 结论", "", "若本阶段 STOP，则不运行 smoke/Pilot/Formal；这符合 v6r2 的 Gate 链要求。"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
