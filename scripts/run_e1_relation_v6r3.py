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

from frauddistill.exp1_ccfa.relation_manifest_v6r3 import write_relation_manifests_v6r3


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6r3.yaml"
TAXONOMY_PATH = ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml"
PREFIX = "E1_V6R3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 Relation-Gate v6r3")
    parser.add_argument("--stage", choices=["g0", "smoke", "pilot", "formal"], required=True)
    parser.add_argument("--manifest_dir", default="data/prepared/e1_relation_gate_v6r3")
    parser.add_argument("--output_dir", default="outputs/e1_relation_gate_v6r3")
    parser.add_argument("--allow_dirty_g0", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    out = ROOT / args.output_dir / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if args.stage != "g0":
        payload = stage_locked_decision(args.stage, ROOT / args.output_dir)
        write_json(out / f"{PREFIX}_DECISION.json", payload)
        write_stage_report(out / f"{PREFIX}_REPORT_CN.md", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(0 if payload["decision"].endswith("_PASS") else 2)
    census = write_relation_manifests_v6r3(ROOT / args.manifest_dir, config, TAXONOMY_PATH, int(config["data"]["seed"]), require_clean_git=not args.allow_dirty_g0)
    payload = {"decision": census["decision"], "stage": "g0", "git_commit": git_commit(), "git_status": git_status(), "census": census}
    write_json(out / f"{PREFIX}_DECISION.json", payload)
    write_g0_report(out / f"{PREFIX}_REPORT_CN.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    if not census.get("passed"):
        raise SystemExit(2)


def stage_locked_decision(stage: str, output_root: Path) -> dict:
    g0 = read_decision(output_root / "g0" / f"{PREFIX}_DECISION.json")
    if not g0 or g0.get("decision") != "E1_V6R3_G0_PASS":
        return {"decision": f"E1_V6R3_{stage.upper()}_LOCKED", "stage": stage, "reason": "G0 PASS is required before this stage"}
    if stage == "smoke":
        return {"decision": "E1_V6R3_SMOKE_NOT_IMPLEMENTED", "stage": stage, "reason": "G0 passed, but model smoke runner is not implemented in this commit"}
    smoke = read_decision(output_root / "smoke" / f"{PREFIX}_DECISION.json")
    if stage == "pilot" and (not smoke or smoke.get("decision") != "E1_V6R3_SMOKE_PASS"):
        return {"decision": "E1_V6R3_PILOT_LOCKED", "stage": stage, "reason": "SMOKE PASS is required before Pilot"}
    pilot = read_decision(output_root / "pilot" / f"{PREFIX}_DECISION.json")
    if stage == "formal" and (not pilot or pilot.get("decision") != "E1_V6R3_FULL_READY"):
        return {"decision": "E1_V6R3_FORMAL_LOCKED", "stage": stage, "reason": "Pilot FULL_READY is required before Formal"}
    return {"decision": f"E1_V6R3_{stage.upper()}_NOT_IMPLEMENTED", "stage": stage}


def write_g0_report(path: Path, payload: dict) -> None:
    census = payload["census"]
    false_checks = [key for key, value in census.get("checks", {}).items() if not value]
    r2 = census.get("r2_audit", {})
    r3 = census.get("r3_audit", {})
    lines = [
        "# FraudDistill E1 v6r3 G0r3 报告",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- Git commit：`{payload['git_commit']}`",
        "",
        "## 核心容量",
        "",
        f"- R1 groups：{census.get('r1_groups')}",
        f"- R2 true max matching groups：{r2.get('max_matching_groups')}",
        f"- R2 selected groups：{r2.get('selected_groups')}",
        f"- R3 max balanced rows：{r3.get('max_balanced_r3_rows')}",
        f"- R3 selected rows：{census.get('r3_rows')}",
        "",
        "## 未通过 Checks",
        "",
    ]
    lines.extend([f"- `{check}`" for check in false_checks] or ["- 无"])
    lines.extend(["", "## 结论", "", "G0r3 是硬准入门。若本阶段 STOP，则不运行 smoke/Pilot/Formal。"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stage_report(path: Path, payload: dict) -> None:
    lines = ["# FraudDistill E1 v6r3 Stage 报告", "", f"- 决策：`{payload['decision']}`", f"- 阶段：`{payload['stage']}`", f"- 原因：{payload.get('reason', '')}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_decision(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
