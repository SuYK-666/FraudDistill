from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_ccfa.relation_manifest import write_relation_manifests


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6.yaml"
TAXONOMY_PATH = ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FraudDistill E1 Relation-Gate v6 manifests")
    parser.add_argument("--output_dir", default="data/prepared/e1_relation_gate_v6")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    census = write_relation_manifests(ROOT / args.output_dir, config, TAXONOMY_PATH, args.seed)
    print(json.dumps(census, ensure_ascii=False, indent=2))
    if not census["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
