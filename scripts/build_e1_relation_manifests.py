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

from frauddistill.exp1_ccfa.relation_manifest import write_relation_manifests, write_relation_manifests_v6r1


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6.yaml"
TAXONOMY_PATH = ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FraudDistill E1 Relation-Gate v6 manifests")
    parser.add_argument("--output_dir", default="data/prepared/e1_relation_gate_v6")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--protocol", choices=["v6", "v6r1", "v6r2", "v6r3"], default="v6")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.protocol == "v6r3":
        from frauddistill.exp1_ccfa.relation_manifest_v6r3 import write_relation_manifests_v6r3

        census = write_relation_manifests_v6r3(ROOT / args.output_dir, config, TAXONOMY_PATH, args.seed, require_clean_git=False)
    elif args.protocol == "v6r2":
        from frauddistill.exp1_ccfa.relation_manifest import write_relation_manifests_v6r2

        census = write_relation_manifests_v6r2(ROOT / args.output_dir, config, TAXONOMY_PATH, args.seed)
    elif args.protocol == "v6r1":
        census = write_relation_manifests_v6r1(ROOT / args.output_dir, config, TAXONOMY_PATH, args.seed)
    else:
        census = write_relation_manifests(ROOT / args.output_dir, config, TAXONOMY_PATH, args.seed)
    print(json.dumps(census, ensure_ascii=False, indent=2))
    if not census["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
