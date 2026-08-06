"""CLI validation for the skills tree (guide section 33.2).

Usage:
    python -m frauddistill.skills.validation --skills-root skills --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frauddistill.skills.registry import SkillRegistry, check_expected_skills


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills-root", type=str, default="skills")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    registry = SkillRegistry(Path(args.skills_root))
    try:
        registry.discover()
    except ValueError as exc:
        print(f"[skills] ERROR: {exc}")
        return 1

    missing = check_expected_skills(registry)
    errors: list[str] = []
    for name in sorted(registry.descriptions()):
        record = registry.get(name)
        if not record.description.strip():
            errors.append(f"{name}: empty description")
        if record.compatibility and "instruction-only" not in record.compatibility:
            errors.append(f"{name}: compatibility must declare instruction-only")
        if record.char_count <= 0:
            errors.append(f"{name}: empty body")
        if any(bad in record.body for bad in ("```sh", "```bash", "subprocess", "os.system")):
            errors.append(f"{name}: executable dependency detected")

    print(f"[skills] {len(registry)} skills discovered")
    print(f"[skills] {len(registry)} valid")
    print(f"[skills] {len(missing)} missing expected: {missing}")
    for err in errors:
        print(f"[skills] ERROR: {err}")
    if args.strict and (missing or errors):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
