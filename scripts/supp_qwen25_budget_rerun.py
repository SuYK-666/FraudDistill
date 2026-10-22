# -*- coding: utf-8 -*-
"""Serialize the Qwen2.5 rerun, audit, cost accounting, and final report."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.qwen25_budget_rerun import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
