# -*- coding: utf-8 -*-
"""Evaluate A0-A6 and generate nested audit statistics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.ablation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
