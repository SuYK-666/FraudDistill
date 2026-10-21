# -*- coding: utf-8 -*-
"""Write a compact checkpoint/result record for one scale experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.record import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
