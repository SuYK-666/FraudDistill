# -*- coding: utf-8 -*-
"""Select, score, calibrate and cost-account one scale checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.scale_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
