# -*- coding: utf-8 -*-
"""One-command dry-run/GPU orchestration for the LightEval supplement."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.run_all import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
