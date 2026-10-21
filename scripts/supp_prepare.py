# -*- coding: utf-8 -*-
"""Offline preparation for both LightEval supplementary experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.prepare import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
