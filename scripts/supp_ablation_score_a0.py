# -*- coding: utf-8 -*-
"""A0: untrained DeepSeek-1.5B valid category interface on frozen E6."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.a0 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
