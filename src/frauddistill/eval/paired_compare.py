from __future__ import annotations

import math
from typing import Any


def mcnemar_exact(y_true: list[str], left: list[str], right: list[str]) -> dict[str, Any]:
    b = sum(a == truth and c != truth for truth, a, c in zip(y_true, left, right))
    c = sum(a != truth and c == truth for truth, a, c in zip(y_true, left, right))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return {"left_only_correct": b, "right_only_correct": c, "n_discordant": n, "p_value": p}
