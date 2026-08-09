# -*- coding: utf-8 -*-
"""E5 P2 selective policy (dual threshold) + P3 cascade helpers + budget ledger."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import clopper_pearson_ucb


def fit_dual_threshold(score: np.ndarray, y: np.ndarray, target_api_rate: float = 0.15,
                       max_cal_fnr_auto_safe: float = 0.05,
                       max_cal_fpr_auto_unsafe: float = 0.05,
                       alpha: float = 0.05) -> dict:
    """Choose (tau_low, tau_high): auto-safe below tau_low with controlled FNR;
    auto-unsafe above tau_high with controlled FPR; abstain in between.
    Maximize coverage (1 - abstain rate) subject to constraints."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    cands = np.unique(np.concatenate([s, [0.0, 1.0]]))
    cands = cands[cands >= 0.0]
    best = None
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    for tau_lo in cands:
        for tau_hi in cands:
            if tau_hi <= tau_lo:
                continue
            auto_safe = s < tau_lo
            auto_unsafe = s >= tau_hi
            abstain = ~auto_safe & ~auto_unsafe
            fn_as = int((auto_safe & (y == 1)).sum())
            fp_au = int((auto_unsafe & (y == 0)).sum())
            fnr_ucb = clopper_pearson_ucb(fn_as, max(n_pos, 1), alpha)
            fpr_ucb = clopper_pearson_ucb(fp_au, max(n_neg, 1), alpha)
            if fnr_ucb > max_cal_fnr_auto_safe or fpr_ucb > max_cal_fpr_auto_unsafe:
                continue
            coverage = float((~abstain).mean())
            api_rate = float(abstain.mean())
            if api_rate > target_api_rate:
                continue
            cand = {"tau_low": float(tau_lo), "tau_high": float(tau_hi),
                    "coverage": coverage, "api_rate": api_rate,
                    "fnr_ucb_auto_safe": fnr_ucb, "fpr_ucb_auto_unsafe": fpr_ucb,
                    "n_abstain": int(abstain.sum()), "n_pos": n_pos, "n_neg": n_neg}
            if best is None or cand["coverage"] > best["coverage"]:
                best = cand
    return best


def apply_dual_threshold(score: np.ndarray, policy: dict) -> np.ndarray:
    """Returns decision: 0=auto-safe, 1=auto-unsafe, -1=abstain."""
    s = np.asarray(score, dtype=float)
    out = np.full(len(s), -1, dtype=int)
    out[s < policy["tau_low"]] = 0
    out[s >= policy["tau_high"]] = 1
    return out


class BudgetLedger:
    def __init__(self, path: Path, hard_stop_cny: float = 10.0, soft_stop_cny: float = 8.0):
        self.path = Path(path)
        self.hard_stop_cny = hard_stop_cny
        self.soft_stop_cny = soft_stop_cny
        self.rows = []
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                try:
                    self.rows.append(json.loads(line))
                except Exception:
                    pass

    def spent(self) -> float:
        return sum(float(r.get("estimated_cost_cny", 0.0)) for r in self.rows)

    def can_call(self, estimated_cny: float, retry_reserve: float = 0.0,
                 allow_soft_optional: bool = False) -> tuple[bool, str]:
        projected = self.spent() + estimated_cny + retry_reserve
        if projected >= self.hard_stop_cny:
            return False, f"hard_stop: projected {projected:.2f} >= {self.hard_stop_cny}"
        if projected >= self.soft_stop_cny and not allow_soft_optional:
            return False, f"soft_stop: projected {projected:.2f} >= {self.soft_stop_cny}"
        return True, "ok"

    def record(self, sample_id: str, qy_hash: str, phase: str, provider: str,
               model_snapshot: str, input_tokens: int, output_tokens: int,
               estimated_cost_cny: float, retry_count: int = 0, cache_hit: bool = False) -> dict:
        row = {
            "request_id": f"req_{len(self.rows) + 1}",
            "sample_id": sample_id,
            "qy_hash": qy_hash,
            "phase": phase,
            "provider": provider,
            "model_snapshot": model_snapshot,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_cny": round(float(estimated_cost_cny), 6),
            "retry_count": retry_count,
            "cache_hit": cache_hit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def state(self) -> dict:
        return {"spent_cny": round(self.spent(), 6), "n_calls": len(self.rows),
                "hard_stop_cny": self.hard_stop_cny, "soft_stop_cny": self.soft_stop_cny}
