"""Conditional report narrative (guide section 16).

Static templates must derive from the actual deltas/CIs; a negative delta
must never produce a positive claim, and "executed" is separate from
"target_met"/"paper_ready".
"""
from __future__ import annotations


def comparison_statement(delta: float, ci_low: float, ci_high: float) -> str:
    """Conditional narrative for one paired comparison (guide 16.2)."""
    if ci_low > 0:
        return "FraudDistill 显著优于基线。"
    if ci_high < 0:
        return "FraudDistill 显著低于基线。"
    if delta > 0:
        return "FraudDistill 点估计较高，但差异不显著。"
    if delta < 0:
        return "FraudDistill 点估计较低，但差异不显著。"
    return "两种方法点估计相同。"


def acceptance_status(*, executed: bool, integrity_valid: bool, target_met: bool) -> dict:
    """Separate execution/validity/target states (guide 16.3)."""
    return {
        "executed": executed,
        "integrity_valid": integrity_valid,
        "target_met": target_met,
        "paper_ready": bool(executed and integrity_valid and target_met),
    }


def target_modeled_as_positive(auprc: float, threshold: float) -> bool:
    """Used by reports to check whether the head clears the offline gate."""
    return auprc >= threshold
