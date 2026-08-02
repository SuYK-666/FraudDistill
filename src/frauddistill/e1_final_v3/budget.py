from __future__ import annotations

from typing import Any


def budget_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pricing_sources": [
            {
                "provider": "qwen",
                "url": "https://help.aliyun.com/zh/model-studio/model-pricing",
                "snapshot_note": "Official Alibaba Cloud Model Studio pricing page checked before v3 implementation.",
            },
            {
                "provider": "deepseek",
                "url": "https://api-docs.deepseek.com/quick_start/pricing/",
                "snapshot_note": "Official DeepSeek API pricing page recorded as required protocol source.",
            },
        ],
        "hard_limits_cny": config["budget"],
        "effective_concurrency": {
            "user_requested_total": config["api"]["user_requested_total_concurrency"],
            "qwen": config["api"]["effective_qwen_concurrency"],
            "deepseek": config["api"]["effective_deepseek_concurrency"],
            "adjudicator": config["api"]["effective_adjudicator_concurrency"],
            "reason": "v3 protocol forbids using total concurrency 120 as default; provider-level caps are enforced.",
        },
        "ledger_policy": {"check_every_calls": config["api"]["budget_check_interval_calls"], "hard_stop_required": True},
    }


def hard_stop_decision(ledger: list[dict[str, Any]], limits: dict[str, Any]) -> dict[str, Any]:
    total = sum(float(r.get("cost_cny", 0) or 0) for r in ledger)
    by_provider: dict[str, float] = {}
    for row in ledger:
        provider = str(row.get("provider", "unknown"))
        by_provider[provider] = by_provider.get(provider, 0.0) + float(row.get("cost_cny", 0) or 0)
    total_limit = float(limits.get("hard_stop_total_cny", 0) or 0)
    qwen_limit = float(limits.get("qwen_hard_stop_cny", 0) or 0)
    deepseek_limit = float(limits.get("deepseek_hard_stop_cny", 0) or 0)
    hard_stop = (total_limit > 0 and total >= total_limit) or (qwen_limit > 0 and by_provider.get("qwen", 0.0) >= qwen_limit) or (deepseek_limit > 0 and by_provider.get("deepseek", 0.0) >= deepseek_limit)
    return {
        "total_cny": total,
        "by_provider": by_provider,
        "hard_stop": hard_stop,
    }
