"""Offline mode hard guard (guide section 4).

FRAUDDISTILL_OFFLINE=1 makes every provider/client constructor raise
OfflineNetworkCallError so that static-repair runs can never hit the network.
"""
from __future__ import annotations

import os


class OfflineNetworkCallError(RuntimeError):
    """Raised when a network/API call is attempted in offline mode."""


def offline_enabled() -> bool:
    return os.getenv("FRAUDDISTILL_OFFLINE", "0") == "1"


def assert_online_allowed() -> None:
    """Raise OfflineNetworkCallError when FRAUDDISTILL_OFFLINE=1."""
    if offline_enabled():
        raise OfflineNetworkCallError(
            "FRAUDDISTILL_OFFLINE=1: network/API calls are disabled."
        )


def require_offline() -> None:
    """Raise RuntimeError when offline mode is NOT enabled.

    Static-repair entry points must run with FRAUDDISTILL_OFFLINE=1 so that
    any accidentally-triggered provider call fails fast (guide 4.3).
    """
    if not offline_enabled():
        raise RuntimeError(
            "FRAUDDISTILL_OFFLINE is not set; static-repair scripts must run "
            "with --offline (guide 4.3)."
        )


def clear_api_keys() -> None:
    """Blank all known API key environment variables (guide 4.1)."""
    for name in (
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "GOOGLE_API_KEY",
        "ZHIPUAI_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        os.environ[name] = ""
