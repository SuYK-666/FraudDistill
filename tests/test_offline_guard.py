# -*- coding: utf-8 -*-
"""Offline-mode guard tests (guide 28.8, section 4)."""
import os

import pytest

from frauddistill.exp2_static_repair.offline_guard import (
    OfflineNetworkCallError,
    assert_online_allowed,
    offline_enabled,
    require_offline,
)


def test_offline_mode_blocks_provider(monkeypatch):
    monkeypatch.setenv("FRAUDDISTILL_OFFLINE", "1")
    with pytest.raises(OfflineNetworkCallError):
        assert_online_allowed()


def test_online_mode_allows():
    os.environ["FRAUDDISTILL_OFFLINE"] = "0"
    assert_online_allowed()  # must not raise


def test_require_offline_raises_when_not_offline(monkeypatch):
    monkeypatch.delenv("FRAUDDISTILL_OFFLINE", raising=False)
    with pytest.raises(RuntimeError):
        require_offline()


def test_require_offline_passes_when_offline(monkeypatch):
    monkeypatch.setenv("FRAUDDISTILL_OFFLINE", "1")
    require_offline()  # must not raise


def test_offline_enabled_flag():
    os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    assert offline_enabled() is True
    os.environ["FRAUDDISTILL_OFFLINE"] = "0"
    assert offline_enabled() is False
