from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResourceProfiler:
    root: Path
    started: float = field(default_factory=time.perf_counter)
    peak_rss_mb: float = 0.0

    def sample(self) -> float:
        rss = current_rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        return rss

    def finish(self) -> dict:
        self.sample()
        cuda_available, device_count = cuda_state()
        artifact_mb = dir_size_mb(self.root)
        return {
            "wall_seconds": time.perf_counter() - self.started,
            "peak_rss_mb": self.peak_rss_mb,
            "artifact_mb": artifact_mb,
            "cpu_threads": os.cpu_count() or 1,
            "cuda_available": cuda_available,
            "device_count": device_count,
        }


def resource_gate(profile: dict, policy: dict) -> dict:
    checks = {
        "cpu_only": (not profile.get("cuda_available")) if policy.get("cpu_only", True) else True,
        "peak_ram_mb": float(profile.get("peak_rss_mb", 0.0)) <= float(policy.get("peak_ram_mb_max", 8192)),
        "wall_time": float(profile.get("wall_seconds", 0.0)) <= float(policy.get("g0_wall_time_minutes_max", 90)) * 60,
        "artifact_mb": float(profile.get("artifact_mb", 0.0)) <= float(policy.get("artifact_mb_max", 500)),
    }
    return {"passed": all(checks.values()), "checks": checks, "profile": profile}


def current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def cuda_state() -> tuple[bool, int]:
    try:
        import torch

        return bool(torch.cuda.is_available()), int(torch.cuda.device_count())
    except Exception:
        return False, 0


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)
