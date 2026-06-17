from __future__ import annotations


def cost_record(method: str, dataset: str, num_samples: int, total_time_seconds: float, llm_calls_per_sample: int = 0, gpu_memory_gb: float | None = None) -> dict:
    return {
        "method": method,
        "dataset": dataset,
        "num_samples": num_samples,
        "total_time_seconds": total_time_seconds,
        "avg_latency_ms": (total_time_seconds / max(num_samples, 1)) * 1000.0,
        "gpu_memory_gb": gpu_memory_gb,
        "llm_calls_per_sample": llm_calls_per_sample,
    }
