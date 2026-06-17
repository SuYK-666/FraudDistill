from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StudentModelConfig:
    model_name: str = "xlm-roberta-base"
    num_binary_labels: int = 2
    num_risk_types: int = 6
    max_length: int = 512


class TorchDependencyError(RuntimeError):
    pass


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise TorchDependencyError("Install the 'student' extra to use trainable student models.") from exc
    return torch, nn


class EncoderMultiHeadDetector:
    """Lazy wrapper placeholder; full construction occurs only when torch/transformers are installed."""

    def __init__(self, config: StudentModelConfig):
        self.config = config
        require_torch()
