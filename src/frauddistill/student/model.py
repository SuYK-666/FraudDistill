from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StudentModelConfig:
    model_name: str = "xlm-roberta-base"
    num_binary_labels: int = 2
    num_risk_types: int = 10
    max_length: int = 384
    dropout: float = 0.1


class TorchDependencyError(RuntimeError):
    pass


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise TorchDependencyError("Install torch and transformers to use trainable student models.") from exc
    return torch, nn


def masked_pair_text(query: str, answer: str, mode: str) -> str:
    if mode == "q_only":
        answer = "<EMPTY_ANSWER>"
    elif mode == "y_only":
        query = "<EMPTY_QUERY>"
    elif mode != "q_y":
        raise ValueError(f"unknown mode: {mode}")
    return f"<QUERY> {query} </QUERY> <ANSWER> {answer} </ANSWER>"


class EncoderMultiHeadDetector:
    """Transformer encoder with binary, teacher-score, type and relevance heads."""

    def __new__(cls, config: StudentModelConfig):
        torch, nn = require_torch()
        from transformers import AutoModel

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.encoder = AutoModel.from_pretrained(config.model_name)
                hidden = int(self.encoder.config.hidden_size)
                self.dropout = nn.Dropout(config.dropout)
                self.binary_head = nn.Linear(hidden, 1)
                self.score_head = nn.Linear(hidden, 1)
                self.type_head = nn.Linear(hidden, config.num_risk_types)
                self.relevance_head = nn.Linear(hidden, 1)

            def forward(self, input_ids, attention_mask, **kwargs):
                encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                pooled = self.dropout(encoded.last_hidden_state[:, 0])
                binary_logits = self.binary_head(pooled).squeeze(-1)
                score_logits = self.score_head(pooled).squeeze(-1)
                return {"binary_logits": binary_logits, "unsafe_probability": torch.sigmoid(binary_logits), "teacher_score_pred": torch.sigmoid(score_logits), "type_logits": self.type_head(pooled), "relevance_pred": torch.sigmoid(self.relevance_head(pooled).squeeze(-1))}

        return _Model()
