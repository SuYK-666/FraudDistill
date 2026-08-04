# -*- coding: utf-8 -*-
"""Neural student models (guide 13): standard sequence classifier + q+y interaction head."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


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


# ---------------------------------------------------------------------------
# Neural student (guide 13)
# ---------------------------------------------------------------------------
@dataclass
class NeuralStudentConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    num_labels: int = 4
    architecture: str = "standard"          # standard | interaction
    max_length: int = 1536
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    use_lora: bool = True


class QueryAnswerInteractionHead(torch.nn.Module):
    """Guide 13.3: q/y segment pooling + last-token pooling + fusion MLP."""

    def __init__(self, hidden_size: int, num_labels: int = 4):
        super().__init__()
        fusion_size = hidden_size * 5
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(fusion_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels),
        )

    def forward(self, hidden_states, query_mask, answer_mask, attention_mask):
        h_query = masked_mean(hidden_states, query_mask)
        h_answer = masked_mean(hidden_states, answer_mask)
        last_index = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        h_last = hidden_states[batch_idx, last_index]
        fused = torch.cat([h_query, h_answer, torch.abs(h_query - h_answer),
                           h_query * h_answer, h_last], dim=-1)
        return self.proj(fused)


def masked_mean(hidden_states, mask):
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def build_neural_student(config: NeuralStudentConfig, freeze_base: bool = False, device=None):
    """Guide 13.1/13.2: Qwen2 backbone -> 4-class unified softmax.

    - standard: AutoModelForSequenceClassification (last non-padding token pooling);
    - interaction: base model + QueryAnswerInteractionHead on segment pools.
    LoRA adapters are attached when use_lora=True (guide 14).
    """
    torch, nn = require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    if config.architecture == "standard":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels, torch_dtype=torch.float32)
        model.config.pad_token_id = model.config.eos_token_id or 0
    elif config.architecture == "interaction":
        from transformers import AutoModel
        base = AutoModel.from_pretrained(config.model_name, torch_dtype=torch.float32)
        hidden = int(base.config.hidden_size)
        head = QueryAnswerInteractionHead(hidden, config.num_labels)
        model = _InteractionStudent(base, head)
    else:
        raise ValueError(f"unknown architecture: {config.architecture}")

    if config.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=config.target_modules, bias="none", task_type="SEQ_CLS")
        model = get_peft_model(model, lora_cfg)
    if freeze_base:
        for name, p in model.named_parameters():
            if "lora" not in name and "score" not in name and "head" not in name:
                p.requires_grad = False
    return model


class _InteractionStudent(torch.nn.Module):
    """Base model + q+y interaction head; exposes the same train/eval interface."""

    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head
        self.config = base.config

    def forward(self, input_ids, attention_mask, query_mask=None, answer_mask=None, labels=None, **kw):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kw)
        qm = query_mask if query_mask is not None else attention_mask
        am = answer_mask if answer_mask is not None else attention_mask
        logits = self.head(out.last_hidden_state, qm, am, attention_mask)
        return _SimpleSeqOut(logits, labels)


class _SimpleSeqOut:
    """Minimal output object matching the Trainer/loop interface."""

    def __init__(self, logits, labels=None):
        self.logits = logits
        self.loss = None
        if labels is not None:
            import torch.nn.functional as F
            self.loss = F.cross_entropy(logits, labels)
