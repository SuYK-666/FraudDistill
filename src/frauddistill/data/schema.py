from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Label(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class Source(StrEnum):
    FRAUDR1 = "Fraud-R1"
    DO_NOT_ANSWER = "Do-Not-Answer"
    AEGIS = "Aegis"
    OR_BENCH = "OR-Bench"
    SYNTHETIC = "synthetic"
    HALUBENCH = "HaluBench"
    RAGTRUTH = "RAGTruth"
    HALUEVAL = "HaluEval"
    FELM = "FELM"


class RiskType(StrEnum):
    SAFE = "safe"
    NONE = "none"
    FRAUD_ASSISTANCE = "fraud_assistance"
    REFUSAL_FAILURE = "refusal_failure"
    PARTIAL_LEAKAGE = "partial_leakage"
    OVER_REFUSAL = "over_refusal"
    CORRECT_REFUSAL = "correct_refusal"
    SAFE_COMPLETION = "safe_completion"
    ANTI_FRAUD_EDUCATION = "anti_fraud_education"
    IRRELEVANT_SAFE = "irrelevant_safe"
    HALLUCINATION = "hallucination"
    MISINFORMATION = "misinformation"
    IRRELEVANT_OR_EVASIVE = "irrelevant_or_evasive"


class FraudDistillSample(BaseModel):
    id: str
    source: Source
    language: str = "unknown"
    user_query: str
    target_model_answer: str
    context: str | None = None
    gold_label: Label
    gold_risk_type: RiskType | None = None
    prompt_harmfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    response_safety: Label | None = None
    response_behavior: str | None = None
    is_over_refusal: bool = False
    should_refuse: bool | None = None
    did_refuse: bool | None = None
    substantive_help: float | None = Field(default=None, ge=0.0, le=1.0)
    reference_type: str = "official_gold"
    reference_source: str | None = None
    teacher_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    parse_status: str = "not_applicable"
    split: str = "unspecified"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "user_query", "target_model_answer")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be blank")
        return value

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str) -> str:
        allowed = {"train", "dev", "test", "unspecified"}
        if value not in allowed:
            raise ValueError(f"split must be one of {sorted(allowed)}")
        return value


class RiskSpan(BaseModel):
    span: str
    risk_type: str = RiskType.NONE
    severity: float = Field(ge=0.0, le=1.0)


class TeacherSignal(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    teacher_label: Label
    teacher_score: float = Field(ge=0.0, le=1.0)
    teacher_type: RiskType
    teacher_spans: list[RiskSpan] = Field(default_factory=list)
    teacher_rationale: str = ""
    conflict_notes: str | None = None
    raw_agent_outputs: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    retry_count: int = Field(default=0, ge=0)
    model_id: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    teacher_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("id must not be blank")
        return value


def sample_from_mapping(row: dict[str, Any]) -> FraudDistillSample:
    return FraudDistillSample.model_validate(row)


def teacher_from_mapping(row: dict[str, Any]) -> TeacherSignal:
    return TeacherSignal.model_validate(row)
