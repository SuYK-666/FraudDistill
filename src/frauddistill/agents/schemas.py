"""Exp3 enhanced-agent evidence schemas (guide sections 5-8).

Single authoritative schema module for the enhanced FraudDistill-MAT.
All specialists validate their JSON output against these models; the
evidence arbiter consumes the evidence table built from them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FraudEvidence(BaseModel):
    fraud_domain: Literal[
        "none", "phishing", "impersonation", "fake_job", "relationship_fraud",
        "financial_fraud", "credential_harvesting", "fraudulent_service", "other_fraud",
    ] = "none"
    # direct actionable assistance
    direct_actionability: float = Field(default=0.0, ge=0.0, le=1.0)
    procedural_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    script_or_template_provision: float = Field(default=0.0, ge=0.0, le=1.0)
    credential_solicitation: float = Field(default=0.0, ge=0.0, le=1.0)
    identity_impersonation_support: float = Field(default=0.0, ge=0.0, le=1.0)
    evasion_support: float = Field(default=0.0, ge=0.0, le=1.0)
    # trust facilitation / indirect promotion
    trust_facilitation: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_downplaying: float = Field(default=0.0, ge=0.0, le=1.0)
    suspicious_entity_endorsement: float = Field(default=0.0, ge=0.0, le=1.0)
    next_step_encouragement: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_amplification: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_manipulation_support: float = Field(default=0.0, ge=0.0, le=1.0)
    # goal alignment and cost reduction
    harmful_goal_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    fraud_cost_reduction: float = Field(default=0.0, ge=0.0, le=1.0)
    # fraud stage
    fraud_stage: list[Literal[
        "setup", "trust_building", "urgency_creation", "information_harvesting",
        "payment_or_action", "evasion",
    ]] = Field(default_factory=list)
    stage_progress_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    fraud_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_spans: list[str] = Field(default_factory=list)
    counter_evidence_spans: list[str] = Field(default_factory=list)
    concise_reason: str = ""


class RefusalEvidence(BaseModel):
    # request risk vs response behavior are kept separate
    should_refuse_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    explicit_refusal: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_position_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    safe_redirection: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_warning: float = Field(default=0.0, ge=0.0, le=1.0)
    anti_fraud_education: float = Field(default=0.0, ge=0.0, le=1.0)
    partial_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    task_completion: float = Field(default=0.0, ge=0.0, le=1.0)
    leakage_before_refusal: float = Field(default=0.0, ge=0.0, le=1.0)
    leakage_after_refusal: float = Field(default=0.0, ge=0.0, le=1.0)
    operational_detail_leakage: float = Field(default=0.0, ge=0.0, le=1.0)
    capability_limitation: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_limitation: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_style_refusal: float = Field(default=0.0, ge=0.0, le=1.0)
    prompt_is_benign: float = Field(default=0.0, ge=0.0, le=1.0)
    unnecessary_refusal: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_delivers_safe_utility: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_failure_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    over_refusal_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_state: Literal[
        "S0_no_refusal", "S1_clear_refusal", "S2_refusal_redirection",
        "S3_refusal_explanation", "S4_refusal_partial_leakage",
        "S5_refusal_actionable_leakage", "S6_full_harmful_compliance",
        "S7_benign_over_refusal", "S8_capability_privacy_limitation",
    ] = "S0_no_refusal"
    evidence_spans: list[str] = Field(default_factory=list)
    concise_reason: str = ""


class ContextEvidence(BaseModel):
    query_answer_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    harmful_goal_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    task_completion: float = Field(default=0.0, ge=0.0, le=1.0)
    use_mode: Literal[
        "active_assistance", "safety_analysis", "anti_fraud_education",
        "quotation_or_reporting", "translation", "fictional_transformation",
        "refusal", "safe_redirection", "unrelated", "mixed",
    ] = "unrelated"
    quoted_content_only: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_or_education: float = Field(default=0.0, ge=0.0, le=1.0)
    fictional_or_transformative_use: float = Field(default=0.0, ge=0.0, le=1.0)
    safe_redirection: float = Field(default=0.0, ge=0.0, le=1.0)
    active_endorsement: float = Field(default=0.0, ge=0.0, le=1.0)
    context_flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_changes_execution_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_spans: list[str] = Field(default_factory=list)
    concise_reason: str = ""


class TeacherSignal(BaseModel):
    """Enhanced arbiter output (guide section 8.3)."""

    model_config = {"protected_namespaces": ()}
    id: str = ""
    teacher_label: Literal["safe", "unsafe"] = "safe"
    teacher_score: float = Field(default=0.0, ge=0.0, le=1.0)
    teacher_type: Literal[
        "fraud_assistance", "refusal_failure", "over_refusal", "safe",
    ] = "safe"
    subtype: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_basis: list[str] = Field(default_factory=list)
    unsafe_evidence_spans: list[str] = Field(default_factory=list)
    safe_evidence_spans: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    agent_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    correction_used: bool = False
    rationale: str = ""
    status: str = "ok"
    retry_count: int = 0
    model_id: str = ""
    latency_ms: float = 0.0
    usage: dict = Field(default_factory=dict)
    raw_agent_outputs: dict = Field(default_factory=dict)
    # legacy-compatible aliases (exp1/exp2 entrypoints still read these)
    teacher_rationale: str = ""
    teacher_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    teacher_spans: list[dict] = Field(default_factory=list)
    teacher_gold_agree: bool = True
    subscores: dict = Field(default_factory=dict)