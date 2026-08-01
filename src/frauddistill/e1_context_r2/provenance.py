from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import sha_text

PLACEHOLDER_RE = re.compile(r"\[REDACTED_FOR_PUBLIC_REPORT\]|private prompt is stored only|attacker-facing evaluation seed", re.I)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_placeholder_q(q: str) -> bool:
    return bool(PLACEHOLDER_RE.search(q or ""))


def get_q(row: dict[str, Any]) -> str:
    return str(row.get("q_private") or row.get("q") or row.get("user_query") or row.get("prompt") or "").strip()


def get_y(row: dict[str, Any]) -> str:
    return str(row.get("y_private") or row.get("y") or row.get("text") or row.get("target_model_answer") or "").strip()


def provider_of(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("target_model", "provider", "requested_model", "response_model", "target_provider", "response_id", "task_id")).lower()
    if "deepseek" in text:
        return "deepseek"
    if "qwen" in text or "dashscope" in text:
        return "qwen"
    return ""


def language_of(row: dict[str, Any], q: str) -> str:
    lang = str(row.get("language") or "").lower()
    if lang.startswith("zh") or "chinese" in lang:
        return "zh"
    if lang.startswith("en") or "english" in lang:
        return "en"
    return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in q) else "en"


def category_of(row: dict[str, Any], q: str = "", y: str = "") -> str:
    cat = str(row.get("category") or row.get("fraud_category") or row.get("subcategory") or "").lower()
    text = f"{cat} {q} {y}".lower()
    if "phish" in cat:
        return "phishing"
    if "imperson" in text:
        return "impersonation"
    if "job" in text or "recruit" in text or "职位" in text or "招聘" in text:
        return "fake_job"
    if any(k in text for k in ["relationship", "romance", "dating", "lover", "investment", "crypto", "恋爱", "交友", "投资"]):
        return "relationship_investment"
    if cat:
        return "fraudulent_service"
    return "unknown"


def actor_role(row: dict[str, Any], q: str) -> str:
    role = str(row.get("actor_role") or row.get("requester_role") or "").upper()
    if role in {"PERPETRATOR", "DUAL_USE_AMBIGUOUS", "VICTIM_DEFENDER"}:
        return role
    low = q.lower()
    if any(x in low for x in ["will you apply", "you received", "你刚收到", "你会理会", "victim", "customer"]):
        return "VICTIM_DEFENDER"
    return "DUAL_USE_AMBIGUOUS"


def canonicalize_row(path: Path, row: dict[str, Any], index: int) -> tuple[dict[str, Any] | None, str]:
    q = get_q(row)
    y = get_y(row)
    provider = provider_of(row)
    if not q:
        return None, "missing_q"
    if not y:
        return None, "missing_y"
    if is_placeholder_q(q):
        return None, "placeholder_q"
    if provider not in {"qwen", "deepseek"}:
        return None, "not_target_qwen_deepseek"
    if str(row.get("status") or "ok").lower() not in {"ok", "success", ""}:
        return None, "bad_status"
    if len(y) < 20:
        return None, "short_y"
    response_id = str(row.get("response_id") or row.get("task_id") or f"{path.name}:{index}")
    q_hash = sha_text(normalize_text(q).lower())
    y_hash = sha_text(normalize_text(y).lower())
    out = {
        "response_id": response_id,
        "source_file": str(path),
        "source_version": path.parent.name,
        "source_index": index,
        "canonical_q_id": str(row.get("canonical_id") or row.get("canonical_q_id") or row.get("q_stage_group_id") or q_hash[:16]),
        "exact_q_hash": q_hash,
        "semantic_q_component": q_hash[:16],
        "template_family_id": str(row.get("stage_id", "stage0")) + "|" + q_hash[:10],
        "source_case_id": str(row.get("canonical_id") or row.get("canonical_q_id") or q_hash[:12]),
        "y_hash": y_hash,
        "q_private": q,
        "y_private": y,
        "target_model": provider,
        "target_model_snapshot": str(row.get("response_model") or row.get("target_model") or row.get("requested_model") or provider),
        "requested_model": str(row.get("requested_model") or row.get("target_model") or provider),
        "language": language_of(row, q),
        "fraud_category": category_of(row, q, y),
        "actor_role": actor_role(row, q),
        "stage_id": int(row.get("stage_id") or 0),
        "replicate_id": int(row.get("replicate_id") or 0),
        "old_material": int(row.get("material_central", row.get("material_assist", row.get("gold", 0))) or 0),
        "old_upper": int(row.get("material_upper", row.get("material_assist", row.get("gold", 0))) or 0),
        "old_uncertain": bool(row.get("gold_uncertain") or row.get("x_consensus_state") == "UNCERTAIN"),
        "generation_mode": "standard-natural" if "v91" in str(path).lower() or "v10_response_registry" in path.name.lower() else "historical-pressure",
    }
    return out, "accepted"
