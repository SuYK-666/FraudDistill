"""Delta planner: cache invalidation and partial re-runs (guide section 17).

Only agents whose prompt/schema digest changed are re-run; unchanged
specialist outputs are merged back into the row, and the arbiter always
re-runs when any specialist changed (guide 17.3).
"""
from __future__ import annotations

import hashlib
from typing import Any

AGENT_NAMES = ("fraud", "refusal", "context", "arbiter")

# prompts live in the agent modules; schema digests cover the guide-9 contract
AGENT_PROMPT_SOURCES = {
    "fraud": ("frauddistill.agents.fraud_assistance_agent", "SYSTEM_PROMPT"),
    "refusal": ("frauddistill.agents.refusal_quality_agent", "SYSTEM_PROMPT"),
    "context": ("frauddistill.agents.relevance_agent", "SYSTEM_PROMPT"),
    "arbiter": ("frauddistill.agents.arbiter_agent", "SYSTEM_PROMPT"),
}

SCHEMA_SOURCES = {
    "fraud": "frauddistill.agents.schemas:FraudEvidence",
    "refusal": "frauddistill.agents.schemas:RefusalEvidence",
    "context": "frauddistill.agents.schemas:ContextEvidence",
    "arbiter": "frauddistill.agents.schemas:TeacherSignal",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_digest(agent: str) -> str:
    """Content digest of the agent system prompt (guide 17.1)."""
    import importlib

    if agent not in AGENT_PROMPT_SOURCES:
        raise KeyError(f"unknown agent: {agent}")
    mod_name, attr = AGENT_PROMPT_SOURCES[agent]
    mod = importlib.import_module(mod_name)
    return _sha256(str(getattr(mod, attr)))[:16]


def schema_digest(agent: str) -> str:
    """Content digest of the agent output schema (guide 17.1)."""
    import inspect

    if agent not in SCHEMA_SOURCES:
        raise KeyError(f"unknown agent: {agent}")
    mod_name, cls_name = SCHEMA_SOURCES[agent].split(":")
    import importlib

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return _sha256(inspect.getsource(cls))[:16]


def qy_hash(query: str, answer: str) -> str:
    """Sample q+y content hash (guide 17.1)."""
    return _sha256(f"{query}\x00{answer}")[:16]


def agent_versions() -> dict[str, str]:
    """Current agent_versions map (guide 17.2)."""
    return {
        agent: f"sha256:{prompt_digest(agent)}:{schema_digest(agent)}"
        for agent in AGENT_NAMES
    }


def agents_to_rerun(old: dict[str, str] | None, new: dict[str, str] | None) -> list[str]:
    """Return agents whose version changed (guide 17.3)."""
    old = old or {}
    new = new or agent_versions()
    changed = [agent for agent in AGENT_NAMES if old.get(agent) != new.get(agent)]
    # arbiter must re-run if any specialist changed (guide 17.3)
    specialists = [a for a in ("fraud", "refusal", "context") if a in changed]
    if specialists and "arbiter" not in changed:
        changed.append("arbiter")
    return changed


def merge_agent_outputs(old_row: dict, new_partial: dict) -> dict:
    """Merge freshly re-run agent outputs into an existing prediction row (guide 17.4)."""
    merged = dict(old_row)
    merged["agent_outputs"] = dict(old_row.get("agent_outputs") or {})
    for agent_name, output in (new_partial.get("agent_outputs") or {}).items():
        merged["agent_outputs"][agent_name] = output
    if new_partial.get("arbiter_output") is not None:
        merged["arbiter_output"] = new_partial["arbiter_output"]
    return merged


def row_agent_versions(row: dict) -> dict[str, str]:
    """Extract agent_versions from a prediction row (default: none)."""
    return dict(row.get("agent_versions") or {})


def prediction_digest(row: dict) -> str:
    """Content hash over the final prediction fields of a row."""
    payload = {
        "id": row.get("id"),
        "prediction_binary": row.get("prediction_binary"),
        "risk_score": row.get("risk_score"),
        "prediction_type": row.get("prediction_type"),
        "arbiter_json": row.get("arbiter_json"),
    }
    return _sha256(str(sorted(payload.items())))[:16]
