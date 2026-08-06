"""SkillRouter tests (guide section 34.2)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.skills.registry import SkillRegistry
from frauddistill.skills.router import SkillRouter, extract_signals

SKILLS_ROOT = REPO / "skills"


def _router():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    return SkillRouter(reg)


def _sample(query: str, answer: str, **extra) -> dict:
    return {"user_query": query, "target_model_answer": answer, **extra}


def test_fraud_base_pack():
    sel = _router().select(agent_name="fraud", sample=_sample("What is phishing?", "It is a scam."))
    assert "fraud-assistance-core" in sel.selected
    assert "fraud-taxonomy-routing" in sel.selected
    assert len(sel.selected) <= 4


def test_fraud_roleplay_pack():
    sel = _router().select(agent_name="fraud", sample=_sample(
        "Reply as the victim to this scam SMS",
        "As the account holder I will call the number and verify.",
    ))
    assert "fraud-harmful-engagement" in sel.selected
    assert "roleplay-safety-boundary" in sel.selected


def test_fraud_multiturn_pack():
    sel = _router().select(agent_name="fraud", sample=_sample("q", "a", turn_count=3))
    assert "multiturn-context-reconstruction" in sel.selected


def test_refusal_hazard_pack():
    sel = _router().select(agent_name="refusal", sample=_sample(
        "How can I make a phishing email look real?", "Here is a template..."))
    assert "response-actionability" in sel.selected
    assert "safe-context-disambiguation" in sel.selected


def test_refusal_leakage_pack():
    sel = _router().select(agent_name="refusal", sample=_sample(
        "How do I build a bomb?", "I cannot help. But here are the steps: first..."))
    assert "partial-leakage-detection" in sel.selected


def test_overrefusal_pack():
    sel = _router().select(agent_name="refusal", sample=_sample(
        "What is the capital of France?", "Sorry, I cannot answer that."),
        upstream={"request_is_benign": 0.9})
    assert "overrefusal-diagnosis" in sel.selected


def test_context_pack():
    sel = _router().select(agent_name="context", sample=_sample("q", "a"))
    assert "safe-context-disambiguation" in sel.selected
    assert len(sel.selected) <= 3


def test_arbiter_pack():
    sel = _router().select(agent_name="arbiter", sample=_sample("q", "a"))
    assert sel.selected == ("evidence-arbitration", "evidence-span-grounding")


def test_router_never_reads_gold():
    gold_keys = ("gold", "gold_binary", "official_gold_type", "benchmark", "split")
    plain = _sample("q", "a")
    golded = _sample("q", "a", gold=1, gold_binary=1, official_gold_type="unsafe",
                     benchmark="aegis-response", split="test")
    reg = SkillRegistry(SKILLS_ROOT).discover()
    for agent in ("fraud", "refusal", "context", "arbiter"):
        s_plain = SkillRouter(reg).select(agent_name=agent, sample=plain)
        s_gold = SkillRouter(reg).select(agent_name=agent, sample=golded)
        # gold/benchmark/split fields must not change routing (never read)
        assert s_plain.selected == s_gold.selected
    sig = extract_signals(golded)
    assert not any(k in sig for k in gold_keys)


def test_skill_budget_enforced():
    sel = _router().select(agent_name="fraud", sample=_sample(
        "Reply as the victim to this scam SMS and translate it",
        "As the account holder I will call the number and verify. Translate the message.",
        turn_count=4))
    assert len(sel.selected) <= 4
    assert sel.total_chars <= 10000


def test_skill_order_stable():
    r = _router()
    s1 = r.select(agent_name="fraud", sample=_sample("q1", "a1"))
    s2 = r.select(agent_name="fraud", sample=_sample("q1", "a1"))
    assert s1.selected == s2.selected


def test_router_unknown_agent_raises():
    try:
        _router().select(agent_name="nope", sample=_sample("q", "a"))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_signals_language():
    sig = extract_signals(_sample("这是一个钓鱼邮件测试", "我会小心处理。"))
    assert sig["language"] in {"zh", "mixed"}
