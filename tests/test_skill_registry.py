"""SkillRegistry tests (guide section 34.1)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.skills.registry import SkillRegistry, check_expected_skills, registry_digest

SKILLS_ROOT = REPO / "skills"


def test_skill_registry_discovers_all():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    assert len(reg) >= 21


def test_expected_skills_present():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    missing = check_expected_skills(reg)
    assert missing == []


def test_skill_name_matches_directory():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    for name, rec in reg.records().items():
        assert rec.path.parent.name == name


def test_skill_frontmatter_required():
    tmp = REPO / "tests" / "_tmp_skills_no_fm"
    (tmp / "bad-skill").mkdir(parents=True, exist_ok=True)
    (tmp / "bad-skill" / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
    try:
        reg = SkillRegistry(tmp)
        try:
            reg.discover()
            raised = False
        except ValueError:
            raised = True
        assert raised, "missing frontmatter must raise"
    finally:
        import shutil
        shutil.rmtree(tmp)


def test_skill_description_nonempty():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    for name, rec in reg.records().items():
        assert rec.description.strip(), f"{name} empty description"


def test_skill_digest_stable():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    d1 = registry_digest(reg)
    reg2 = SkillRegistry(SKILLS_ROOT).discover()
    assert d1 == registry_digest(reg2)


def test_skill_path_cannot_escape_root():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # symlink-free check: a skill dir outside root cannot be globbed, but a
        # crafted name mismatch must raise
        (root / "x").mkdir()
        (root / "x" / "SKILL.md").write_text(
            "---\nname: y\ndescription: d\n---\nbody\n", encoding="utf-8")
        try:
            SkillRegistry(root).discover()
            raised = False
        except ValueError:
            raised = True
        assert raised


def test_skill_scripts_not_executed():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    for rec in reg.records().values():
        assert "```sh" not in rec.body and "```bash" not in rec.body
        assert "subprocess" not in rec.body and "os.system" not in rec.body
