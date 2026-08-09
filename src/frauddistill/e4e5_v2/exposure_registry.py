# -*- coding: utf-8 -*-
"""Exposure registry: merge all historical assets and audit exact/family/template exposure."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import family_key_of, norm_text, q_hash, qy_hash, read_jsonl, write_jsonl, y_hash


class ExposureRegistry:
    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self.rows: list[dict] = []
        self.qy: set[str] = set()
        self.q: set[str] = set()
        self.y: set[str] = set()
        self.families: set[str] = set()
        self.templates: set[str] = set()
        self.ids: set[str] = set()
        # hard exposure: rows the student / evaluators actually consumed
        self.qy_hard: set[str] = set()
        self.families_hard: set[str] = set()
        self.templates_hard: set[str] = set()
        # soft exposure: prepared candidate pools never consumed
        self.qy_pool: set[str] = set()
        self.families_pool: set[str] = set()

    # ------------------------------------------------------------------ ingest
    def add(self, rows: Iterable[dict], role: str, experiment: str, was_label_visible: bool = True,
            was_prediction_visible: bool = False) -> int:
        n = 0
        for r in rows:
            q = str(r.get("user_query") or r.get("query") or r.get("prompt") or "")
            y = str(r.get("target_model_answer") or r.get("answer") or r.get("response") or "")
            qy = qy_hash(q, y) if q and y else None
            row = {
                "sample_id": str(r.get("id") or r.get("sample_id") or r.get("record_id") or f"anon_{len(self.rows)}"),
                "canonical_case_id": str(r.get("canonical_case_id") or r.get("group_id") or ""),
                "family_id": str(r.get("template_family_id") or r.get("family_id") or r.get("group_id") or r.get("group") or ""),
                "pair_id": str(r.get("pair_id") or ""),
                "template_id": str(r.get("template_id") or r.get("group_id") or ""),
                "source": str(r.get("source") or r.get("dataset") or ""),
                "source_version": str(r.get("source_version") or ""),
                "fraud_category": str(r.get("fraud_category") or r.get("category") or r.get("subtype") or ""),
                "language": str(r.get("language") or r.get("lang") or ""),
                "target_model": str(r.get("target_model") or (r.get("metadata") or {}).get("target_model") or ""),
                "target_model_version": str(r.get("target_model_version") or ""),
                "generation_prompt_hash": str(r.get("generation_prompt_hash") or ""),
                "q_exact_hash": q_hash(q) if q else "",
                "y_exact_hash": y_hash(y) if y else "",
                "qy_exact_hash": qy or "",
                "q_normalized_hash": q_hash(q) if q else "",
                "y_normalized_hash": y_hash(y) if y else "",
                "exposure_role": role,
                "first_exposure_experiment": experiment,
                "was_label_visible": bool(was_label_visible),
                "was_prediction_visible": bool(was_prediction_visible),
            }
            self.rows.append(row)
            if qy:
                self.qy.add(qy)
            if q:
                self.q.add(q_hash(q))
            if y:
                self.y.add(y_hash(y))
            fam = row["family_id"] or row["canonical_case_id"]
            if fam:
                self.families.add(fam)
            tpl = row["template_id"] or fam
            if tpl:
                self.templates.add(tpl)
            if row["sample_id"]:
                self.ids.add(row["sample_id"])
            if role in HARD_ROLES:
                if qy:
                    self.qy_hard.add(qy)
                if fam:
                    self.families_hard.add(fam)
                if tpl:
                    self.templates_hard.add(tpl)
            else:
                if qy:
                    self.qy_pool.add(qy)
                if fam:
                    self.families_pool.add(fam)
            n += 1
        return n

    def add_jsonl(self, path: Path, role: str, experiment: str, q_key: str = "auto",
                  a_key: str = "auto", **kw) -> int:
        rows = read_jsonl(path)
        return self.add(rows, role=role, experiment=experiment, **kw)

    # ------------------------------------------------------------------ audit
    def exact_overlap(self, q: str, y: str) -> dict:
        qy = qy_hash(q, y)
        return {
            "qy_exact": qy in self.qy,
            "q_exact": q_hash(q) in self.q,
            "y_exact": y_hash(y) in self.y,
        }

    def family_overlap(self, fam: str) -> bool:
        return bool(fam) and fam in self.families

    def template_overlap(self, tpl: str) -> bool:
        return bool(tpl) and tpl in self.templates

    def audit_candidate(self, row: dict) -> dict:
        q = str(row.get("user_query") or "")
        y = str(row.get("target_model_answer") or "")
        fam = str(row.get("family_id") or "")
        tpl = str(row.get("template_id") or fam)
        ex = self.exact_overlap(q, y)
        qy_hard = qy_hash(q, y) in self.qy_hard
        qy_pool = (not qy_hard) and qy_hash(q, y) in self.qy_pool
        fam_hard = bool(fam) and fam in self.families_hard
        tpl_hard = bool(tpl) and tpl in self.templates_hard
        fam_pool = (not fam_hard) and bool(fam) and fam in self.families_pool
        return {
            "id": row.get("id"),
            "qy_exact_overlap": ex["qy_exact"],
            "qy_hard_overlap": qy_hard,
            "qy_pool_overlap": qy_pool,
            "q_exact_overlap": ex["q_exact"],
            "y_exact_overlap": ex["y_exact"],
            "family_overlap": fam_hard,
            "template_overlap": tpl_hard,
            "family_pool_overlap": fam_pool,
            "passed": not (qy_hard or fam_hard or tpl_hard),
        }

    def save(self, path: Path) -> None:
        write_jsonl(path, self.rows)

    def summary(self) -> dict:
        roles = Counter(r["exposure_role"] for r in self.rows)
        exps = Counter(r["first_exposure_experiment"] for r in self.rows)
        return {
            "n_rows": len(self.rows),
            "n_unique_qy": len(self.qy),
            "n_unique_q": len(self.q),
            "n_unique_y": len(self.y),
            "n_families": len(self.families),
            "by_role": dict(roles),
            "by_experiment": dict(exps),
        }


# ------------------------------------------------------------------ asset catalogue
HARD_ROLES = {"train", "dev", "test", "teacher_expansion", "error_analysis", "pilot"}

ASSET_SPECS = [
    # (relative path, role, experiment, q_key, a_key)
    ("data/prepared/exp3_neural_student/final_train_manifest.jsonl", "train", "exp3_final_train", "user_query", "target_model_answer"),
    ("data/prepared/exp3_agent_distillation/dev.jsonl", "dev", "exp3_dev", "user_query", "target_model_answer"),
    ("data/prepared/exp3_agent_distillation/test.jsonl", "test", "exp3_test", "user_query", "target_model_answer"),
    ("data/prepared/exp3_agent_distillation/train.jsonl", "train_legacy", "exp3_legacy_train", "user_query", "target_model_answer"),
    ("data/prepared/exp3_agent_distillation/exp3_dataset.jsonl", "dataset", "exp3_dataset", "user_query", "target_model_answer"),
    ("data/prepared/full/evaluation_qy/aegis_qy.jsonl", "candidate_pool", "exp1_full_pool", "user_query", "target_model_answer"),
    ("data/prepared/full/evaluation_qy/do_not_answer_qy.jsonl", "candidate_pool", "exp1_full_pool", "user_query", "target_model_answer"),
    ("data/prepared/full/evaluation_qy/fraudr1_all_categories_qy.jsonl", "candidate_pool", "exp1_full_pool", "user_query", "target_model_answer"),
    ("data/prepared/full/evaluation_qy/v2_hard_control_full.jsonl", "test", "exp1_v2_hard_control", "user_query", "target_model_answer"),
    ("data/prepared/full/evaluation_qy/exp1_fraudr1_full.jsonl", "test", "exp1_fraudr1_full", "user_query", "target_model_answer"),
    ("data/unified/exp1_fraudr1_full.jsonl", "test", "exp1_fraudr1_full", "user_query", "target_model_answer"),
    ("data/unified/exp3_fraudr1_all_categories.jsonl", "candidate_pool", "exp3_all_categories", "user_query", "target_model_answer"),
    ("data/unified/v2_hard_control_full.jsonl", "test", "exp1_v2_hard_control", "user_query", "target_model_answer"),
    ("experiments/exp2_prior_work_comparison/aegis2/unified/aegis2_eval.jsonl", "test", "exp2_aegis", "user_query", "target_model_answer"),
    ("experiments/exp2_prior_work_comparison/aegis2/unified/aegis2_eval_valid_qy.jsonl", "test", "exp2_aegis", "user_query", "target_model_answer"),
    ("experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_test_manifest.jsonl", "test", "exp2_balanced_test", "user_query", "target_model_answer"),
    ("experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_dev_manifest.jsonl", "dev", "exp2_balanced_dev", "user_query", "target_model_answer"),
]


def build_registry(repo: Path, extra_assets: list[tuple[Path, str, str]] | None = None) -> ExposureRegistry:
    reg = ExposureRegistry(repo)
    loaded = []
    for rel, role, exp, qk, ak in ASSET_SPECS:
        p = repo / rel
        if not p.exists():
            print(f"[registry] MISSING {rel}")
            continue
        rows = read_jsonl(p)
        # remap q/a keys when the schema differs
        mapped = []
        for r in rows:
            if qk not in r and "user_query" in r:
                qk2, ak2 = "user_query", "target_model_answer"
            else:
                qk2, ak2 = qk, ak
            mapped.append(r)
        n = reg.add_jsonl(p, role, exp)
        loaded.append((rel, n))
    for p, role, exp in (extra_assets or []):
        if p.exists():
            n = reg.add_jsonl(p, role, exp)
            loaded.append((str(p), n))
    return reg, loaded
