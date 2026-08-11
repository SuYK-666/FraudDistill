# -*- coding: utf-8 -*-
"""E6 common helpers: paths, IO, normalization, hashing, cost ledger, budget gate."""
from __future__ import annotations
import hashlib, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E6_DIR = ROOT / "experiments" / "exp6_multi_api"
PROTOCOL_DIR = E6_DIR / "protocol"
DATA_DIR = E6_DIR / "data"
GEN_DIR = E6_DIR / "generations"
STUDENT_DIR = E6_DIR / "student"
SILVER_DIR = E6_DIR / "silver_audit"
BUDGET_DIR = E6_DIR / "budget"
TABLES_DIR = E6_DIR / "tables"
FIGURES_DIR = E6_DIR / "figures"

PROTOCOL_VERSION = "E6-DIRECT-API-v1.0-50CNY"
SEED = 20260810
STUDENT_THRESHOLD = 0.5622
STUDENT_MAX_LENGTH = 512
STUDENT_CKPT = ROOT / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120"
TRAIN_MANIFEST = ROOT / "data/prepared/exp3_neural_student/final_train_manifest.jsonl"
BUDGET_HARD_LIMIT = 50.0
BUDGET_WARN_45 = 45.0

# price snapshot (CNY per 1K tokens), updated after probe from real usage where possible
PRICE_DEFAULTS = {
    "qwen-flash": {"in": 0.0005, "out": 0.002},
    "qwen-plus": {"in": 0.0008, "out": 0.002},
    "deepseek-v4-flash": {"in": 0.001, "out": 0.002},
    "deepseek-v4-pro": {"in": 0.002, "out": 0.008},
    "glm-4-flash": {"in": 0.0005, "out": 0.001},
    "moonshot-v1-8k": {"in": 0.0012, "out": 0.0024},
}

MODEL_SLOTS = [
    {"slot": "M1", "provider": "qwen", "role": "low_cost", "candidate_ids": ["qwen-flash", "qwen-plus"]},
    {"slot": "M2", "provider": "qwen", "role": "balanced", "candidate_ids": ["qwen-plus", "qwen-flash"]},
    {"slot": "M3", "provider": "deepseek", "role": "low_cost", "candidate_ids": ["deepseek-v4-flash", "deepseek-chat"]},
    {"slot": "M4", "provider": "deepseek", "role": "high_capability", "candidate_ids": ["deepseek-v4-pro", "deepseek-reasoner"]},
    {"slot": "M5", "provider": "glm", "role": "cross_family_low_cost", "candidate_ids": ["glm-4-flash", "glm-4-flashx"]},
    {"slot": "M6", "provider": "kimi", "role": "cross_family_low_cost", "candidate_ids": ["moonshot-v1-8k", "kimi-latest"]},
]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def norm_query(q: str) -> str:
    q = q or ""
    q = q.strip()
    q = re.sub(r"[?？]{2,}\s*$", "", q)
    q = re.sub(r"^\s*[?？]{2,}\s*", "", q)
    q = re.sub(r"[ \t\u3000]+", " ", q)
    q = re.sub(r"\n{3,}", "\n\n", q)
    return q.strip()

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")

def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def manifest_sha256(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

class CostLedger:
    """Append-only ledger with pre-request budget gate."""
    def __init__(self, path: Path | None = None):
        self.path = path or (BUDGET_DIR / "cost_ledger.jsonl")
        self.rows = read_jsonl(self.path)
        self.cumulative = sum(float(r.get("cost_cny", 0.0) or 0.0) for r in self.rows)

    def remaining(self) -> float:
        return BUDGET_HARD_LIMIT - self.cumulative

    def gate_ok(self, estimated_next: float = 0.0, stage: str = "") -> bool:
        if self.cumulative + estimated_next > BUDGET_HARD_LIMIT:
            raise RuntimeError(
                f"[BUDGET-GATE] cumulative={self.cumulative:.2f} + est_next={estimated_next:.2f} > {BUDGET_HARD_LIMIT} "
                f"(stage={stage}). HARD STOP before request."
            )
        if self.cumulative >= BUDGET_WARN_45 and stage not in ("adjudication", "judge"):
            print(f"[BUDGET-WARN] cumulative={self.cumulative:.2f} >= {BUDGET_WARN_45}; only judge/adjudication allowed.")
            if stage not in ("judge", "adjudication"):
                raise RuntimeError("[BUDGET-WARN] non-essential stage blocked near budget limit.")
        return True

    def append(self, entry: dict) -> None:
        entry["cumulative_cost_cny"] = round(self.cumulative + float(entry.get("cost_cny", 0.0)), 4)
        entry["budget_remaining_cny"] = round(BUDGET_HARD_LIMIT - entry["cumulative_cost_cny"], 4)
        entry["timestamp_utc"] = entry.get("timestamp_utc") or utc_now()
        self.rows.append(entry)
        self.cumulative = entry["cumulative_cost_cny"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        by_stage: dict[str, dict] = {}
        for r in self.rows:
            st = r.get("stage", "?")
            d = by_stage.setdefault(st, {"calls": 0, "cost_cny": 0.0, "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["cost_cny"] = round(d["cost_cny"] + float(r.get("cost_cny", 0.0)), 4)
            d["input_tokens"] += int(r.get("input_tokens", 0) or 0)
            d["output_tokens"] += int(r.get("output_tokens", 0) or 0)
        return {"cumulative_cost_cny": round(self.cumulative, 4),
                "budget_hard_limit": BUDGET_HARD_LIMIT,
                "remaining_cny": round(self.remaining(), 4),
                "by_stage": by_stage}

    def write_summary(self) -> None:
        write_json(BUDGET_DIR / "cost_summary.json", self.summary())
        write_json(BUDGET_DIR / "budget_gate.json", {
            "hard_limit_cny": BUDGET_HARD_LIMIT,
            "cumulative_cny": round(self.cumulative, 4),
            "remaining_cny": round(self.remaining(), 4),
            "status": "OK" if self.cumulative <= BUDGET_HARD_LIMIT else "HARD_STOP_EXCEEDED",
        })

def load_manifest(full: bool = False) -> list[dict]:
    """Load frozen E6 manifest (full local copy)."""
    return read_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl")

def load_registry() -> dict:
    return read_json(PROTOCOL_DIR / "model_registry_frozen.yaml".replace(".yaml", ".json")) or {}

def est_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICE_DEFAULTS.get(model, {"in": 0.002, "out": 0.008})
    return round((in_tok * p["in"] + out_tok * p["out"]) / 1000.0, 6)
