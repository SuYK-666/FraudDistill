# -*- coding: utf-8 -*-
"""E6 v2 common helpers: paths, IO, normalization, hashing, cost ledger, budget gate."""
from __future__ import annotations
import hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V2_DIR = ROOT / "experiments" / "exp6_balanced_multi_api"
PROTOCOL_DIR = V2_DIR / "protocol"
DATA_DIR = V2_DIR / "data"
GEN_DIR = V2_DIR / "generations"
SILVER_DIR = V2_DIR / "silver"
BALANCED_DIR = V2_DIR / "balanced"
STUDENT_DIR = V2_DIR / "student"
BUDGET_DIR = V2_DIR / "budget"
TABLES_DIR = V2_DIR / "tables"
FIGURES_DIR = V2_DIR / "figures"

PROTOCOL_VERSION = "E6-V2-BALANCED-RESPONSE-DIRECT-API"
SEED = 20260811
STUDENT_THRESHOLD = 0.5622
STUDENT_MAX_LENGTH = 512
STUDENT_CKPT = ROOT / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120"
# v1 E6 spent 1.4153 CNY; v2 shares the same 50 CNY envelope
V1_COST_CNY = 1.4153
BUDGET_HARD_LIMIT = 50.0
BUDGET_WARN_45 = 45.0

PRICE_DEFAULTS = {
    "qwen-flash": {"in": 0.0005, "out": 0.002},
    "qwen-plus": {"in": 0.0008, "out": 0.002},
    "deepseek-v4-flash": {"in": 0.001, "out": 0.002},
    "deepseek-v4-pro": {"in": 0.002, "out": 0.008},
    "glm-4-flash": {"in": 0.0005, "out": 0.001},
    "moonshot-v1-8k": {"in": 0.0012, "out": 0.0024},
}

MODEL_SLOTS = ["M1", "M2", "M3", "M4", "M5", "M6"]
SLOT_MODEL = {"M1": "qwen-flash", "M2": "qwen-plus", "M3": "deepseek-v4-flash",
              "M4": "deepseek-v4-pro", "M5": "glm-4-flash", "M6": "moonshot-v1-8k"}
SLOT_PROVIDER = {"M1": "qwen", "M2": "qwen", "M3": "deepseek", "M4": "deepseek", "M5": "glm", "M6": "kimi"}
SLOT_LABEL = {"M1": "Qwen Flash", "M2": "Qwen Plus", "M3": "DeepSeek Flash",
              "M4": "DeepSeek Pro", "M5": "GLM Flash", "M6": "Kimi"}
FAMILY_OF = {"M1": "qwen", "M2": "qwen", "M3": "deepseek", "M4": "deepseek", "M5": "glm_kimi", "M6": "glm_kimi"}

JUDGE_PANEL = [
    {"tag": "J1", "provider": "qwen", "model": "qwen-flash", "role": "judge"},
    {"tag": "J2", "provider": "deepseek", "model": "deepseek-v4-flash", "role": "judge",
     "extra_body": {"thinking": {"type": "disabled"}}},
    {"tag": "J3", "provider": "glm", "model": "glm-4-flash", "role": "judge"},
]
ADJUDICATOR = {"tag": "J4", "provider": "kimi", "model": "moonshot-v1-8k", "role": "adjudicator"}

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
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

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

def est_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICE_DEFAULTS.get(model, {"in": 0.002, "out": 0.008})
    return round((in_tok * p["in"] + out_tok * p["out"]) / 1000.0, 6)

class CostLedger:
    """Append-only ledger seeded with v1 cumulative cost; 50 CNY hard limit."""
    def __init__(self, path: Path | None = None):
        self.path = path or (BUDGET_DIR / "cost_ledger.jsonl")
        self.rows = self._read_tolerant()
        self.cumulative = V1_COST_CNY
        if self.rows:
            self.cumulative = max(self.cumulative, sum(float(r.get("cost_cny", 0.0) or 0.0) for r in self.rows) + V1_COST_CNY if self.rows and self.rows[0].get("seed_entry") else sum(float(r.get("cost_cny", 0.0) or 0.0) for r in self.rows))

    def _read_tolerant(self) -> list[dict]:
        """Read ledger; if trailing line(s) are torn (crash/interleave), drop them."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rows = []
        for l in lines:
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except Exception:
                # drop torn tail lines silently (append-only ledger, no overwrite)
                continue
        return rows

    def remaining(self) -> float:
        return BUDGET_HARD_LIMIT - self.cumulative

    def gate_ok(self, estimated_next: float = 0.0, stage: str = "") -> bool:
        if self.cumulative + estimated_next > BUDGET_HARD_LIMIT:
            raise RuntimeError(
                f"[BUDGET-GATE] cumulative={self.cumulative:.2f} + est_next={estimated_next:.2f} > {BUDGET_HARD_LIMIT} (stage={stage}). HARD STOP."
            )
        if self.cumulative >= BUDGET_WARN_45 and stage not in ("judge", "adjudication"):
            raise RuntimeError("[BUDGET-WARN] non-essential stage blocked near budget limit.")
        return True

    def _lock(self):
        import time as _t
        lock = self.path.with_suffix(".lock")
        for _ in range(600):
            try:
                fd = lock.open("x")
                fd.close()
                return lock
            except (FileExistsError, PermissionError, OSError):
                _t.sleep(0.2)
        raise RuntimeError(f"ledger lock timeout: {lock}")

    @staticmethod
    def _unlock(lock) -> None:
        try:
            lock.unlink()
        except Exception:
            pass

    def append(self, entry: dict) -> None:
        entry["cumulative_cost_cny"] = round(self.cumulative + float(entry.get("cost_cny", 0.0)), 4)
        entry["budget_remaining_cny"] = round(BUDGET_HARD_LIMIT - entry["cumulative_cost_cny"], 4)
        entry["timestamp_utc"] = entry.get("timestamp_utc") or utc_now()
        lock = self._lock()
        try:
            self.rows.append(entry)
            self.cumulative = entry["cumulative_cost_cny"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            self._unlock(lock)

    def summary(self) -> dict:
        by_stage: dict[str, dict] = {}
        for r in self.rows:
            st = r.get("stage", "?")
            d = by_stage.setdefault(st, {"calls": 0, "cost_cny": 0.0, "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["cost_cny"] = round(d["cost_cny"] + float(r.get("cost_cny", 0.0)), 4)
            d["input_tokens"] += int(r.get("input_tokens", 0) or 0)
            d["output_tokens"] += int(r.get("output_tokens", 0) or 0)
        return {"v1_cost_cny": V1_COST_CNY, "cumulative_cost_cny": round(self.cumulative, 4),
                "budget_hard_limit": BUDGET_HARD_LIMIT, "remaining_cny": round(self.remaining(), 4),
                "by_stage": by_stage}

    def write_summary(self) -> None:
        write_json(BUDGET_DIR / "cost_summary.json", self.summary())
        write_json(BUDGET_DIR / "budget_gate.json", {
            "hard_limit_cny": BUDGET_HARD_LIMIT, "v1_cost_cny": V1_COST_CNY,
            "cumulative_cny": round(self.cumulative, 4), "remaining_cny": round(self.remaining(), 4),
            "status": "OK" if self.cumulative <= BUDGET_HARD_LIMIT else "HARD_STOP_EXCEEDED",
        })
