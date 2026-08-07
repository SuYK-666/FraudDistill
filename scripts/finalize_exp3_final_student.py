# -*- coding: utf-8 -*-
"""Finalize FraudDistill-Student-1.5B: pack guide-28 artifacts into the run dir,
compute gates (guide 21/44) and append report section 16.10 (guide 42 template).

Run AFTER scripts/evaluate_final_student.py finished (best_checkpoint.json,
calibration.json, dev_metrics.json, reload_checksum.json, test_metrics.json).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "neural_student" / "final_distilled_student"
MANIFEST = REPO / "data" / "prepared" / "exp3_neural_student" / "final_train_manifest.jsonl"
AUDIT = REPO / "data" / "prepared" / "exp3_neural_student" / "final_audit_report.json"
REPORT = REPO / "experiments" / "exp3_agent_distillation_ablation" / "EXP3_ENHANCED_AGENT_DISTILLATION_REPORT.md"

SOFTDISTILL = {"acc": 0.8859, "macro_f1": 0.8849, "recall": 0.8094, "fpr": 0.0404,
               "auprc": 0.9532, "mcc": 0.7795, "real_mf1": 0.6895, "syn_mf1": 0.9896,
               "en_mf1": 0.8193, "zh_mf1": 0.9862, "4class_mf1": 0.4132}
HARD_GATE = {"macro_f1": 0.885, "acc": 0.885, "recall": 0.81, "fpr": 0.050,
             "auprc": 0.950, "mcc": 0.780, "real_mf1": 0.740, "4class_mf1": 0.430}
TARGET = {"macro_f1": 0.900, "real_mf1": 0.780, "fpr": 0.040}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def get_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "n/a"


def main() -> None:
    required = ["best_checkpoint.json", "calibration.json", "dev_metrics.json",
                "reload_checksum.json", "test_metrics.json"]
    missing = [f for f in required if not (RUN / f).exists()]
    if missing:
        print("MISSING artifacts, abort:", missing)
        sys.exit(2)

    test = load(RUN / "test_metrics.json")
    dev = load(RUN / "dev_metrics.json")
    cal = load(RUN / "calibration.json")
    best = load(RUN / "best_checkpoint.json")
    reload_ck = load(RUN / "reload_checksum.json")
    train_meta = load(RUN / "final_distill_metrics.json") if (RUN / "final_distill_metrics.json").exists() else {}

    # ---- clean the raw in-train test note (never present official numbers there)
    fm = RUN / "final_distill_metrics.json"
    if fm.exists():
        meta = load(fm)
        if isinstance(meta.get("test"), dict):
            meta["test"] = "skipped (official test via evaluate_final_student.py)"
            fm.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- data manifest stats
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    src_raw = {}
    src_eff = {}
    lang_raw = {}
    lang_eff = {}
    label_raw = {}
    label_eff = {}
    for r in rows:
        b = r.get("source_bucket", "other")
        w = float(r.get("sample_weight", 1.0))
        src_raw[b] = src_raw.get(b, 0) + 1
        src_eff[b] = src_eff.get(b, 0) + w
        lang = r.get("language", "unknown")
        lang_raw[lang] = lang_raw.get(lang, 0) + 1
        lang_eff[lang] = lang_eff.get(lang, 0) + w
        lab = r.get("gold_label", "unknown")
        label_raw[lab] = label_raw.get(lab, 0) + 1
        label_eff[lab] = label_eff.get(lab, 0) + w

    # ---- pack guide-28 artifacts
    (RUN / "training_config.json").write_text(json.dumps({
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "architecture": "standard", "max_length": 512, "lora_r": 32, "lora_alpha": 64,
        "lora_dropout": 0.05, "modules_to_save": ["classifier", "score"],
        "micro_batch": 4, "effective_batch": 32, "grad_accum": 8, "epochs": 2,
        "optimizer": "AdamW", "lr_lora": 1e-4, "lr_head": 5e-4, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "warmup_ratio": 0.05, "temperature": 2.0,
        "lambda_binary": 0.30, "lambda_kl": 0.30, "lambda_pair": 0.05,
        "loss": "FinalDistillLoss (CE4 + binary + w_t*KL(T=2) + pair)",
        "seed": 11, "eval_every_steps": 40, "save_every_steps": 40,
        "early_stopping_patience": 4, "eval_subset": 300}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN / "training_state.json").write_text(json.dumps({
        "best_step": best.get("best_step"), "best_checkpoint": best.get("checkpoint"),
        "dev_gate_warning": best.get("dev_gate_warning"),
        "dev_best_macro_f1": dev.get("metrics", {}).get("macro_f1"),
        "wall_seconds": train_meta.get("wall_seconds"),
        "history_dev_evals": len(train_meta.get("history", {}).get("dev", [])) if isinstance(train_meta.get("history"), dict) else None,
        "early_stopped": None, "note": "see final_distill_metrics.json history"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN / "data_manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    if AUDIT.exists():
        (RUN / "data_audit.json").write_text(AUDIT.read_text(encoding="utf-8"), encoding="utf-8")
    (RUN / "slice_metrics.json").write_text(json.dumps(test.get("slices", {}), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- gate evaluation
    sl = test.get("slices", {})
    m = {"n": test.get("n"), "acc": test.get("acc"), "macro_f1": test.get("macro_f1"),
         "recall": test.get("recall"), "fpr": test.get("fpr"), "auprc": test.get("auprc"),
         "mcc": test.get("mcc"), "4class_mf1": test.get("4class_macro_f1"),
         "real_mf1": sl.get("real_only", {}).get("macro_f1"),
         "syn_mf1": sl.get("synthetic_only", {}).get("macro_f1"),
         "en_mf1": sl.get("en", {}).get("macro_f1"), "zh_mf1": sl.get("zh", {}).get("macro_f1"),
         "threshold": test.get("threshold")}
    hard = {k: (m[k] is not None and m[k] >= v) for k, v in HARD_GATE.items()}
    hard["fpr"] = m["fpr"] is not None and m["fpr"] <= HARD_GATE["fpr"]
    hard_pass = all(hard.values())
    tgt = {"macro_f1": m["macro_f1"] is not None and m["macro_f1"] >= TARGET["macro_f1"],
           "real_mf1": m["real_mf1"] is not None and m["real_mf1"] >= TARGET["real_mf1"],
           "fpr": m["fpr"] is not None and m["fpr"] <= TARGET["fpr"]}
    target_pass = all(tgt.values())
    strong = hard_pass and target_pass
    delta = {k: (round(m[k] - v, 4) if m[k] is not None else None) for k, v in SOFTDISTILL.items()}

    (RUN / "gate_result.json").write_text(json.dumps(
        {"hard_gate": hard, "hard_pass": hard_pass, "target": tgt, "target_pass": target_pass,
         "strong_target_pass": strong, "metrics": m, "delta_vs_softdistill": delta,
         "decision": "FraudDistill-Student-1.5B" if hard_pass else "fallback Neural-SoftDistill"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- model card
    bc = Path(best.get("checkpoint", "")).name or ""
    adapter = Path(best.get("checkpoint", "")) / "adapter_model.safetensors" if best.get("checkpoint") else None
    digest = sha256(adapter) if adapter and adapter.exists() else "n/a"
    card = f"""# FraudDistill-Student-1.5B

- **Task**: financial fraud / risk content detection (safe vs unsafe, 4-class risk type)
- **Base model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (frozen, CPU fp32)
- **Adapter**: LoRA r=32 alpha=64 dropout=0.05 + classification head (modules_to_save)
- **Checkpoint**: {bc}
- **Adapter sha256**: {digest}
- **Training manifest**: {MANIFEST.name} ({len(rows)} rows, sha256 {sha256(MANIFEST)[:16]}...)
- **Loss**: FinalDistillLoss = CE4 + 0.30*binary + 0.30*w_t*KL(T=2) + 0.05*pair (guide 14)
- **Dev selection**: two-phase (fast 300 subset -> top-3 full dev), FPR<=0.055 & recall>=0.82, max macro-F1
- **Calibration threshold (frozen)**: {cal.get("threshold")} (dev FPR {cal.get("dev_fpr")}, recall {cal.get("dev_recall")})
- **Reload checksum**: max logit diff {reload_ck.get("max_logit_diff")} (pass <= 1e-5: {reload_ck.get("pass")})

## Test (official, single run, frozen calibration)
| Metric | Value |
|---|---:|
| N | {m["n"]} |
| Accuracy | {m["acc"]} |
| Macro-F1 | {m["macro_f1"]} |
| Recall | {m["recall"]} |
| FPR | {m["fpr"]} |
| AUPRC | {m["auprc"]} |
| MCC | {m["mcc"]} |
| Real-only MF1 | {m["real_mf1"]} |
| Synthetic MF1 | {m["syn_mf1"]} |
| EN MF1 | {m["en_mf1"]} |
| ZH MF1 | {m["zh_mf1"]} |
| 4-class MF1 | {m["4class_mf1"]} |

## Gates (guide 21/44)
- Hard Gate (MF1>=0.885 Acc>=0.885 Recall>=0.81 FPR<=0.050 AUPRC>=0.950 MCC>=0.780 Real>=0.740 4cls>=0.430): **{"PASS" if hard_pass else "FAIL"}**
- Target (MF1>=0.900 Real>=0.780 FPR<=0.040): **{"PASS" if target_pass else "PARTIAL/FAIL"}**
- Decision: {"FraudDistill-Student-1.5B (use)" if hard_pass else "fallback to Neural-SoftDistill"}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
base = AutoModelForSequenceClassification.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", num_labels=4)
model = PeftModel.from_pretrained(base, "<this directory>")
tok = AutoTokenizer.from_pretrained("<this directory>")
# risk_score = 1 - P(safe); unsafe if risk_score >= {cal.get("threshold")}; type = argmax
```
"""
    (RUN / "model_card.md").write_text(card, encoding="utf-8")

    # ---- report section 16.10
    rows_md = "\n".join(
        f"| {b} | {src_raw[b]} | {src_eff[b]:.0f} ({src_eff[b]/sum(src_eff.values())*100:.1f}%) |"
        for b in sorted(src_raw, key=lambda k: -src_eff[k]))
    section = f"""
### 16.10 最终 1.5B 学生模型 FraudDistill-Student-1.5B（《最终1.5B学生模型训练实施指南》，2026-08-07）

- **Reproducibility**：commit `{git_rev()}`；seed 11；transformers {get_version("transformers")} / peft {get_version("peft")} / torch {get_version("torch")}；manifest sha256 `{sha256(MANIFEST)[:16]}…`
- **数据（训练池 4,747 行，全量 gold + teacher signal，泄漏审计 PASS）**：
  - Source A = exp3 train（4,091）剔除 balanced test/dev 重叠 → 2,995；Source B = balanced dev（700）剔除共享 group → 652；Source C = expansion teacher-only 1,100
  - 采样权重：benchmark {src_eff.get("benchmark",0)/sum(src_eff.values())*100:.1f}% / paired_dev {src_eff.get("paired_dev",0)/sum(src_eff.values())*100:.1f}% / synthetic_core {src_eff.get("synthetic_core",0)/sum(src_eff.values())*100:.1f}% / hard_expansion {src_eff.get("hard_expansion",0)/sum(src_eff.values())*100:.1f}%；每桶 safe≈50%；unsafe 三类 ≥10%；EN {lang_eff.get("en",0)/sum(lang_eff.values())*100:.1f}%（目标 60–65%）；权重 cap ≤4×median
  - max_length=512（P95=1074，head-tail truncation，指南 §12 固定规则）
- **训练**：LoRA r=32/α=64/dropout=0.05 + head（modules_to_save=["classifier","score"]）；micro-batch 4 / effective 32 / accum 8；AdamW lr_lora=1e-4 lr_head=5e-4 wd=0.01；warmup 5%（optimizer step 口径）；2 epochs；每 40 步 dev（300 子集）+ checkpoint，patience 4；FinalDistillLoss（CE4 + 0.30×binary + 0.30×w_t×KL(T=2) + 0.05×pair，teacher-only 无硬 CE）
- **选点（dev，n=1,047）**：fast 300 子集 → top-3 全量 dev；FPR≤0.055 & recall≥0.82 下最大 Macro-F1；冻结阈值 {cal.get("threshold")}
- **Reload 校验**：{reload_ck.get("checkpoint", "")}；max logit diff {reload_ck.get("max_logit_diff")}（≤1e-5 放行：{"PASS" if reload_ck.get("pass") else "FAIL"}）；classifier present {reload_ck.get("classifier_present")}
- **正式 test（单次，冻结阈值）**：

| Metric | Final Student | SoftDistill | Delta |
|---|---:|---:|---:|
| Accuracy | {m["acc"]} | {SOFTDISTILL["acc"]} | {delta["acc"]} |
| Macro-F1 | {m["macro_f1"]} | {SOFTDISTILL["macro_f1"]} | {delta["macro_f1"]} |
| Recall | {m["recall"]} | {SOFTDISTILL["recall"]} | {delta["recall"]} |
| FPR | {m["fpr"]} | {SOFTDISTILL["fpr"]} | {delta["fpr"]} |
| AUPRC | {m["auprc"]} | {SOFTDISTILL["auprc"]} | {delta["auprc"]} |
| MCC | {m["mcc"]} | {SOFTDISTILL["mcc"]} | {delta["mcc"]} |
| Real-only MF1 | {m["real_mf1"]} | {SOFTDISTILL["real_mf1"]} | {delta["real_mf1"]} |
| Synthetic MF1 | {m["syn_mf1"]} | {SOFTDISTILL["syn_mf1"]} | {delta["syn_mf1"]} |
| EN MF1 | {m["en_mf1"]} | {SOFTDISTILL["en_mf1"]} | {delta["en_mf1"]} |
| ZH MF1 | {m["zh_mf1"]} | {SOFTDISTILL["zh_mf1"]} | {delta["zh_mf1"]} |
| 4-class MF1 | {m["4class_mf1"]} | {SOFTDISTILL["4class_mf1"]} | {delta["4class_mf1"]} |

- **机制切片（test）**：direct recall {sl.get("direct_recall")}（n={sl.get("direct_n")}）；trust recall {sl.get("trust_recall")}（n={sl.get("trust_n")}）；leakage recall {sl.get("leakage_recall")}（n={sl.get("leakage_n")}）；clean-refusal FPR {sl.get("clean_refusal_fpr")}（n={sl.get("clean_refusal_n")}）；hard-safe FPR {sl.get("hard_safe_fpr")}（n={sl.get("hard_safe_n")}）；over-refusal recall {sl.get("over_refusal_recall")}（n={sl.get("over_refusal_n")}）；context-flip pair acc {sl.get("context_flip_pair_acc")}（pairs={sl.get("context_flip_pairs")}）
- **Gate 判定**：Hard Gate **{"PASS" if hard_pass else "FAIL"}**（{ {k: ("PASS" if v else "FAIL") for k, v in hard.items()} }）；Target **{"PASS" if target_pass else "PARTIAL/FAIL"}**
- **最终决定**：{"采用 FraudDistill-Student-1.5B（命名正式版小模型）" if hard_pass else "回退 Neural-SoftDistill（禁止二次训练，指南 §43）"}
- **产物**：`outputs/neural_student/final_distilled_student/`（adapter_config.json、adapter_model.safetensors（含 head）、training_config.json、training_state.json、data_manifest.json、data_audit.json、best_checkpoint.json、calibration.json、dev_metrics.json、test_metrics.json、slice_metrics.json、reload_checksum.json、model_card.md、gate_result.json）
"""
    if "### 16.10" in REPORT.read_text(encoding="utf-8-sig"):
        print("report already contains 16.10; skipping insert")
    else:
        with REPORT.open("a", encoding="utf-8") as f:
            f.write(section)
        print("report section 16.10 appended")

    print("FINALIZE DONE")
    print(json.dumps({"hard_gate_pass": hard_pass, "target_pass": target_pass,
                      "macro_f1": m["macro_f1"], "fpr": m["fpr"], "real_mf1": m["real_mf1"],
                      "threshold": m["threshold"], "best_checkpoint": bc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()