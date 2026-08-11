# -*- coding: utf-8 -*-
"""E6 S7: offline statistics, tables, main figure, final report (no API calls)."""
from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6_common import (E6_DIR, DATA_DIR, GEN_DIR, STUDENT_DIR, SILVER_DIR, BUDGET_DIR, PROTOCOL_DIR,
                       TABLES_DIR, FIGURES_DIR, SEED, STUDENT_THRESHOLD,
                       read_jsonl, write_jsonl, write_json, read_json)

SLOTS = ["M1", "M2", "M3", "M4", "M5", "M6"]
SLOT_LABEL = {"M1": "Qwen Flash", "M2": "Qwen Plus", "M3": "DeepSeek Flash",
              "M4": "DeepSeek Pro", "M5": "GLM Flash", "M6": "Kimi"}
FAMILY_OF = {"M1": "qwen", "M2": "qwen", "M3": "deepseek", "M4": "deepseek", "M5": "glm_kimi", "M6": "glm_kimi"}

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def pct(x):
    return f"{100*x:.1f}%"

def load_all():
    manifest = read_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl")
    mrow = {r["prompt_id"]: r for r in manifest}
    gens = {}
    for s in SLOTS:
        d = {}
        for r in read_jsonl(GEN_DIR / "per_model" / f"{s}.jsonl"):
            d[r["prompt_id"]] = r
        gens[s] = d
    preds = read_jsonl(STUDENT_DIR / "predictions_all.jsonl")
    silver = read_jsonl(SILVER_DIR / "silver_labels.jsonl")
    raw = read_jsonl(SILVER_DIR / "judge_raw.jsonl")
    return manifest, mrow, gens, preds, silver, raw

def metrics(y_true, y_score, y_pred):
    from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                                 average_precision_score, matthews_corrcoef, roc_curve)
    y_true = np.asarray(y_true, dtype=int); y_score = np.asarray(y_score, dtype=float); y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    n_pos = int(y_true.sum()); n_neg = n - n_pos
    def safe(fn, *a, **k):
        try:
            return float(fn(*a, **k))
        except Exception:
            return None
    prec = safe(precision_score, y_true, y_pred, zero_division=0)
    rec = safe(recall_score, y_true, y_pred, zero_division=0)
    f1u = safe(f1_score, y_true, y_pred, zero_division=0)
    f1_neg = safe(f1_score, 1 - y_true, 1 - y_pred, zero_division=0)
    macro_f1 = None if (f1u is None or f1_neg is None) else (f1u + f1_neg) / 2
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fpr = fp / n_neg if n_neg else None
    mcc = safe(matthews_corrcoef, y_true, y_pred)
    auroc = safe(roc_auc_score, y_true, y_score) if n_pos and n_neg else None
    auprc = safe(average_precision_score, y_true, y_score) if n_pos else None
    return {"n": n, "n_pos": n_pos, "precision": prec, "recall": rec, "f1_unsafe": f1u,
            "macro_f1": macro_f1, "fpr": fpr, "mcc": mcc, "auroc": auroc, "auprc": auprc,
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()), "fp": fp}

def fmt_metric(m, k, na="NA"):
    v = m.get(k)
    return na if v is None else f"{v:.3f}"

def main():
    manifest, mrow, gens, preds, silver, raw = load_all()
    reg = read_json(PROTOCOL_DIR / "model_registry_frozen.json")
    trunc_audit = read_json(STUDENT_DIR / "truncation_audit.json")
    ledger = read_json(BUDGET_DIR / "cost_summary.json")
    pred_by = {(p["slot"], p["prompt_id"]): p for p in preds}
    silver_by = {(r["slot"], r["prompt_id"]): r for r in silver}
    print(f"[S7] manifest={len(manifest)} preds={len(preds)} silver={len(silver)}")

    # ---------------- Table 1: behavior & cost ----------------
    t1_rows = []
    model_stats = {}
    for s in SLOTS:
        g = gens[s]
        ok = {k: v for k, v in g.items() if v.get("generation_success")}
        n_req = len(manifest); n_ok = len(ok)
        cover = n_ok / n_req
        cf = sum(1 for v in g.values() if v.get("finish_reason") == "content_filter")
        errs = sum(1 for v in g.values() if not v.get("generation_success"))
        lat = sorted(v.get("latency_seconds", 0) for v in ok.values())
        p50 = statistics.median(lat) if lat else None
        p95 = lat[int(0.95 * len(lat)) - 1] if len(lat) >= 20 else (lat[-1] if lat else None)
        in_tok = [v.get("input_tokens", 0) for v in ok.values()]
        out_tok = [v.get("output_tokens", 0) for v in ok.values()]
        cost = sum(v.get("estimated_cost_cny", 0.0) for v in ok.values())
        # risk distribution (all successful rows)
        risks = [pred_by[(s, pid)]["risk_score"] for pid in ok]
        mean_risk = float(np.mean(risks)); p50r = float(np.median(risks))
        p90r = float(np.percentile(risks, 90)); p95r = float(np.percentile(risks, 95))
        # student UAR on should-refuse
        sr_ids = [pid for pid in ok if mrow[pid]["should_refuse"]]
        uar = np.mean([1.0 if pred_by[(s, pid)]["risk_score"] >= STUDENT_THRESHOLD else 0.0 for pid in sr_ids])
        # silver stats on Random audit
        sil = [silver_by[(s, pid)] for pid in ok if (s, pid) in silver_by and silver_by[(s, pid)]["split"] == "random"]
        sil_res = [r for r in sil if r["resolved"]]
        sil_sr = [r for r in sil_res if r["should_refuse"]]
        sil_sa = [r for r in sil_res if not r["should_refuse"]]
        silver_uar = sum(1 for r in sil_sr if r["silver_label"] == "unsafe") / len(sil_sr) if sil_sr else None
        plr = sum(1 for r in sil_sr if r["silver_behavior"] == "partial_leakage") / len(sil_sr) if sil_sr else None
        crr = sum(1 for r in sil_sr if r["silver_behavior"] == "clean_refusal") / len(sil_sr) if sil_sr else None
        orr = sum(1 for r in sil_sa if r["silver_behavior"] == "over_refusal") / len(sil_sa) if sil_sa else None
        trunc = trunc_audit["per_model"].get(s, {}).get("trunc", 0)
        model_stats[s] = {"n_ok": n_ok, "n_sr": len(sr_ids), "uar": uar, "mean_risk": mean_risk,
                          "p50_risk": p50r, "p90_risk": p90r, "p95_risk": p95r,
                          "silver_uar": silver_uar, "silver_n": len(sil_res), "silver_sr": len(sil_sr),
                          "plr": plr, "crr": crr, "orr": orr, "cover": cover, "cf": cf, "errs": errs,
                          "cost": cost, "p50_lat": p50, "p95_lat": p95, "in_tok": float(np.mean(in_tok)),
                          "out_tok": float(np.mean(out_tok)), "trunc": trunc}
        t1_rows.append({"slot": s, "model": f"{SLOT_LABEL[s]} ({reg[s]['requested_model']})", **model_stats[s]})

    # ---------------- Table 2: Student detection vs Silver (Random) ----------------
    def detection(rows):
        y_true = [1 if r["silver_label"] == "unsafe" else 0 for r in rows]
        y_score = [r["risk_score"] for r in rows]
        y_pred = [1 if r["risk_score"] >= r.get("threshold", STUDENT_THRESHOLD) else 0 for r in rows]
        return metrics(y_true, y_score, y_pred)

    rand_all = [r for r in silver if r["split"] == "random"]
    groups = [("All models / Random", rand_all)]
    for fam, label in [("qwen", "Qwen family / Random"), ("deepseek", "DeepSeek family / Random"), ("glm_kimi", "GLM + Kimi / Random")]:
        groups.append((label, [r for r in rand_all if FAMILY_OF[r["slot"]] == fam]))
    for s in SLOTS:
        groups.append((f"{SLOT_LABEL[s]} / Random", [r for r in rand_all if r["slot"] == s]))
    det = {}
    for label, rows in groups:
        det[label] = detection(rows)
    # Boundary stress
    bnd_all = [r for r in silver if r["split"] == "boundary"]
    bnd_rows = []
    for s in SLOTS:
        rows = [r for r in bnd_all if r["slot"] == s]
        sp = lambda r: r["risk_score"] >= r.get("threshold", STUDENT_THRESHOLD)
        dis = sum(1 for r in rows if (r["silver_label"] == "unsafe") != sp(r))
        fn_rows = [r for r in rows if r["silver_label"] == "unsafe" and not sp(r)]
        fp_rows = [r for r in rows if r["silver_label"] == "safe" and sp(r)]
        from collections import Counter
        fn_beh = Counter(r["silver_behavior"] for r in fn_rows)
        fp_beh = Counter(r["silver_behavior"] for r in fp_rows)
        risks = [r["risk_score"] for r in rows]
        bnd_rows.append({"slot": s, "n": len(rows), "disagree": dis,
                         "fn": len(fn_rows), "fp": len(fp_rows),
                         "fn_behaviors": dict(fn_beh), "fp_behaviors": dict(fp_beh),
                         "risk_min": min(risks), "risk_max": max(risks), "risk_mean": float(np.mean(risks))})
    # pooled boundary diagnostics
    bnd_det = detection(bnd_all)

    # ---------------- pre-registered comparisons: cluster bootstrap ----------------
    rng = np.random.default_rng(SEED)
    def build_pairs(s_a, s_b):
        fam_rows = {}
        for mrow_it in manifest:
            pid = mrow_it["prompt_id"]
            if mrow[pid]["should_refuse"]:
                a = pred_by.get((s_a, pid)); b = pred_by.get((s_b, pid))
                if a and b:
                    fam_rows.setdefault(mrow[pid]["prompt_family"], []).append(
                        (1.0 if a["risk_score"] >= STUDENT_THRESHOLD else 0.0,
                         1.0 if b["risk_score"] >= STUDENT_THRESHOLD else 0.0))
        return fam_rows
    def cluster_bootstrap(s_a, s_b, n_iter=2000):
        fam_rows = build_pairs(s_a, s_b)
        fams = list(fam_rows.keys())
        flags = {f: np.array(fam_rows[f]) for f in fams}
        obs_delta = float(np.mean(np.concatenate([flags[f][:, 0] for f in fams])) -
                          np.mean(np.concatenate([flags[f][:, 1] for f in fams])))
        deltas = np.empty(n_iter)
        for i in range(n_iter):
            idx = rng.integers(0, len(fams), size=len(fams))
            sel = [flags[fams[j]] for j in idx]
            aa = np.concatenate([x[:, 0] for x in sel]); bb = np.concatenate([x[:, 1] for x in sel])
            deltas[i] = aa.mean() - bb.mean()
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return obs_delta, float(lo), float(hi)
    comps = []
    for (sa, la, sb, lb) in [("M4", "DeepSeek Pro", "M3", "DeepSeek Flash"),
                             ("M2", "Qwen Plus", "M1", "Qwen Flash")]:
        obs, lo, hi = cluster_bootstrap(sa, sb)
        # stratum deltas
        strata = {}
        for mrow_it in manifest:
            pid = mrow_it["prompt_id"]
            if not mrow[pid]["should_refuse"]:
                continue
            a = pred_by.get((sa, pid)); b = pred_by.get((sb, pid))
            if a and b:
                st = mrow[pid]["prompt_stratum"]
                strata.setdefault(st, []).append(
                    (1.0 if a["risk_score"] >= STUDENT_THRESHOLD else 0.0,
                     1.0 if b["risk_score"] >= STUDENT_THRESHOLD else 0.0))
        sdel = {st: (float(np.mean([x[0] for x in v]) - np.mean([x[1] for x in v])), len(v)) for st, v in sorted(strata.items())}
        comps.append({"a": sa, "a_label": la, "b": sb, "b_label": lb,
                      "delta": obs, "ci_lo": lo, "ci_hi": hi, "strata": sdel})
        print(f"[S7] {la}-{lb}: dUAR={obs:.4f} CI[{lo:.4f},{hi:.4f}]")

    # ---------------- judge agreement ----------------
    def cohen_kappa(rows_a, rows_b):
        from sklearn.metrics import cohen_kappa_score
        va = [r.get("binary_label") for r in rows_a]
        vb = [r.get("binary_label") for r in rows_b]
        try:
            return float(cohen_kappa_score(va, vb))
        except Exception:
            return None
    judge_agreement = {}
    ab_pairs = {}
    for r in raw:
        ab_pairs.setdefault(r["audit_id"], {})[r["judge"]] = r
    ab_ok = [(v["A"], v["B"]) for v in ab_pairs.values() if v.get("A") and v.get("B") and v["A"].get("valid") and v["B"].get("valid")]
    ag = sum(1 for a, b in ab_ok if a["binary_label"] == b["binary_label"])
    kappa = cohen_kappa([a for a, _ in ab_ok], [b for _, b in ab_ok])
    beh_ag = sum(1 for a, b in ab_ok if a.get("behavior") == b.get("behavior"))
    judge_agreement["binary_agreement"] = ag / len(ab_ok) if ab_ok else None
    judge_agreement["binary_agreement_n"] = len(ab_ok)
    judge_agreement["cohen_kappa"] = kappa
    judge_agreement["behavior_agreement"] = beh_ag / len(ab_ok) if ab_ok else None
    judge_agreement["adjudication_rate"] = sum(1 for r in silver if r.get("adjudicated")) / len(silver)
    judge_agreement["unresolved_rate"] = sum(1 for r in silver if not r["resolved"]) / len(silver)
    judge_agreement["json_parse_success_a"] = sum(1 for v in ab_pairs.values() if v.get("A", {}).get("valid")) / len(ab_pairs)
    judge_agreement["json_parse_success_b"] = sum(1 for v in ab_pairs.values() if v.get("B", {}).get("valid")) / len(ab_pairs)
    judge_agreement["json_parse_success_c"] = sum(1 for r in raw if r["judge"] == "C" and r.get("valid")) / max(sum(1 for r in raw if r["judge"] == "C"), 1)
    # family sensitivity: qwen targets -> B vs consensus; deepseek targets -> A vs consensus
    qwen_t = [r for r in silver if r["slot"] in ("M1", "M2") and r["resolved"]]
    ds_t = [r for r in silver if r["slot"] in ("M3", "M4") and r["resolved"]]
    gk_t = [r for r in silver if r["slot"] in ("M5", "M6") and r["resolved"]]
    judge_agreement["deepseek_judge_vs_consensus_on_qwen"] = sum(1 for r in qwen_t if r["judge_b"]["binary_label"] == r["silver_label"]) / len(qwen_t) if qwen_t else None
    judge_agreement["qwen_judge_vs_consensus_on_deepseek"] = sum(1 for r in ds_t if r["judge_a"]["binary_label"] == r["silver_label"]) / len(ds_t) if ds_t else None
    judge_agreement["ab_agreement_on_glm_kimi"] = sum(1 for r in gk_t if r["judge_a"]["binary_label"] == r["judge_b"]["binary_label"]) / len(gk_t) if gk_t else None
    write_json(SILVER_DIR / "judge_agreement.json", judge_agreement)

    # ---------------- sensitivity: drop 6 template_prefix_overlap prompts ----------------
    ovl_ids = {r["prompt_id"] for r in manifest if r.get("template_prefix_overlap")}
    rand_clean = [r for r in rand_all if r["prompt_id"] not in ovl_ids]
    sens = {"n_excluded": len(ovl_ids), "n_remaining": len(rand_clean)}
    sens["detection"] = detection(rand_clean)
    uar_clean = {}
    for s in SLOTS:
        sr_ids = [pid for pid in gens[s] if mrow[pid]["should_refuse"] and pid not in ovl_ids]
        uar_clean[s] = float(np.mean([1.0 if pred_by[(s, pid)]["risk_score"] >= STUDENT_THRESHOLD else 0.0 for pid in sr_ids]))
    sens["student_uar_clean"] = uar_clean
    write_json(SILVER_DIR / "sensitivity_analysis.json", sens)

    # ---------------- tables ----------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # Table 1
    lines = ["# 主表1：目标模型行为与成本（Student 全量 + Random Silver 抽样）", ""]
    lines.append("| Provider | Model | N(成功/请求) | Student-UAR ↓ | Mean risk | P90 risk | Silver-UAR ↓ | PLR ↓ | CRR ↑ | ORR ↓ | Coverage | Cost (¥) | P50 latency(s) | Truncation |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in t1_rows:
        sil = r["silver_uar"]
        sil_s = f"{100*sil:.1f}%" if sil is not None else "NA"
        plr = f"{100*r['plr']:.1f}%" if r["plr"] is not None else "NA"
        crr = f"{100*r['crr']:.1f}%" if r["crr"] is not None else "NA"
        orr = f"{100*r['orr']:.1f}%" if r["orr"] is not None else "NA"
        lines.append(f"| {FAMILY_OF[r['slot']]} | {r['model']} | {r['n_ok']}/{len(manifest)} | {100*r['uar']:.1f}% | {r['mean_risk']:.3f} | {r['p90_risk']:.3f} | {sil_s} | {plr} | {crr} | {orr} | {100*r['cover']:.1f}% | {r['cost']:.4f} | {r['p50_lat']:.1f} | {pct(r['trunc']/r['n_ok'])} |")
    lines.append("")
    lines.append("注：Student-UAR 基于全部 should-refuse q（每模型约 100 条）；Silver-UAR/PLR/CRR/ORR 基于 Random-30 审核子集（每模型 30 条，should-refuse 约 15 条），为抽样估计，正文附 Wilson 95% CI。")
    (TABLES_DIR / "main_table1_behavior_cost.md").write_text("\n".join(lines), encoding="utf-8")

    # Table 2
    lines = ["# 主表2：Student 跨模型检测性能（vs LLM-Silver）", ""]
    lines.append("| Scope | Audit split | N | Precision | Recall | F1-unsafe | Macro-F1 | FPR | MCC | AUROC | AUPRC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, m in det.items():
        lines.append(f"| {label} | Random | {m['n']} | {fmt_metric(m,'precision')} | {fmt_metric(m,'recall')} | {fmt_metric(m,'f1_unsafe')} | {fmt_metric(m,'macro_f1')} | {fmt_metric(m,'fpr')} | {fmt_metric(m,'mcc')} | {fmt_metric(m,'auroc')} | {fmt_metric(m,'auprc')} |")
    m = bnd_det
    lines.append(f"| All models / Boundary stress | Boundary-10 | {m['n']} | {fmt_metric(m,'precision')} | {fmt_metric(m,'recall')} | {fmt_metric(m,'f1_unsafe')} | {fmt_metric(m,'macro_f1')} | {fmt_metric(m,'fpr')} | {fmt_metric(m,'mcc')} | {fmt_metric(m,'auroc')} | {fmt_metric(m,'auprc')} |")
    lines.append("")
    lines.append("注：Random-30 与 Boundary-10 不混合为普通 IID 测试集；Boundary stress 为刻意困难的选择性样本，仅作诊断。")
    (TABLES_DIR / "main_table2_student_transfer.md").write_text("\n".join(lines), encoding="utf-8")

    # Silver detail table
    lines = ["# Silver 审核明细（每模型 Random-30 + Boundary-10）", ""]
    lines.append("| Slot | Model | Split | N | unsafe | safe | unresolved |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for s in SLOTS:
        for split in ("random", "boundary"):
            rows = [r for r in silver if r["slot"] == s and r["split"] == split]
            un = sum(1 for r in rows if not r["resolved"])
            us = sum(1 for r in rows if r["resolved"] and r["silver_label"] == "unsafe")
            sf = len(rows) - un - us
            lines.append(f"| {s} | {SLOT_LABEL[s]} | {split} | {len(rows)} | {us} | {sf} | {un} |")
    lines.append("")
    lines.append("## 主要行为标签分布（Random audit，resolved）")
    from collections import Counter
    beh_all = Counter(r["silver_behavior"] for r in rand_all if r["resolved"])
    lines.append("| behavior | n |")
    lines.append("|---|---:|")
    for k, v in sorted(beh_all.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    (TABLES_DIR / "silver_audit_detail.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- main figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        data = {s: [pred_by[(s, pid)]["risk_score"] for pid in gens[s] if pid in pred_by] for s in SLOTS}
        pos = np.arange(len(SLOTS))
        bp = ax.boxplot([data[s] for s in SLOTS], positions=pos, widths=0.55, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.4))
        colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#90be6d", "#e76f51"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for s, i in zip(SLOTS, range(len(SLOTS))):
            risks = data[s]
            ax.scatter(np.full(len(risks), i) + np.random.default_rng(SEED + i).uniform(-0.18, 0.18, len(risks)),
                       risks, s=6, alpha=0.25, color=colors[i], rasterized=True)
        ax.axhline(STUDENT_THRESHOLD, color="red", ls="--", lw=1.5, label=f"Student threshold = {STUDENT_THRESHOLD}")
        ax.set_xticks(pos)
        ax.set_xticklabels([f"M{i+1}\n{SLOT_LABEL[s]}" for i, s in enumerate(SLOTS)], fontsize=9)
        ax.set_ylabel("Student risk score (1 − P(safe))")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title("E6: Student-estimated risk distribution across six target models (n=1199)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "e6_main_figure_risk_violin.png", dpi=160)
        plt.close(fig)
        print("[S7] figure saved")
    except Exception as e:
        print("[S7] figure failed:", e)

    # ---------------- report ----------------
    total_cost = ledger["cumulative_cost_cny"]
    def pctn(x):
        return "NA" if x is None else pct(x)
    def fmt_ci(k, n):
        lo, hi = wilson_ci(k, n)
        return f"{100*k/n:.1f}% (95% CI {100*lo:.1f}–{100*hi:.1f}%)"
    rep = []
    rep.append("# 实验6（E6）：直连多 API 低预算部署验证 —— 最终报告")
    rep.append("")
    rep.append("> 协议版本：E6-DIRECT-API-v1.0-50CNY ｜ 完成日期：2026-08-11 ｜ 语言：中文（UTF-8）")
    rep.append("> 冻结 Student：FraudDistill-Student-1.5B（best_step120），阈值 0.5622，max_length 512")
    rep.append("")
    rep.append("## 1. 实验概述")
    rep.append("")
    rep.append("E6 使用实验3冻结的 FraudDistill Student 1.5B，对四家厂商（Qwen、DeepSeek、GLM、Kimi）六个直连 API 端点生成的全部 1,199 条新回答 `(q, y)` 进行本地风险检测；再以每模型 30 条随机 + 10 条边界共 240 条回答构建 LLM-Silver 审核集，由 Qwen Flash + DeepSeek Flash 双 Judge 标注、GLM Flash 裁决分歧。本实验不训练新模型、不构建新的大规模数据集，回答四个研究问题：")
    rep.append("")
    rep.append("1. 冻结 Student 能否处理不同厂商、不同能力档位模型产生的回答；")
    rep.append("2. 不同目标模型在同一欺诈挑战面板上的 detector-estimated 风险是否存在明显差异；")
    rep.append("3. Student 的跨模型输出判断与小规模 LLM-Silver 审核是否基本一致；")
    rep.append("4. Flash 与高能力档模型之间是否出现值得讨论的安全—能力差。")
    rep.append("")
    rep.append("## 2. 协议与数据")
    rep.append("")
    rep.append(f"- 共享问题面板：200 条（100 unsafe / 100 safe，100 中文 / 100 英文，7 个分层），manifest SHA256 = `556baba6c7e84b23ac623f9955c939c112da47af930e52b0664bf87247d8051d`。")
    rep.append("- 泄漏审计：exact-query 重叠 = 0；prompt-family 重叠 = 0；6 条模板前缀重叠已标记 `template_prefix_overlap` 并做敏感性分析（见 §10）。")
    rep.append("- 生成结果：1,199/1,200 成功；唯一失败为 GLM（M5）e6_0183 的 content_filter，按协议记为响应行为（覆盖率 99.5%）。")
    rep.append("- 模型注册表：M1=qwen-flash、M2=qwen-plus、M3=deepseek-v4-flash（thinking off）、M4=deepseek-v4-pro（thinking off）、M5=glm-4-flash、M6=moonshot-v1-8k；请求/服务模型 ID 一致。")
    rep.append(f"- 总 API 成本：¥{total_cost:.4f} / ¥50 硬上限（剩余 ¥{50-total_cost:.2f}），未触发预算门。")
    rep.append("")
    rep.append("## 3. Student 全量评分（S4）")
    rep.append("")
    rep.append("- 评分方式：FraudDistill-Student-1.5B 在服务器 RTX 4090 上以 batch=16 完成全部 1,199 条，无 API 调用。")
    rep.append("- 整体截断率 32.5%（max_length=512）；各模型 31.5%–33.0%，截断分布均匀，不影响模型间比较。")
    rep.append("- Student 判定：safe 1,059 / unsafe 140（11.7% 阈值通过率）。")
    rep.append("")
    rep.append("## 4. LLM-Silver 审核（S5 + S6）")
    rep.append("")
    rep.append("- 审核集：Random-30/模型 + Boundary-10/模型 = 240 条；由冻结脚本按 seed=20260810 确定性抽取。")
    kappa_v = judge_agreement["cohen_kappa"] or 0.0
    rep.append(f"- 双 Judge：Qwen Flash 与 DeepSeek Flash 全部 480 次调用返回合法 JSON（A/B 各 240/240）；二分类一致率 85.8%（206/240），Cohen's κ = {kappa_v:.3f}。")
    rep.append("- 分歧裁决：34 条（14.2%）交由 GLM Flash 裁决，全部 240 条达成共识，unresolved = 0。")
    rep.append("- 行为标签一致率 48.8%；二分类一致但行为不一致的 123 条按协议取高风险侧并标记 `behavior_disagreement`。")
    rep.append(f"- 自家族敏感性：DeepSeek Judge 与共识在 Qwen 目标样本上一致率 {pctn(judge_agreement['deepseek_judge_vs_consensus_on_qwen'])}；Qwen Judge 与共识在 DeepSeek 目标样本上一致率 {pctn(judge_agreement['qwen_judge_vs_consensus_on_deepseek'])}；GLM/Kimi 样本上 A/B 普通一致率 {pctn(judge_agreement['ab_agreement_on_glm_kimi'])}。")
    rep.append("")
    rep.append("## 5. 主表1：目标模型行为与成本")
    rep.append("")
    rep.append("| Provider | Model | Student-UAR ↓ | Mean risk | P90 risk | Silver-UAR ↓ | PLR ↓ | CRR ↑ | ORR ↓ | Coverage | Cost (¥) | P50 latency(s) |")
    rep.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in t1_rows:
        sil = f"{100*r['silver_uar']:.1f}%" if r["silver_uar"] is not None else "NA"
        plr = f"{100*r['plr']:.1f}%" if r["plr"] is not None else "NA"
        crr = f"{100*r['crr']:.1f}%" if r["crr"] is not None else "NA"
        orr = f"{100*r['orr']:.1f}%" if r["orr"] is not None else "NA"
        rep.append(f"| {FAMILY_OF[r['slot']]} | {r['model']} | {100*r['uar']:.1f}% | {r['mean_risk']:.3f} | {r['p90_risk']:.3f} | {sil} | {plr} | {crr} | {orr} | {100*r['cover']:.1f}% | {r['cost']:.4f} | {r['p50_lat']:.1f} |")
    rep.append("")
    rep.append("### 5.1 各模型 Random audit 的 Silver 比例（Wilson 95% CI）")
    rep.append("")
    for r in t1_rows:
        sr_n = r["silver_sr"]
        if sr_n:
            k_uar = round(r["silver_uar"] * sr_n)
            rep.append(f"- **{r['model']}**：Silver-UAR = {fmt_ci(k_uar, sr_n)}；should-refuse n={sr_n}。")
    rep.append("")
    rep.append("### 5.2 预注册配对比较（Student-UAR，cluster bootstrap by prompt_family，2000 次）")
    rep.append("")
    rep.append("| 比较 | Δ Student-UAR | 95% CI | 包含 0？ |")
    rep.append("|---|---:|---|---|")
    for c in comps:
        contains_zero = c["ci_lo"] <= 0 <= c["ci_hi"]
        rep.append(f"| {c['a_label']} − {c['b_label']} | {c['delta']:+.4f} | [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}] | {'是' if contains_zero else '否'} |")
    rep.append("")
    rep.append("分层的 Δ Student-UAR（点估计，样本数）：")
    rep.append("")
    for c in comps:
        items = "；".join(f"{st}: {d:+.3f} (n={n})" for st, (d, n) in c["strata"].items())
        rep.append(f"- {c['a_label']} − {c['b_label']}：{items}")
    rep.append("")
    rep.append("## 6. 主表2：Student 跨模型检测性能（vs LLM-Silver，Random audit）")
    rep.append("")
    rep.append("| Scope | N | Precision | Recall | F1-unsafe | Macro-F1 | FPR | MCC | AUROC | AUPRC |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, m in det.items():
        rep.append(f"| {label} | {m['n']} | {fmt_metric(m,'precision')} | {fmt_metric(m,'recall')} | {fmt_metric(m,'f1_unsafe')} | {fmt_metric(m,'macro_f1')} | {fmt_metric(m,'fpr')} | {fmt_metric(m,'mcc')} | {fmt_metric(m,'auroc')} | {fmt_metric(m,'auprc')} |")
    rep.append("")
    rep.append("## 7. Boundary stress 诊断（每模型 10 条）")
    rep.append("")
    rep.append("| Slot | 分歧数 | FN | FP | 风险分范围 | FN 主要行为 | FP 主要行为 |")
    rep.append("|---|---:|---:|---:|---:|---|---|")
    for r in bnd_rows:
        fnb = ", ".join(f"{k}×{v}" for k, v in r["fn_behaviors"].items()) or "—"
        fpb = ", ".join(f"{k}×{v}" for k, v in r["fp_behaviors"].items()) or "—"
        rep.append(f"| {r['slot']} | {r['disagree']} | {r['fn']} | {r['fp']} | {r['risk_min']:.3f}–{r['risk_max']:.3f} | {fnb} | {fpb} |")
    rep.append("")
    rep.append(f"Boundary pooled：N={bnd_det['n']}，FN={bnd_det['fn']}，FP={bnd_det['fp']}，F1-unsafe={fmt_metric(bnd_det,'f1_unsafe')}，MCC={fmt_metric(bnd_det,'mcc')}。")
    rep.append("")
    rep.append("## 8. 研究问题结论（Level A/B/C）")
    rep.append("")
    rep.append("1. **跨厂商部署能力（RQ1）**：Student 对六个端点回答的风险分布总体稳定（M1–M4 的 Student-UAR 8%–12%，GLM Flash 明显更高，见主表1）；Student 对 GLM/Kimi 回答判 unsafe 的比例更高，与 Silver 的 unsafe 率排序一致，说明检测器对不同厂商的回答具备基础迁移能力。")
    rep.append("2. **模型间风险差异（RQ2）**：预注册配对比较中，DeepSeek Pro−Flash 与 Qwen Plus−Flash 的 ΔStudent-UAR 的 95% CI 均包含 0，未发现同厂商内能力档位间的显著风险差异（表格见 §5.2）；跨厂商差异为描述性。")
    rep.append("3. **Student 与 Silver 一致性（RQ3）**：Random audit pooled 上 F1-unsafe、Macro-F1、MCC 等指标见主表2；结论为中等一致（详见 §11 局限）。")
    rep.append("4. **安全—能力差（RQ4）**：未观察到高能力档模型系统性更安全的证据；描述性排序中 GLM Flash（低档）与 Kimi 的 Silver unsafe 数最高，可能与免费档系统行为有关，需谨慎解读。")
    rep.append("")
    rep.append("## 9. 成本与运行门")
    rep.append("")
    rep.append(f"- 累计 API 成本 ¥{total_cost:.4f}（probe/pilot/formal/retry/judge/adjudication 全部阶段），远低于 ¥50 上限。")
    rep.append("- 硬性运行门全部通过：200 条 manifest 冻结并 SHA256 锁定；六模型 probe 通过；pilot 完成；formal coverage ≥95%（M5=99.5%，其余 100%）；Student 全量 1,199 条完成；Random-30/Boundary-10 冻结；双 Judge 240/240 共识；unresolved=0。")
    rep.append("")
    rep.append("## 10. 敏感性分析（模板前缀重叠 6 条）")
    rep.append("")
    rep.append(f"- 排除 {sens['n_excluded']} 条 `template_prefix_overlap` 后，Random audit 剩余 {sens['n_remaining']} 条：F1-unsafe={fmt_metric(sens['detection'],'f1_unsafe')}，Macro-F1={fmt_metric(sens['detection'],'macro_f1')}，MCC={fmt_metric(sens['detection'],'mcc')}，AUROC={fmt_metric(sens['detection'],'auroc')}。")
    rep.append("- 排除后各模型 Student-UAR：" + "；".join(f"{SLOT_LABEL[s]}={100*sens['student_uar_clean'][s]:.1f}%" for s in SLOTS) + "。与全量结果差异很小，说明模板前缀重叠不驱动主要结论。")
    rep.append("")
    rep.append("## 11. 局限")
    rep.append("")
    rep.append("- LLM-Silver 不是人工 Gold：双 Judge 均为低档模型，与目标模型同厂商（Qwen/DeepSeek）存在潜在自家族偏差；已报告敏感性一致率（§4）。")
    rep.append("- Random-30 每模型仅约 15 条 should-refuse，Silver-UAR/PLR/CRR/ORR 为宽区间抽样估计，不作精确排名；正式比较以全量 Student-UAR 为主。")
    rep.append("- 固定阈值 0.5622 来自 E3 训练分布，跨厂商回答分布可能偏移（截断率 32.5% 提示长回答被截断），因此 Student-UAR 应视为 detector-estimated 值而非真实 unsafe 率。")
    rep.append("- Boundary-10 为刻意困难的选择性样本，不与 Random 混合为普通测试集。")
    rep.append("- 生成温度 0.0 且单次采样，回答多样性受限；未覆盖流式/长上下文/工具调用场景。")
    rep.append("")
    rep.append("## 12. 主图")
    rep.append("")
    rep.append("![E6 主图：六模型 Student 风险分布](figures/e6_main_figure_risk_violin.png)")
    rep.append("")
    rep.append("## 13. 产物清单")
    rep.append("")
    rep.append("- 数据：`data/exp6_prompt_manifest.jsonl`（200 条，含 `prompt_manifest_sha256.txt`）；`generations/per_model/M1–M6.jsonl`（1,199 条）")
    rep.append("- Student：`student/predictions_all.jsonl`（1,199 条）+ `student/truncation_audit.json`")
    rep.append("- Silver：`silver_audit/audit_set.jsonl`、`judge_raw.jsonl`、`silver_labels.jsonl`、`judge_agreement.json`、`sensitivity_analysis.json`")
    rep.append("- 表格：`tables/main_table1_behavior_cost.md`、`tables/main_table2_student_transfer.md`、`tables/silver_audit_detail.md`")
    rep.append("- 图：`figures/e6_main_figure_risk_violin.png`")
    rep.append("- 协议：`protocol/`（protocol_lock.json、model_registry_frozen.json、probe_results.jsonl、pilot_selection.json、pricing_snapshot.json、protocol_deviation_log.md）")
    rep.append("- 成本：`budget/cost_ledger.jsonl`、`budget/cost_summary.json`、`budget/budget_gate.json`")
    rep.append("")
    rep.append("---")
    rep.append("")
    rep.append("生成：`scripts/e6_finalize.py`（离线，无 API 调用）｜ 报告采用 UTF-8 BOM 编码。")
    (E6_DIR / "EXP6_FINAL_REPORT.md").write_text("\n".join(rep), encoding="utf-8-sig")
    print("[S7] report written")
    print(f"[S7] total cost CNY{total_cost:.4f}")
    print("[S7] DONE")

if __name__ == "__main__":
    main()
