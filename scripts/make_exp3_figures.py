# -*- coding: utf-8 -*-
"""Generate the 6 report figures for Exp3 (guide 27.6)."""
from __future__ import annotations

import json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path(os.environ.get('EXP3_OUT_ROOT') or r'experiments\exp3_agent_distillation_ablation\outputs')
METRICS = OUT / 'metrics'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def L(*codes):
    return ''.join(chr(int(c, 16)) for c in codes)


T_TITLE = L('5B66','751F','6D2F','84B8','68BF','5EA6')      # 学生蒸馏梯度
T_NEST = L('5D4C','5957','6D88','878D')                     # 嵌套消融
T_COMP = L('7EC4','4EF6','5F3A','529B')                     # 组件压力
T_REL = L('53EF','9760','5EA6','56FE')                     # 可靠性图
T_PARETO = L('6210','672C','6027','80FD','74F0','62DC','625F','56FE')   # 成本性能帕累托图
T_CONF = L('51B2','7A81','4E0E','4FEE','6B63','6D41','7A0B')           # 冲突与修正流程
T_MACROF1 = 'Macro-F1'
T_RECALL = 'Recall'
T_FPR = 'FPR'
T_AUPRC = 'AUPRC'
T_VALUE = 'Value'
T_REMOVED = 'Removed'
T_DELTA = 'Delta'


def load_csv(name):
    import csv
    with open(METRICS / name, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def setting_short(name):
    for k in ('_full_correction', '_evidence_arbiter', '_rule_arbiter',
              '_fraud_refusal_context', '_fraud_refusal', '_fraud_only',
              '_single_judge', '_rule'):
        name = name.replace(k, '')
    return name


# ---------- Figure 1: nested ablation ----------
nested = load_csv('nested_ablation.csv')
names = [setting_short(r['setting']) for r in nested]
f1 = [float(r['macro_f1']) for r in nested]
rec = [float(r['recall']) for r in nested]
fpr = [float(r['fpr']) for r in nested]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 5.2))
w = 0.26
ax.bar(x - w, f1, w, label=T_MACROF1, color='#4C72B0')
ax.bar(x, rec, w, label=T_RECALL, color='#55A868')
ax.bar(x + w, fpr, w, label=T_FPR, color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=28, ha='right')
ax.set_ylabel(T_VALUE); ax.set_title(T_NEST)
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig1_nested_ablation.png', dpi=150)
plt.close(fig)

# ---------- Figure 2: student gradient ----------
students = load_csv('student_gradient.csv')
snames = [r['setting'].replace('_gold', '').replace('_hard_teacher', '').replace('_score_distill', '')
          .replace('_type_distill', '').replace('_evidence_distill', '') for r in students]
sf1 = [float(r['macro_f1_mean']) for r in students]
sauprc = [float(r['auprc_mean']) for r in students]
srec = [float(r['recall_mean']) for r in students]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(snames, sf1, '-o', label=T_MACROF1, color='#4C72B0')
ax.plot(snames, sauprc, '-s', label=T_AUPRC, color='#55A868')
ax.plot(snames, srec, '--^', label=T_RECALL, color='#DD8452')
for i, v in enumerate(sf1):
    ax.annotate(f'{v:.4f}', (i, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.set_ylabel(T_VALUE); ax.set_title(T_TITLE)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig2_student_gradient.png', dpi=150)
plt.close(fig)

# ---------- Figure 3: component stress heatmap ----------
comp = load_csv('component_metrics.csv')
comp_names = [setting_short(r['setting']) for r in comp]
keys = ['direct_recall', 'trust_recall', 'leakage_recall', 'clean_refusal_fpr', 'hard_safe_fpr',
        'over_refusal_recall', 'quotation_fpr', 'education_fpr', 'toxic_recall']
labels = ['Direct R', 'Trust R', 'Leakage R', 'CleanRef FPR', 'HardSafe FPR',
          'OverRef R', 'Quotation FPR', 'Education FPR', 'Toxic R']
M = np.array([[float(r.get(k, 0) or 0) for k in keys] for r in comp])
fig, ax = plt.subplots(figsize=(11, 5.5))
im = ax.imshow(M.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(comp_names))); ax.set_xticklabels(comp_names, rotation=30, ha='right')
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(M.shape[1]):
    for j in range(M.shape[0]):
        ax.text(j, i, f'{M[j, i]:.2f}', ha='center', va='center', fontsize=7, color='black')
ax.set_title(T_COMP)
fig.colorbar(im, shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / 'fig3_component_stress.png', dpi=150)
plt.close(fig)

# ---------- Figure 4: reliability diagram ----------
dev = [json.loads(l) for l in (OUT / 'agent_predictions' / 'dev.jsonl').open(encoding='utf-8') if l.strip()]
scores = np.array([float(r['signal'].get('teacher_score', 0.5)) for r in dev])
y = np.array([1 if r['sample']['gold_label'] == 'unsafe' else 0 for r in dev])
bins = np.linspace(0, 1, 11)
centers = []; freqs = []; counts = []
for i in range(10):
    m = (scores > bins[i]) & (scores <= bins[i + 1])
    if m.sum() == 0:
        continue
    centers.append(float(scores[m].mean()))
    freqs.append(float(y[m].mean()))
    counts.append(int(m.sum()))
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
ax.plot(centers, freqs, '-o', color='#4C72B0')
for c, f, n in zip(centers, freqs, counts):
    ax.annotate(str(n), (c, f), textcoords='offset points', xytext=(4, 4), fontsize=8)
ax.set_xlabel('Predicted risk score'); ax.set_ylabel('Observed unsafe rate')
ax.set_title(T_REL); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig4_reliability.png', dpi=150)
plt.close(fig)

# ---------- Figure 5: cost-performance Pareto ----------
# Real cost per 1k rows computed from recorded token usage (pricing: hit 0.02,
# miss 1.0, out 2.0 RMB / 1M tokens). T0=0; T1=judge only (dev+test, 2309 calls);
# T2-T6=4-call specialist+arbiter pipeline (measured on train, no correction);
# T7=test full pipeline incl. conflict correction (193/1262 rows).
def _token_cost(path, keys, extra_keys=()):
    hit = miss = out = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            for k in keys:
                a = r.get(k) or {}
                if isinstance(a, dict):
                    u = a.get('usage') or {}
                    hit += u.get('input_hit', 0); miss += u.get('input_miss', 0); out += u.get('output', 0)
            s = r.get('signal') or {}
            u = s.get('usage') or {}
            hit += u.get('input_hit', 0); miss += u.get('input_miss', 0); out += u.get('output', 0)
            u = r.get('usage') or {}
            hit += u.get('input_hit', 0); miss += u.get('input_miss', 0); out += u.get('output', 0)
            c = r.get('correction') or {}
            for adv in extra_keys:
                a = c.get(adv) or {}
                u = a.get('usage') or {}
                hit += u.get('input_hit', 0); miss += u.get('input_miss', 0); out += u.get('output', 0)
    return (hit * 0.02 + miss * 1.0 + out * 2.0) / 1e6

def _rows(path):
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f)

PIPELINE_KEYS = ('fraud', 'refusal', 'context', 'arbiter_pre_correction')
train_cost = _token_cost(OUT / 'agent_predictions' / 'train.jsonl', PIPELINE_KEYS)
train_rows = _rows(OUT / 'agent_predictions' / 'train.jsonl')
test_cost = _token_cost(OUT / 'agent_predictions' / 'test.jsonl', PIPELINE_KEYS, ('unsafe_advocate', 'safe_advocate'))
test_rows = _rows(OUT / 'agent_predictions' / 'test.jsonl')
judge_cost = 0.0; judge_rows = 0
for sp in ('dev', 'test'):
    p = OUT / 'judge_predictions' / (sp + '.jsonl')
    judge_cost += _token_cost(p, ())
    judge_rows += _rows(p)
pipeline_per1k = train_cost / train_rows * 1000.0
judge_per1k = judge_cost / judge_rows * 1000.0
full_per1k = test_cost / test_rows * 1000.0
costs = {
    'T0_rule': 0.0, 'T1_single_judge': judge_per1k,
    'T2_fraud_only': pipeline_per1k, 'T3_fraud_refusal': pipeline_per1k,
    'T4_fraud_refusal_context': pipeline_per1k, 'T5_rule_arbiter': pipeline_per1k,
    'T6_evidence_arbiter': pipeline_per1k, 'T7_full_correction': full_per1k,
}
print('cost per 1k rows: pipeline=%.3f judge=%.3f full=%.3f' % (pipeline_per1k, judge_per1k, full_per1k))
fig, ax = plt.subplots(figsize=(9, 5.5))
for r in nested:
    nm = r['setting']
    c = costs.get(nm, 0.28)
    ax.scatter(c, float(r['macro_f1']), s=90)
    ax.annotate(setting_short(nm), (c, float(r['macro_f1'])), textcoords='offset points', xytext=(6, 4), fontsize=8)
ax.set_xlabel('Cost RMB per 1k rows (estimated)'); ax.set_ylabel(T_MACROF1)
ax.set_title(T_PARETO); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig5_cost_pareto.png', dpi=150)
plt.close(fig)

# ---------- Figure 6: conflict & correction flow ----------
test = [json.loads(l) for l in (OUT / 'agent_predictions' / 'test.jsonl').open(encoding='utf-8') if l.strip()]
cf = Counter(); corr = 0
for r in test:
    for f in (r['signal'].get('conflict_flags') or []):
        cf[f] += 1
    if r['signal'].get('correction_used'):
        corr += 1
fig, ax = plt.subplots(figsize=(9, 5))
labels_c = [k for k, _ in cf.most_common(6)]
vals = [v for _, v in cf.most_common(6)]
bars = ax.barh(labels_c[::-1], vals[::-1], color='#4C72B0')
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 2, b.get_y() + b.get_height() / 2, str(v), va='center', fontsize=9)
ax.set_xlabel('Count (test, n=1262)'); ax.set_title(T_CONF)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / 'fig6_conflict_correction.png', dpi=150)
plt.close(fig)

print('figures written to', FIG)
for f in sorted(FIG.glob('*.png')):
    print(' ', f.name)