# Exp2 Preregistration ? Cross-Benchmark Comparison (Task-Aligned Track)

> Frozen 2026-08-05, before any API call for this run. All decisions below are
> locked; results are reported without post-hoc selection.

## 1. Research question
On publicly available safety benchmarks (Fraud-R1, OR-Bench, Do-Not-Answer,
Aegis 2.0), when the evaluation target is unified as *"does the model ANSWER
produce fraud assistance, dangerous refusal failure, or unnecessary over-
refusal for the given request"*, does the FraudDistill evidence-based
multi-agent teacher (T6 Evidence MAT) outperform each benchmark's original
evaluator on the same `q+y` samples with the same blind gold?

## 2. Method (frozen)
- Teacher: Exp3 T6 Evidence MAT (Fraud + Refusal + Context specialists,
  Evidence Arbiter; conflict correction OFF). Model deepseek-v4-flash,
  thinking disabled, JSON mode, temperature 0, output caps 160/160/140/160
  (guide 13.2). Agents never see benchmark name / gold / baseline / target model.
- Baselines (reused, no re-run): Fraud-R1 Official Judge (GPTCheck protocol,
  deepseek-v4-flash), OR-Bench Official Response Checker (deepseek-v4-flash),
  Do-Not-Answer Longformer-Harmful (prob_harmful), Aegis NemoGuard-8B
  (full 1,964 local predictions).
- Gold: two independent BLINDED LLM annotators (deepseek-v4-flash A,
  deepseek-v4-pro B) + third adjudicator on disagreements, per guide 12.
  Annotators see only query + answer. Gold fields per guide 12.3.
  Pilot: 25 per benchmark; stop and re-calibrate if binary kappa < 0.75
  or 4-class kappa < 0.60.
- Primary label: frozen categorical arbiter decision (no per-benchmark
  threshold tuning; no separate calibration split needed, guide 15.1).
- Primary metric: True Macro-F1 (average="macro"). Also report Accuracy,
  Precision, Unsafe Recall, Unsafe-F1, Safe-F1, FPR, Balanced Accuracy, MCC,
  AUPRC (teacher score; label-only baselines = "-"), 4-class Macro-F1.

## 3. Data (frozen manifests, group-disjoint)
| Source | Test N | Sampling | Exp3 overlap |
|---|---:|---|---|
| Fraud-R1 | 800 | 5 categories x 160 (80 zh + 80 en; per lang 40 assistant + 40 roleplay; per scenario 20 base + 20 levelup); one row per case group | 0 |
| OR-Bench | 800 | hard_safe 350 / regular_safe 200 / toxic 250; category round-robin; one row per prompt | 0 |
| Do-Not-Answer | 900 (150 prompt groups x 6 models) | response-diverse groups first (136 exist), then clean top-up; group-disjoint | 123/900 (13.7%, new blind gold, marked frozen benchmark reuse) |
| Aegis 2.0 | 813 | ALL valid response-level samples (guide 11.3 fallback); official strata 394 unsafe / 419 safe | 793/813 (guide 7.3 frozen benchmark reuse, new blind gold) |

Overlap = exact qy_hash match with Exp3 train/dev/test (incl. pilot) or
neural-student train manifests (guide 7.2). No baseline prediction is used
in sampling; only official categories, languages, prompt types, response
existence, official gold strata and group ids.

## 4. Statistics (guide 19)
- Clustered paired bootstrap, 10,000 reps, sampling unit = group
  (fraudr1 case, orbench prompt, dna prompt group, aegis interaction).
- Exact McNemar on discordant pairs per benchmark.
- Holm correction over the four primary comparisons only.
- Mechanism sub-analyses (direct/trust/leakage/clean-refusal/hard-safe/
  quotation/education, within-prompt pair accuracy) are secondary.

## 5. Success criteria (guide 20.5)
1. Macro-F1 higher than the original method on >= 3/4 benchmarks;
2. >= 2/4 delta 95% CI entirely above 0;
3. no benchmark shows > 2pp significant regression;
4. Fraud-R1 and OR-Bench must clearly win;
5. DNA/Aegis win on at least one task-specific mechanism metric;
6. new API spend <= 40 RMB (hard stop 36 + 4 reserve);
7. teacher coverage >= 99.5%, parse failure <= 0.5%.

## 6. Deviations (recorded a priori)
- No separate 100-per-benchmark calibration split: primary label is the
  frozen categorical decision (guide 15.1), so no test-time threshold fitting.
- Blind gold is produced by two independent LLM annotators (no human
  annotators available in this environment); human labels can be swapped in
  later by sample_id without changing the pipeline.
- Fraud-R1 main test uses 50% roleplay-scenario rows so that the gold
  contains a meaningful share of unsafe answer behaviors (assistant-only
  gold was ~0% unsafe in the legacy audit); the guide 8.4 explicitly keeps
  role-play variants in the same split, and each group contributes one row.
- DNA response-diverse groups are fewer than 150 (136); all 136 diverse
  groups are included, topped up with clean non-diverse groups.
- Aegis valid response-level pool is 813; all 813 are used (guide 11.3
  fallback), 793 marked frozen benchmark reuse with new blind gold.

## 7. Exclusions
- Native-task full-pool teacher runs are NOT part of this budget; native
  appendix reuses existing official labels + baseline predictions and the
  task-aligned T6 predictions on the same samples.
- No prompt-only Aegis samples (response must be present).
- No test-set threshold tuning of any kind.
