# E6 Protocol Deviation Log

## v1.0 (manifest freeze)

- **anti_fraud stratum (30)**: the pre-existing anti-fraud-education pool
  (`data/unified/v2_hard_control_full.jsonl`) overlaps E3 Student train 100%
  (152/152 zh, 148/148 en exact query matches). Quota could not be filled from
  existing pools without leaking training data. 30 questions (15 zh / 15 en)
  were authored offline for E6 in the style of the original pool; no API used;
  all verified exact-query disjoint from Student train (leakage_audit.json).
- **hard_safe zh (20)**: OR-Bench has no official Chinese subset; the 20 zh
  hard-safe questions are offline manual translations of 20 OR-Bench hard-safe
  questions (same prompt_family per en/zh pair, recorded in source_id).
- **matched_safe (20)**: selected from E4 U1 safe rows after manual curation;
  rows whose text also appeared in the E4 U1 unsafe pool (label conflict) were
  excluded and replaced.
- **Normalization**: queries with leading/trailing `???` template padding were
  normalized (padding stripped) before freezing; original text is recoverable
  from source_id + source pool.
- **Template-prefix overlaps (6)**: e6_0009/0027/0036/0040 (direct_unsafe)
  embed fraud-artifact text that also appears inside Student-train roleplay
  wrappers; e6_0051/0056 (roleplay_unsafe) share wrapper templates with train.
  No exact-query overlap. Flagged `template_prefix_overlap: true`; a sensitivity
  analysis (stats without these rows) is planned.

## v1.1 (S4-S6 execution)

- **S4 infrastructure**: Student scoring executed on the research server (RTX 4090,
  batch=16) instead of the local CPU; the same frozen checkpoint and thresholds were
  used. Local partial run (168 rows) was discarded and superseded by the full GPU
  run; `student/score_progress.jsonl` (leftover) moved to `archive/`.
- **S6 judge normalization**: GLM Flash (Judge C) occasionally emitted
  `binary_label`/`behavior` fields swapped or used a behavior value in
  `binary_label`. A deterministic schema normalization (documented in
  `scripts/e6_judge.py::validate`) maps these to the protocol schema before
  validation; the normalization is order-invariant and does not change semantics.
- **S6 repair policy**: Judge outputs failing JSON validation get exactly one repair
  prompt (per protocol); for GLM Flash the repair additionally asks for a shorter
  `brief_reason` to avoid max_tokens truncation of the JSON object. Repairs are
  bounded; any remaining invalid C output is marked `silver_unresolved` per protocol.
- **S6 result**: after repairs, 240/240 audit rows reached consensus; unresolved = 0.
