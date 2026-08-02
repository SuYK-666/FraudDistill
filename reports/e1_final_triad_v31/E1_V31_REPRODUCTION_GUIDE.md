# E1 v3.1 复现指南

```powershell
python scripts/run_e1_a7500.py --phase p0
python scripts/run_e1_a7500.py --phase build-manifest
python scripts/run_e1_a7500.py --phase health --run-api --confirm-budget --limit-q 50
python scripts/run_e1_a7500.py --phase generate --run-api --confirm-budget --batch-size-q 500 --resume
python scripts/run_e1_a7500.py --phase validate-targets
python scripts/run_e1_a7500.py --phase gold --run-api --confirm-budget --resume
python scripts/run_e1_a7500.py --phase adjudicate --run-api --confirm-budget --resume
python scripts/run_e1_a7500.py --phase freeze
python scripts/run_e1_b3200.py --phase build-panel --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-gold --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-adjudicate --run-api --confirm-budget --resume
python scripts/run_e1_b3200.py --phase b-consensus
python scripts/run_e1_b3200.py --phase validate-panel
python scripts/run_e1_b3200.py --phase model-dev
python scripts/run_e1_b3200.py --phase calibration
python scripts/run_e1_b3200.py --phase freeze-b
python scripts/run_e1_b3200.py --phase anchor --consume-anchor
python scripts/run_e1_c_real_prevalence.py --phase c-all
python scripts/run_e1_final_triad_v3.py --phase final-report
```
