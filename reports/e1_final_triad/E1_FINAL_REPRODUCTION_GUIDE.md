# E1 FINAL 复现指南

```powershell
python scripts/run_e1_final_triad.py --phase p0
python scripts/run_e1_final_triad.py --phase registry --cache-only
python scripts/run_e1_final_triad.py --phase build-candidates --cache-only
python scripts/run_e1_final_triad.py --phase budget-plan --dry-run
python scripts/run_e1_final_triad.py --phase gold-and-freeze --confirm-budget
python scripts/run_e1_final_triad.py --phase train-calibrate
python scripts/run_e1_final_triad.py --phase anchor --consume-anchor
python scripts/run_e1_final_triad.py --phase c-transfer
python scripts/run_e1_final_triad.py --phase report
```
