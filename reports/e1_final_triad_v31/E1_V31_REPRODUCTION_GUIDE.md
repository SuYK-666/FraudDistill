# E1 v3.1 复现指南

```powershell
python scripts/run_e1_a7500.py --phase p0
python scripts/run_e1_a7500.py --phase build-manifest
python scripts/run_e1_a7500.py --phase health --run-api --confirm-budget --limit-q 50
python scripts/run_e1_a7500.py --phase generate --run-api --confirm-budget --batch-size-q 500 --resume
python scripts/run_e1_a7500.py --phase validate-targets
python scripts/run_e1_a7500.py --phase report
```
