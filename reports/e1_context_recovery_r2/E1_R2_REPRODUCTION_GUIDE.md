# E1 R2 复现指南

```powershell
python scripts/run_e1_context_recovery_r2.py --phase p0-audit --dry-run
python scripts/run_e1_context_recovery_r2.py --phase p1-census
python scripts/run_e1_context_recovery_r2.py --phase p2-freeze-pilot-manifest
python scripts/run_e1_context_recovery_r2.py --phase p3-gold-pilot --confirm-budget
python scripts/run_e1_context_recovery_r2.py --phase p4-pilot-gate
python scripts/run_e1_context_recovery_r2.py --phase p8-report
```
