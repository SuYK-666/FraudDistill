# E1 TRIAD 复现指南

```powershell
python scripts/run_e1_triad_final.py --phase p0-audit --dry-run
python scripts/run_e1_triad_final.py --phase p1-census --cache-only
python scripts/run_e1_triad_final.py --phase p3-freeze-panels --confirm-budget
python scripts/run_e1_triad_final.py --phase p4-modeldev-calibration --confirm-budget
python scripts/run_e1_triad_final.py --phase p5-anchor-c --confirm-budget --consume-anchor
python scripts/run_e1_triad_final.py --phase p6-report --cache-only
```
