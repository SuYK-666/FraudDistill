# E1 Final Push 复现指南

```powershell
python scripts/run_e1_final_push.py --phase p0-code-audit --dry-run
python scripts/run_e1_final_push.py --phase p0-reuse-audit --cache-only
python scripts/run_e1_final_push.py --phase p1-build-q-pool --cache-only
python scripts/run_e1_final_push.py --phase p1-pilot-decision --cache-only
python scripts/run_e1_final_push.py --phase p6-report --cache-only
```
