# Reproduce Six Experiments

```powershell
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_high_standard_rerun.py protocol_gate_v2 --bootstrap 500 --api-provider qwen --api-probe-limit 12
python scripts/write_six_experiment_master_report.py --run-id protocol_gate_v2
```

`data/`, `outputs/`, `archive/`, model files, and `api_keys.py` are local-only and are not committed to GitHub.
