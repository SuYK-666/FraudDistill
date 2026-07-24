# Reproduce Six Experiments

本仓库的统一重跑入口：

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/run_high_standard_rerun.py all --bootstrap 500
python scripts/write_six_experiment_master_report.py
pytest -q
```

运行顺序固定为 smoke、pilot、full；smoke/pilot 会自动归档，全量正式产物保留在 `outputs/*/high_standard_full/`。全量重跑前脚本会把已有 `high_standard_full` 归档到 `archive/pre_high_standard_full_rerun_*`。

注意：`data/`、`outputs/`、`archive/`、模型文件和 `api_keys.py` 不提交到 GitHub。报告副本位于 `docs/results/`，原始实验产物仍以本地 `outputs/` 为准。
