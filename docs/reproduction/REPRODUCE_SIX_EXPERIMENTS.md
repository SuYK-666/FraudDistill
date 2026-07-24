# Reproduce Six Experiments

本仓库的统一重跑入口：

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/run_high_standard_rerun.py small --bootstrap 300 --small-limit 720 --api-provider qwen --api-probe-limit 6
python scripts/write_six_experiment_master_report.py --run-id ccfa_small_qwen
python scripts/run_high_standard_rerun.py all --bootstrap 500
python scripts/write_six_experiment_master_report.py
pytest -q
```

本轮小规模审查产物保留在 `outputs/*/ccfa_small_qwen/`；E1-E5 使用 qwen API teacher probe 写入各实验 `raw_outputs/qwen_teacher_probe.jsonl`，实验指标仍保留 gold/silver/proxy 的标签边界。

运行顺序固定为 smoke、pilot、full；smoke/pilot 会自动归档，全量正式产物保留在 `outputs/*/high_standard_full/`。全量重跑前脚本会把已有 `high_standard_full` 归档到 `archive/pre_high_standard_full_rerun_*`。

注意：`data/`、`outputs/`、`archive/`、模型文件和 `api_keys.py` 不提交到 GitHub。报告副本位于 `docs/results/`，原始实验产物仍以本地 `outputs/` 为准。
