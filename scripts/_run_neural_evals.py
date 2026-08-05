import subprocess, sys, time, os
from pathlib import Path

base = Path(r"C:\Users\18201\Desktop\FraudDistill\experiments\exp3_agent_distillation_ablation\outputs\neural_student")
jobs = [
    (base / "gold_standard_seed11_final", base / "eval_gold"),
    (base / "soft_distill_standard_seed11_final", base / "eval_soft"),
    (base / "full_distill_standard_seed11_final", base / "eval_full"),
    (base / "lowlabel" / "gold_standard_seed11_gf0.1_final", base / "eval_lowlabel_gold10"),
    (base / "zero_shot_standard", base / "eval_zero_shot"),
]
env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
log = Path(r"C:\Users\18201\Desktop\FraudDistill\outputs\neural_eval_driver.log")
with log.open("a", encoding="utf-8") as f:
    for ckpt, out in jobs:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {ckpt.name} -> {out.name}\n")
        f.flush()
        rc = subprocess.call([sys.executable, "scripts/evaluate_neural_student.py",
                              "--checkpoint", str(ckpt), "--architecture", "standard",
                              "--max-length", "384", "--micro-batch", "8", "--out-dir", str(out)],
                             stdout=f, stderr=f, env=env)
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE {ckpt.name} rc={rc}\n")
        f.flush()
