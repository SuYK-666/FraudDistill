# -*- coding: utf-8 -*-
"""Background runner: neural-student training with self-contained logging."""
import subprocess, sys, time
cmd = [sys.executable, "scripts/train_exp3_students.py", "--backend", "neural", "--setting", "gold",
       "--architecture", "standard", "--seeds", "11", "--epochs", "2", "--max-length", "384",
       "--micro-batch", "2", "--effective-batch", "32", "--eval-steps", "200", "--patience", "2",
       "--eval-subset", "200", "--lora-r", "32"]
log = open("outputs/train_neural_gold_seed11.log", "a", encoding="utf-8")
err = open("outputs/train_neural_gold_seed11.err", "a", encoding="utf-8")
p = subprocess.Popen(cmd, stdout=log, stderr=err)
print("launched", p.pid)
