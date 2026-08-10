# -*- coding: utf-8 -*-
"""v4 detectors: M0 TF-IDF/LR diagnostic + M1 XLM-R joint encoder (CPU)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frauddistill.e1_final_v3.detector_v31 import ViewDetector as LRViewDetector  # M0 (reuse)
from frauddistill.e1_final_v3.detector_v31 import choose_threshold, metrics_at_threshold, panel_rows_to_eval  # noqa: F401


def _text(mode: str, q: str, y: str, wrong_q: str | None = None) -> str:
    if mode == "q_only":
        return f"[Q] {q}"
    if mode == "y_only":
        return f"[Y] {y}"
    if mode == "wrong_q_y":
        return f"[Q] {wrong_q or ''} [SEP] [Y] {y}"
    return f"[Q] {q} [SEP] [Y] {y}"


class NeuralJointDetector:
    def __init__(self, mode: str, model_name: str = "xlm-roberta-base", max_length: int = 256, seed: int = 13,
                 q_cap: int | None = None, y_cap: int | None = None):
        if mode not in ("q_only", "y_only", "q_y", "wrong_q_y"):
            raise ValueError(f"unknown neural mode: {mode}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.mode = mode
        self.model_name = model_name
        self.max_length = int(max_length)
        self.joint = mode in ("q_y", "wrong_q_y")
        self.q_cap = int(q_cap) if q_cap and self.joint else int(max_length)
        self.y_cap = int(y_cap) if y_cap and self.joint else int(max_length)
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.config.problem_type = "single_label_classification"
        self.threshold: float = 0.5
        self.fitted = False

    def _encode(self, rows: list[dict[str, Any]], wrong_q_map: dict[str, str] | None = None):
        if self.joint:
            all_ids: list[list[int]] = []
            for r in rows:
                wq = wrong_q_map.get(r.get("response_id", "")) if wrong_q_map else None
                q = wq or str(r.get("q_private") or "")
                y = str(r.get("y_private") or "")
                qi = self.tokenizer(q, add_special_tokens=False, max_length=self.q_cap, truncation=True).input_ids
                yi = self.tokenizer(y, add_special_tokens=False, max_length=self.y_cap, truncation=True).input_ids
                ids = [self.tokenizer.bos_token_id] + qi + [self.tokenizer.sep_token_id] + yi + [self.tokenizer.sep_token_id]
                all_ids.append(ids[: self.max_length])
            max_len = max(len(x) for x in all_ids) if all_ids else 0
            input_ids = torch.full((len(all_ids), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for i, ids in enumerate(all_ids):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[i, : len(ids)] = 1
            return {"input_ids": input_ids, "attention_mask": attention}
        texts = []
        for r in rows:
            wq = wrong_q_map.get(r.get("response_id", "")) if wrong_q_map else None
            texts.append(_text(self.mode, str(r.get("q_private") or ""), str(r.get("y_private") or ""), wq))
        return self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")

    def fit(self, rows: list[dict[str, Any]], labels: list[int], *, epochs: int = 2, batch_size: int = 8,
            grad_accum: int = 4, lr: float = 2e-5, lr_head: float = 5e-4, warmup_steps: int = 40,
            max_grad_norm: float = 1.0, seed: int | None = None, log_every: int = 50, on_step=None) -> dict[str, Any]:
        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed)
        torch.set_num_threads(max(4, torch.get_num_threads()))
        enc = self._encode(rows)
        y = torch.tensor([int(l) for l in labels], dtype=torch.long)
        n = len(rows)
        head_params = [p for name, p in self.model.named_parameters() if "classifier" in name]
        rest_params = [p for name, p in self.model.named_parameters() if "classifier" not in name]
        opt = torch.optim.AdamW([
            {"params": rest_params, "lr": lr},
            {"params": head_params, "lr": lr_head},
        ], weight_decay=0.01)
        steps_per_epoch = (n + batch_size * grad_accum - 1) // (batch_size * grad_accum)
        total_steps = steps_per_epoch * epochs
        def lr_lambda(s):
            if s < warmup_steps:
                return (s + 1) / warmup_steps
            return 1.0
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        self.model.train()
        history = []
        t0 = time.time()
        step = 0
        opt.zero_grad()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            loss_sum = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                out = self.model(input_ids=enc["input_ids"][idx], attention_mask=enc["attention_mask"][idx], labels=y[idx])
                (out.loss / grad_accum).backward()
                loss_sum += float(out.loss.detach())
                n_batches += 1
                if (i // batch_size + 1) % grad_accum == 0 or (i + batch_size) >= n:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    step += 1
                    if step % log_every == 0:
                        info = {"epoch": epoch, "step": step, "loss": round(loss_sum / max(1, n_batches), 4), "elapsed_s": round(time.time() - t0, 1)}
                        history.append(info)
                        print(f"[M1-{self.mode}] epoch {epoch} step {step}/{total_steps} loss {info['loss']} ({info['elapsed_s']}s)", flush=True)
                        if on_step:
                            on_step(info)
            print(f"[M1-{self.mode}] epoch {epoch} done, mean loss {round(loss_sum / max(1, n_batches), 4)}", flush=True)
        self.fitted = True
        return {"history": history, "total_steps": step, "elapsed_s": round(time.time() - t0, 1)}

    @torch.no_grad()
    def predict_proba(self, rows: list[dict[str, Any]], wrong_q_map: dict[str, str] | None = None, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        enc = self._encode(rows, wrong_q_map)
        scores = []
        for i in range(0, len(rows), batch_size):
            out = self.model(input_ids=enc["input_ids"][i:i + batch_size], attention_mask=enc["attention_mask"][i:i + batch_size])
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            scores.append(probs.cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        (path / "meta.json").write_text(json.dumps({"mode": self.mode, "threshold": self.threshold, "seed": self.seed, "model_name": self.model_name}), encoding="utf-8")


def run_neural_seed(dev, cal, anchor, mode: str, seed: int, cfg, wrong_q_map=None, out_dir=None) -> dict[str, Any]:
    neural = cfg["e1_v4"]["neural"]
    det = NeuralJointDetector(mode, model_name=neural["model_name"], max_length=int(neural["max_length"]), seed=seed,
                              q_cap=int(neural.get("q_cap", 0)) or None, y_cap=int(neural.get("y_cap", 0)) or None)
    d_rows, d_labels = panel_rows_to_eval(dev)
    c_rows, _ = panel_rows_to_eval(cal)
    a_rows, _ = panel_rows_to_eval(anchor)
    fit_info = det.fit(d_rows, d_labels, epochs=int(neural["epochs"]), batch_size=int(neural["batch_size"]),
                       grad_accum=int(neural["grad_accum"]), lr=float(neural["lr_lora"]), lr_head=float(neural["lr_head"]),
                       warmup_steps=int(neural["warmup_steps"]), seed=seed)
    cal_scores = det.predict_proba(c_rows)
    det.threshold = choose_threshold(c_rows, cal_scores)
    anchor_scores = det.predict_proba(a_rows, wrong_q_map)
    m = metrics_at_threshold(a_rows, anchor_scores, det.threshold)
    result = {"mode": mode, "seed": seed, "threshold": det.threshold, "anchor": m, "fit": fit_info}
    if out_dir:
        det.model.half()  # fp16 storage keeps disk usage ~560MB/model (CPU inference only)
        det.save(out_dir / "models" / f"{mode}_seed{seed}")
    return result
