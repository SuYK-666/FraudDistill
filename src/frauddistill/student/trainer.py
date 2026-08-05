# -*- coding: utf-8 -*-
"""CPU-friendly neural-student trainer (guide 4.2, 14.3, 18).

- separate LR for LoRA params vs classification head (lr_lora / lr_head);
- gradient accumulation to reach effective batch size;
- dev evaluation every eval_steps with early stopping (patience);
- per-step loss components logged (loss_gold/loss_soft/loss_pair/loss_total);
- deterministic seeds; saves adapter + head checkpoint and dev best.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def split_param_groups(model, lr_lora: float, lr_head: float, weight_decay: float = 0.01):
    lora_params, head_params, rest_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name:
            lora_params.append(p)
        elif "score" in name or "head" in name:
            head_params.append(p)
        else:
            rest_params.append(p)
    groups = [{"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay}]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if rest_params:
        groups.append({"params": rest_params, "lr": lr_lora * 0.5, "weight_decay": weight_decay})
    return groups


def train_neural(model, train_loader, dev_loader, loss_fn, tokenizer,
                 epochs=3, lr_lora=1e-4, lr_head=5e-4, weight_decay=0.01,
                 warmup_steps=50, grad_accum=8, eval_steps=100, patience=3,
                 max_grad_norm=1.0, seed=11, out_dir=None, device=None,
                 architecture="standard", log_every=10, resume=None, history=None,
                 max_steps=None):
    """Train the neural student on CPU/CUDA with checkpoint/resume support.

    - every eval_steps: dev eval, best-state snapshot and a resume checkpoint
      (model + optimizer + scheduler + step counters) are saved under out_dir;
    - resume=<checkpoint.json> continues from the saved state (same seed/loader
      order required for exact continuation; shuffle seed is fixed by seed).
    Returns (best_state, history).
    """
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    groups = split_param_groups(model, lr_lora, lr_head, weight_decay)
    optimizer = torch.optim.AdamW(groups)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(warmup_steps, 1)))
    history = history if history is not None else {"train": [], "dev": []}
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    global_step = 0
    start_epoch = 0
    start_step = 0
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"])
        start_step = int(ck["step_in_epoch"])
        global_step = int(ck["global_step"])
        best_metric = float(ck.get("best_metric", -1.0))
        best_epoch = int(ck.get("best_epoch", 0))
        history = ck.get("history", history)
        print(f"resumed from {resume}: epoch={start_epoch} step={start_step} global={global_step}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss_total": 0.0, "loss_gold": 0.0, "loss_soft": 0.0, "loss_pair": 0.0}
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            loss, comps = loss_fn(out.logits, batch["gold_type_id"], batch["teacher_distribution"],
                                  batch["sample_weight"], batch.get("pair_metadata"))
            loss = loss / grad_accum
            loss.backward()
            for k in running:
                running[k] += float(comps[k]) / grad_accum
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    print(f"max_steps reached ({max_steps}); stopping", flush=True)
                    if best_state:
                        model.load_state_dict(best_state)
                    return best_state, history
                if global_step % log_every == 0:
                    print(f"  epoch {epoch+1} step {global_step} " +
                          " ".join(f"{k}={v / log_every:.4f}" for k, v in running.items()), flush=True)
                    running = {k: 0.0 for k in running}
                if out_dir:
                    # crash-safe periodic resume snapshot every log_every steps
                    save_resume(model, optimizer, scheduler, out_dir / "resume.pt", epoch, step, global_step,
                                best_metric, best_epoch, history)
            # eval on the micro-batch cadence, independent of grad accumulation
            if (step + 1) % eval_steps == 0:
                dev_metric = evaluate_neural(model, dev_loader, loss_fn, device, architecture)
                history["dev"].append({"step": global_step, "micro_step": step + 1, **dev_metric})
                print(f"  eval step {global_step}: macro_f1={dev_metric['macro_f1']:.4f} "
                      f"recall={dev_metric['recall']:.4f} fpr={dev_metric['fpr']:.4f}", flush=True)
                if dev_metric["macro_f1"] > best_metric:
                    best_metric = dev_metric["macro_f1"]
                    best_epoch = epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if out_dir:
                        save_checkpoint(model, tokenizer, out_dir / f"best_step{global_step}", architecture)
                else:
                    if epoch >= best_epoch + patience:
                        print(f"early stop at epoch {epoch+1} step {global_step}", flush=True)
                        if best_state:
                            model.load_state_dict(best_state)
                        return best_state, history
                if out_dir:
                    save_resume(model, optimizer, scheduler, out_dir / "resume.pt", epoch, step, global_step,
                                best_metric, best_epoch, history)
        # end of epoch eval
        dev_metric = evaluate_neural(model, dev_loader, loss_fn, device, architecture)
        history["dev"].append({"step": global_step, "epoch": epoch + 1, **dev_metric})
        print(f"epoch {epoch+1} dev: macro_f1={dev_metric['macro_f1']:.4f} recall={dev_metric['recall']:.4f} fpr={dev_metric['fpr']:.4f}", flush=True)
        if dev_metric["macro_f1"] > best_metric:
            best_metric = dev_metric["macro_f1"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if out_dir:
                save_checkpoint(model, tokenizer, out_dir / f"best_epoch{epoch+1}", architecture)
    if best_state:
        model.load_state_dict(best_state)
    return best_state, history


def evaluate_neural(model, loader, loss_fn, device, architecture="standard"):
    """Dev evaluation: pooled macro-F1 / recall / fpr at the 0.5 operating point."""
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            loss, _ = loss_fn(out.logits, batch["gold_type_id"], batch["teacher_distribution"],
                              batch["sample_weight"], batch.get("pair_metadata"))
            probs = torch.softmax(out.logits, dim=-1)
            pred = (1.0 - probs[:, 0]) >= 0.5
            all_y.extend(batch["gold_type_id"].tolist())
            all_pred.extend([int(p) for p in pred.tolist()])
            total_loss += float(loss)
            n += 1
    y = [1 if v != 0 else 0 for v in all_y]
    pred = all_pred
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 0)
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    return {
        "macro_f1": round(macro_f1, 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
        "acc": round((tp + tn) / max(len(y), 1), 4),
        "loss": round(total_loss / max(n, 1), 4),
        "n": len(y),
    }


def save_resume(model, optimizer, scheduler, path, epoch, step_in_epoch, global_step,
                best_metric, best_epoch, history):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # lean snapshot: trainable (LoRA/head) params only; base weights come from
    # the frozen checkpoint on resume. Old full-state snapshots still load via strict=False.
    trainable = {k: v for k, v in model.state_dict().items()
                 if any(tag in k for tag in ("lora_", "score", "head"))}
    torch.save({"model": trainable, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "step_in_epoch": step_in_epoch,
                "global_step": global_step, "best_metric": best_metric, "best_epoch": best_epoch,
                "history": history}, path)


def save_checkpoint(model, tokenizer, path, architecture="standard"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    (path / "architecture.txt").write_text(architecture, encoding="utf-8")


def load_checkpoint(model, path):
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model.load_adapter(str(path))
    else:
        model.load_state_dict(torch.load(str(path / "model.pt"), map_location="cpu"))
    return model
    # LoRA adapters only: save the (non-LoRA) classification head explicitly so
    # eval can reproduce training-time metrics (guide 28.1: classifier head must
    # be persisted; PeftModel.save_pretrained skips non-adapter modules).
    head_state = {k: v for k, v in model.state_dict().items()
                  if any(tag in k for tag in ("score", "head")) and "lora_" not in k}
    if head_state:
        torch.save(head_state, path / "classifier_head.pt")
