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
                best_metric, best_epoch, history, no_improve=0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # lean snapshot: trainable (LoRA/head) params only; base weights come from
    # the frozen checkpoint on resume. Old full-state snapshots still load via strict=False.
    trainable = {k: v for k, v in model.state_dict().items()
                 if any(tag in k for tag in ("lora_", "score", "head"))}
    torch.save({"model": trainable, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "step_in_epoch": step_in_epoch,
                "global_step": global_step, "best_metric": best_metric, "best_epoch": best_epoch,
                "history": history, "no_improve": no_improve}, path)


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


def evaluate_neural_full(model, loader, device, architecture="standard",
                         loss_fn=None, threshold=0.5):
    """Full dev/test evaluation: macro-F1, recall, fpr, AUPRC, MCC, ECE,
    Brier plus real/synthetic/language/subtype slices (guide 17)."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, roc_auc_score)
    model.eval()
    all_y, all_pred, all_scores, all_gtype = [], [], [], []
    all_ids, all_src, all_sub, all_lang = [], [], [], []
    total_loss = 0.0
    n_batch = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            if loss_fn is not None:
                if getattr(loss_fn, "uses_batch", False):
                    loss, _ = loss_fn(out.logits, batch)
                else:
                    loss, _ = loss_fn(out.logits, batch["gold_type_id"], batch["teacher_distribution"],
                                      batch["sample_weight"], batch.get("pair_metadata"))
                total_loss += float(loss)
                n_batch += 1
            probs = torch.softmax(out.logits, dim=-1)
            scores = (1.0 - probs[:, 0]).tolist()
            all_y.extend(batch["gold_type_id"].tolist())
            all_pred.extend([1 if sc >= threshold else 0 for sc in scores])
            all_scores.extend(scores)
            all_gtype.extend(batch["gold_type_id"].tolist())
            all_ids.extend(batch.get("ids", []))
            all_src.extend(batch.get("sources", []))
            all_sub.extend(batch.get("subtypes", []))
            all_lang.extend(batch.get("languages", []))
    y = [1 if v != 0 else 0 for v in all_y]
    y_arr = np.array(y, dtype=int)
    p_arr = np.array(all_pred, dtype=int)
    scores_arr = np.array(all_scores, dtype=float)
    tn = int(((p_arr == 0) & (y_arr == 0)).sum()); fp = int(((p_arr == 1) & (y_arr == 0)).sum())
    fn = int(((p_arr == 0) & (y_arr == 1)).sum()); tp = int(((p_arr == 1) & (y_arr == 1)).sum())

    def ece(ps, ys, bins=10):
        ps = np.clip(np.asarray(ps, float), 0, 1); ys = np.asarray(ys, float)
        edges = np.linspace(0, 1, bins + 1); out = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ps >= lo) & (ps <= hi) if hi == 1.0 else (ps >= lo) & (ps < hi)
            if not m.any():
                continue
            out += (m.sum() / len(ys)) * abs(float(ps[m].mean()) - float(ys[m].mean()))
        return float(out)

    m = {
        "macro_f1": round(float(f1_score(y_arr, p_arr, average="macro", zero_division=0)), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
        "acc": round(float(accuracy_score(y_arr, p_arr)), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_arr, p_arr)), 4),
        "mcc": round(float(matthews_corrcoef(y_arr, p_arr)), 4),
        "auprc": round(float(average_precision_score(y_arr, scores_arr)), 4),
        "auroc": round(float(roc_auc_score(y_arr, scores_arr)), 4),
        "ece": round(ece(scores_arr, y_arr), 4),
        "brier": round(float(np.mean((scores_arr - y_arr) ** 2)), 4),
        "n": len(y),
    }
    if n_batch:
        m["loss"] = round(total_loss / n_batch, 4)
    m["type_ids"] = list(all_gtype)

    slices = {}
    for key, cond in [("real_only", lambda s, sub, lg: s != "synthetic"),
                      ("synthetic_only", lambda s, sub, lg: s == "synthetic"),
                      ("en", lambda s, sub, lg: lg == "en"),
                      ("zh", lambda s, sub, lg: lg == "zh")]:
        idx = [i for i in range(len(y)) if cond(all_src[i], all_sub[i], all_lang[i])]
        if not idx:
            continue
        ys = y_arr[idx]; ps = p_arr[idx]; sc = scores_arr[idx]
        tp2 = int(((ps == 1) & (ys == 1)).sum()); fn2 = int(((ps == 0) & (ys == 1)).sum())
        fp2 = int(((ps == 1) & (ys == 0)).sum()); tn2 = int(((ps == 0) & (ys == 0)).sum())
        slices[key] = {"n": len(idx),
                       "macro_f1": round(float(f1_score(ys, ps, average="macro", zero_division=0)), 4),
                       "recall": round(tp2 / max(tp2 + fn2, 1), 4),
                       "fpr": round(fp2 / max(tn2 + fp2, 1), 4),
                       "auprc": round(float(average_precision_score(ys, sc)), 4)
                       if len(np.unique(ps)) > 1 and 0 < ys.sum() < len(ys) else None}
    for sub, key, kind in [("direct_fraud", "direct", "recall"), ("trust_facilitation", "trust", "recall"),
                           ("partial_leakage", "leakage", "recall"), ("clean_refusal", "clean_refusal", "fpr"),
                           ("hard_safe", "hard_safe", "fpr"), ("over_refusal", "over_refusal", "recall"),
                           ("context_flip", "context_flip", "recall"), ("quotation_analysis", "quotation", "fpr")]:
        idx = [i for i in range(len(y)) if all_sub[i] == sub]
        if not idx:
            continue
        ys = y_arr[idx]; ps = p_arr[idx]
        if kind == "recall":
            tp2 = int(((ps == 1) & (ys == 1)).sum()); fn2 = int(((ps == 0) & (ys == 1)).sum())
            slices[key + "_recall"] = round(tp2 / max(tp2 + fn2, 1), 4)
        else:
            fp2 = int(((ps == 1) & (ys == 0)).sum()); tn2 = int(((ps == 0) & (ys == 0)).sum())
            slices[key + "_fpr"] = round(fp2 / max(tn2 + fp2, 1), 4)
        slices[key + "_n"] = len(idx)
    m["slices"] = slices
    return m


def train_neural_final(model, train_loader, dev_loader, loss_fn, tokenizer,
                       epochs=2, lr_lora=1e-4, lr_head=5e-4, weight_decay=0.01,
                       warmup_ratio=0.05, grad_accum=16, eval_steps=40, save_steps=40,
                       patience=4, max_grad_norm=1.0, seed=11, out_dir=None,
                       device=None, architecture="standard", resume=None,
                       max_steps=None, log_every=20):
    """Final student training loop (guide 14, 34-36)."""
    from tqdm import tqdm
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    groups = split_param_groups(model, lr_lora, lr_head, weight_decay)
    optimizer = torch.optim.AdamW(groups)
    # optimizer-step accounting: progress bar / warmup / log denominators
    # count optimizer updates, not micro-batches (guide 34, 36)
    total_micro = len(train_loader) * epochs
    total_steps = max(1, total_micro // grad_accum)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(warmup_steps, 1)))
    history = {"train": [], "dev": []}
    best_metric = -1.0
    best_step = 0
    best_state = None
    global_step = 0
    start_epoch = 0
    start_step = 0
    no_improve = 0
    real_down_streak = 0
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"])
        start_step = int(ck["step_in_epoch"])
        global_step = int(ck["global_step"])
        best_metric = float(ck.get("best_metric", -1.0))
        best_step = int(ck.get("best_step", 0))
        no_improve = int(ck.get("no_improve", 0))
        history = ck.get("history", history)
        print(f"[final] resumed from {resume}: epoch={start_epoch} step={start_step} global={global_step}", flush=True)

    pbar = tqdm(total=total_steps, desc="final-distill", unit="step", dynamic_ncols=True)
    pbar.update(min(global_step, total_steps))
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss_gold": 0.0, "loss_binary": 0.0, "loss_kl": 0.0, "loss_pair": 0.0, "loss_total": 0.0}
        _steps_since_log = 0
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            if getattr(loss_fn, "uses_batch", False):
                loss, comps = loss_fn(out.logits, batch)
            else:
                loss, comps = loss_fn(out.logits, batch["gold_type_id"], batch["teacher_distribution"],
                                      batch["sample_weight"], batch.get("pair_metadata"))
            loss = loss / grad_accum
            loss.backward()
            for k in running:
                running[k] += float(comps[k]) / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"abort: non-finite loss at micro-step {step} (guide 35)")
            if (step + 1) % grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(torch.as_tensor(gnorm)):
                    raise RuntimeError(f"abort: non-finite grad norm at step {global_step} (guide 35)")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                _steps_since_log += 1
                pbar.update(1)
                pbar.set_postfix({k: f"{v / max(_steps_since_log, 1):.4f}" for k, v in running.items()})
                if max_steps is not None and global_step >= max_steps:
                    print(f"max_steps reached ({max_steps}); stopping", flush=True)
                    if best_state:
                        model.load_state_dict(best_state)
                    pbar.close()
                    return best_state, history
                if global_step % log_every == 0 and out_dir:
                    save_resume(model, optimizer, scheduler, out_dir / "resume.pt", epoch, step, global_step,
                                best_metric, best_step, history, no_improve=no_improve)
                if global_step % log_every == 0:
                    lr0 = optimizer.param_groups[0]["lr"]
                    lr1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr0
                    safe_n = int((batch.get("gold_binary", torch.zeros(1)) == 0).sum())
                    en_n = int(sum(1 for x in batch.get("languages", []) if x == "en"))
                    srcs = {}
                    for x in batch.get("sources", []):
                        srcs[x] = srcs.get(x, 0) + 1
                    to_n = int(batch.get("teacher_only", torch.zeros(1)).sum())
                    print(f"  [step {global_step}/{total_steps}] " +
                          " ".join(f"{k}={v / max(_steps_since_log, 1):.4f}" for k, v in running.items()) +
                          f" | grad={float(gnorm):.4f} lr_lora={lr0:.2e} lr_head={lr1:.2e}"
                          f" | safe={safe_n}/{len(batch.get('gold_binary', []))} en={en_n}"
                          f" | src={srcs} teacher_only={to_n}", flush=True)
                    running = {k: 0.0 for k in running}
                    _steps_since_log = 0
            if global_step > 0 and global_step % save_steps == 0:
                if out_dir:
                    ck_path = out_dir / f"checkpoint-{global_step}"
                    save_checkpoint(model, tokenizer, ck_path, architecture)
                    _prune_checkpoints(out_dir, keep=3)
            if global_step > 0 and global_step % eval_steps == 0:
                dev_metric = evaluate_neural_full(model, dev_loader, device, architecture, loss_fn)
                history["dev"].append({"step": global_step, "epoch": epoch + 1, **dev_metric})
                mf1 = dev_metric["macro_f1"]
                sl = dev_metric.get("slices") or {}
                real_mf1 = sl.get("real_only", {}).get("macro_f1", 0.0)
                syn_mf1 = sl.get("synthetic_only", {}).get("macro_f1", 0.0)
                print(f"  [eval step {global_step}] macro_f1={mf1:.4f} recall={dev_metric['recall']:.4f} "
                      f"fpr={dev_metric['fpr']:.4f} auprc={dev_metric['auprc']:.4f} mcc={dev_metric['mcc']:.4f} "
                      f"4class={dev_metric.get('4class_macro_f1')} real_mf1={real_mf1:.4f} syn_mf1={syn_mf1:.4f} "
                      f"en_mf1={sl.get('en', {}).get('macro_f1', 0):.4f} "
                      f"zh_mf1={sl.get('zh', {}).get('macro_f1', 0):.4f}", flush=True)
                improved = mf1 > best_metric
                if improved:
                    best_metric = mf1
                    best_step = global_step
                    no_improve = 0
                    real_down_streak = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if out_dir:
                        save_checkpoint(model, tokenizer, out_dir / f"best_step{global_step}", architecture)
                        (out_dir / "best_metric.json").write_text(
                            json.dumps({"best_metric": best_metric, "best_step": best_step,
                                        "dev": dev_metric}, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    no_improve += 1
                    if len(history["dev"]) >= 2:
                        prev = history["dev"][-2]
                        prev_sl = prev.get("slices") or {}
                        prev_real = prev_sl.get("real_only", {}).get("macro_f1", 0.0)
                        prev_syn = prev_sl.get("synthetic_only", {}).get("macro_f1", 0.0)
                        if real_mf1 < prev_real and syn_mf1 > prev_syn:
                            real_down_streak += 1
                        else:
                            real_down_streak = 0
                    if no_improve >= patience or real_down_streak >= 3:
                        print(f"early stop at step {global_step} (no_improve={no_improve} real_streak={real_down_streak})", flush=True)
                        if best_state:
                            model.load_state_dict(best_state)
                        pbar.close()
                        return best_state, history
                if out_dir:
                    save_resume(model, optimizer, scheduler, out_dir / "resume.pt", epoch, step, global_step,
                                best_metric, best_step, history, no_improve=no_improve)
        dev_metric = evaluate_neural_full(model, dev_loader, device, architecture, loss_fn)
        history["dev"].append({"step": global_step, "epoch": epoch + 1, **dev_metric})
        print(f"epoch {epoch+1} dev: macro_f1={dev_metric['macro_f1']:.4f} recall={dev_metric['recall']:.4f} "
              f"fpr={dev_metric['fpr']:.4f}", flush=True)
        if dev_metric["macro_f1"] > best_metric:
            best_metric = dev_metric["macro_f1"]
            best_step = global_step
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if out_dir:
                save_checkpoint(model, tokenizer, out_dir / f"best_step{global_step}", architecture)
        if out_dir:
            save_resume(model, optimizer, scheduler, out_dir / "resume.pt", epoch, len(train_loader) - 1, global_step,
                        best_metric, best_step, history, no_improve=no_improve)
    pbar.close()
    if best_state:
        model.load_state_dict(best_state)
    return best_state, history


def _prune_checkpoints(out_dir, keep=3):
    """Keep the most recent `keep` checkpoint-* dirs (guide 28)."""
    import shutil
    cks = sorted([d for d in Path(out_dir).glob("checkpoint-*") if d.is_dir()],
                 key=lambda d: int(d.name.split("-")[1]))
    for d in cks[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
