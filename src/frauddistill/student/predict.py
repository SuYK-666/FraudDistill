# -*- coding: utf-8 -*-
"""Neural-student inference (guide 21.1): label / risk_type / risk_score / type_probabilities."""
from __future__ import annotations

import torch

from frauddistill.student.dataset import ID_TO_LABEL, neural_input_text


def predict_with_rule_fallback(rows: list[dict]) -> list[dict]:
    from frauddistill.eval.rule_baseline import predict_rule
    return [predict_rule(row) for row in rows]


def predict_neural_batch(model, tokenizer, rows, device=None, batch_size=8,
                         max_length=1536, architecture="standard"):
    """Batch inference; rows are dicts with user_query/target_model_answer/context."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            texts = [neural_input_text(str(r.get("user_query", "")), str(r.get("target_model_answer", "")),
                                       r.get("context")) for r in chunk]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for r, p in zip(chunk, probs):
                type_probs = {ID_TO_LABEL[j]: float(p[j]) for j in range(p.shape[0])}
                risk_score = float(1.0 - p[0])
                risk_type = max(type_probs, key=type_probs.get)
                out.append({
                    "id": r.get("id"),
                    "label": "unsafe" if risk_score >= 0.5 else "safe",
                    "risk_type": risk_type,
                    "risk_score": round(risk_score, 4),
                    "type_probabilities": {k: round(v, 4) for k, v in type_probs.items()},
                })
    return out
