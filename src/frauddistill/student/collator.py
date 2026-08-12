# -*- coding: utf-8 -*-
"""Neural-student batch collator (guide 4.2, 20.1).

Pads input_ids/attention_mask to the longest sequence in the batch and packs
gold_type_id, teacher_distribution, sample_weight and pair metadata.
For the interaction architecture, query/answer segment masks are derived from
the input text markers ([QUERY] / [ANSWER]).
"""
from __future__ import annotations

import torch


def segment_masks(input_ids, tokenizer):
    """Build query/answer masks from special segment ids (interaction head)."""
    qid = _seg_id(tokenizer, "[QUERY]")
    aid = _seg_id(tokenizer, "[ANSWER]")
    ids = input_ids
    query_mask = torch.zeros_like(ids, dtype=torch.float32)
    answer_mask = torch.zeros_like(ids, dtype=torch.float32)
    in_q = torch.zeros(ids.shape[0], dtype=torch.bool)
    in_a = torch.zeros(ids.shape[0], dtype=torch.bool)
    for t in range(ids.shape[1]):
        col = ids[:, t]
        in_q = in_q | (col == qid)
        in_a = in_a | (col == aid)
        query_mask[:, t] = in_q.float()
        answer_mask[:, t] = in_a.float()
        in_q = in_q & (col != qid)
        in_a = in_a & (col != aid)
    return query_mask, answer_mask


def _seg_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def head_tail_truncate(tokens: list[int], max_length: int, answer_start_token_id=None) -> list[int]:
    """Guide 12.5: keep query/context head + answer head 45% / tail 55%.

    If the answer marker token is found, the query part keeps its first 40% of
    the budget and the answer part keeps front 45% + back 55% of its share;
    otherwise a plain head+tail split is used. Never truncates from the tail only.
    """
    if len(tokens) <= max_length:
        return tokens
    budget = max(max_length - 4, 32)
    if answer_start_token_id is not None and answer_start_token_id in tokens:
        split_at = tokens.index(answer_start_token_id)
        q_part = tokens[:split_at]
        a_part = tokens[split_at:]
        q_budget = int(budget * 0.40)
        a_budget = budget - q_budget
        if len(q_part) > q_budget:
            q_part = q_part[:q_budget]
        if len(a_part) > a_budget:
            head_n = int(a_budget * 0.45)
            tail_n = a_budget - head_n
            a_part = a_part[:head_n] + a_part[-tail_n:]
        return q_part + a_part
    head_n = int(budget * 0.5)
    return tokens[:head_n] + tokens[-(budget - head_n):]


def neural_collate(batch, tokenizer, max_length=None, architecture="standard", trunc_mode="headtail"):
    texts = [ex["text"] for ex in batch]
    if max_length is not None:
        answer_marker = tokenizer("[ANSWER]", add_special_tokens=False)["input_ids"]
        answer_id = answer_marker[0] if answer_marker else None
        encs = []
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            if trunc_mode == "tail":
                encs.append(ids[:max_length])
            else:
                encs.append(head_tail_truncate(ids, max_length, answer_id))
        enc = tokenizer.pad({"input_ids": encs}, padding=True, return_tensors="pt")
    else:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "gold_type_id": torch.tensor([ex["gold_type_id"] for ex in batch], dtype=torch.long),
        "teacher_distribution": torch.tensor([ex["teacher_distribution"] for ex in batch], dtype=torch.float32),
        "sample_weight": torch.tensor([ex["sample_weight"] for ex in batch], dtype=torch.float32),
        "ids": [ex["id"] for ex in batch],
        "group_ids": [ex["group_id"] for ex in batch],
        "sources": [ex.get("source", "") for ex in batch],
        "subtypes": [ex.get("subtype", "") for ex in batch],
        "languages": [ex.get("language", "") for ex in batch],
        "gold_binary": torch.tensor([1 if ex["gold_label"] == "unsafe" else 0 for ex in batch], dtype=torch.long),
        "teacher_confidence": torch.tensor([float(ex.get("teacher_confidence") if ex.get("teacher_confidence") is not None else 0.5) for ex in batch], dtype=torch.float32),
        "teacher_gold_agree": torch.tensor([bool(ex["teacher_gold_agree"]) if ex.get("teacher_gold_agree") is not None else True for ex in batch], dtype=torch.bool),
        "teacher_only": torch.tensor([bool(ex.get("teacher_only", False)) for ex in batch], dtype=torch.bool),
    }
    if architecture == "interaction":
        qm, am = segment_masks(enc["input_ids"], tokenizer)
        out["query_mask"] = qm
        out["answer_mask"] = am
    pairs = []
    pair_index = {}
    for i, ex in enumerate(batch):
        pid = ex.get("pair_id")
        if pid:
            pair_index.setdefault(pid, []).append(i)
    for pid, idxs in pair_index.items():
        if len(idxs) == 2:
            a, b = idxs
            # order: unsafe side must come first
            if batch[a]["gold_label"] == "safe":
                a, b = b, a
            pairs.append({"unsafe_idx": a, "safe_idx": b, "pair_id": pid})
    out["pair_metadata"] = pairs
    return out
