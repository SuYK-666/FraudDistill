from __future__ import annotations

import hashlib
import re


_SPACE = re.compile(r"\s+")


def duplicate_audit(splits: dict[str, list[dict]]) -> dict:
    exact_text = _cross_split_duplicates(splits, "pair_text_exact", lambda row: _norm_pair(row, exact=True))
    exact_query = _cross_split_duplicates(splits, "query_exact", lambda row: _norm_text(row.get("user_query", ""), exact=True))
    exact_answer = _cross_split_duplicates(splits, "answer_exact", lambda row: _norm_text(row.get("target_model_answer", ""), exact=True))
    near_text = _cross_split_duplicates(splits, "pair_text_near", lambda row: _norm_pair(row, exact=False))
    hit_count = exact_text["hit_count"] + exact_query["hit_count"] + exact_answer["hit_count"] + near_text["hit_count"]
    return {
        "passed": hit_count == 0,
        "hit_count": hit_count,
        "exact_pair_text": exact_text,
        "exact_query": exact_query,
        "exact_answer": exact_answer,
        "near_pair_text": near_text,
    }


def _cross_split_duplicates(splits: dict[str, list[dict]], name: str, key_fn) -> dict:
    seen: dict[str, dict] = {}
    hits: list[dict] = []
    for split_name, rows in splits.items():
        for row in rows:
            key = key_fn(row)
            if not key:
                continue
            digest = _sha1(key)
            prior = seen.get(digest)
            if prior and prior["split"] != split_name:
                hits.append(
                    {
                        "type": name,
                        "digest": digest,
                        "first_split": prior["split"],
                        "first_id": prior["id"],
                        "second_split": split_name,
                        "second_id": row.get("id"),
                    }
                )
            else:
                seen[digest] = {"split": split_name, "id": row.get("id")}
    return {"passed": not hits, "hit_count": len(hits), "examples": hits[:50]}


def _norm_pair(row: dict, exact: bool) -> str:
    return f"{_norm_text(row.get('user_query', ''), exact)}\n{_norm_text(row.get('target_model_answer', ''), exact)}"


def _norm_text(value: object, exact: bool) -> str:
    text = str(value or "").strip().lower()
    text = _SPACE.sub(" ", text)
    if not exact:
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
