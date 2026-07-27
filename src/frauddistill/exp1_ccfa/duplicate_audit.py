from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


_SPACE = re.compile(r"\s+")


def duplicate_audit(splits: dict[str, list[dict]]) -> dict:
    exact_text = _cross_split_duplicates(splits, "pair_text_exact", lambda row: _norm_pair(row, exact=True))
    exact_query = _cross_split_duplicates(splits, "query_exact", lambda row: _norm_text(row.get("user_query", ""), exact=True))
    exact_answer = _cross_split_duplicates(splits, "answer_exact", lambda row: _norm_text(row.get("target_model_answer", ""), exact=True))
    near_text = _cross_split_duplicates(splits, "pair_text_near", lambda row: _norm_pair(row, exact=False))
    cosine_near = char_ngram_near_duplicate_audit(splits)
    hit_count = (
        exact_text["hit_count"]
        + exact_query["hit_count"]
        + exact_answer["hit_count"]
        + near_text["hit_count"]
        + cosine_near["hit_count"]
    )
    return {
        "passed": hit_count == 0,
        "hit_count": hit_count,
        "exact_pair_text": exact_text,
        "exact_query": exact_query,
        "exact_answer": exact_answer,
        "near_pair_text": near_text,
        "char_5gram_near": cosine_near,
    }


def char_ngram_near_duplicate_audit(splits: dict[str, list[dict]], threshold: float = 0.92, block_prefix: int = 8) -> dict:
    records: list[tuple[str, str, str]] = []
    for split_name, rows in splits.items():
        for row in rows:
            text = _norm_pair(row, exact=True)
            if len(text) < 32:
                continue
            records.append((split_name, str(row.get("id")), text))
    if len(records) < 2:
        return {"passed": True, "hit_count": 0, "threshold": threshold, "examples": []}
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), min_df=1, norm="l2")
    matrix = vectorizer.fit_transform([record[2] for record in records])
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, (_, _, text) in enumerate(records):
        blocks[_sha1(text[:128])[:block_prefix]].append(index)
        blocks[_sha1(text[-128:])[:block_prefix]].append(index)
    hits: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for indices in blocks.values():
        unique = sorted(set(indices))
        if len(unique) < 2:
            continue
        sub = matrix[unique]
        sim = (sub @ sub.T).toarray()
        for i_pos in range(len(unique)):
            for j_pos in range(i_pos + 1, len(unique)):
                i = unique[i_pos]
                j = unique[j_pos]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if records[i][0] == records[j][0]:
                    continue
                score = float(sim[i_pos, j_pos])
                if score >= threshold:
                    hits.append(
                        {
                            "type": "char_5gram_cosine",
                            "similarity": score,
                            "first_split": records[i][0],
                            "first_id": records[i][1],
                            "second_split": records[j][0],
                            "second_id": records[j][1],
                        }
                    )
    hits = sorted(hits, key=lambda row: -float(row["similarity"]))
    return {"passed": not hits, "hit_count": len(hits), "threshold": threshold, "examples": hits[:50]}


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
