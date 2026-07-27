from __future__ import annotations

import hashlib

from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components


def saferlhf_rows(split: str, revision: str = "main") -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, revision=revision)
    rows: list[dict] = []
    for index, item in enumerate(dataset):
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        prompt_hash = _sha1(prompt)[:24]
        category_union = _category_union(item)
        for response_index in (0, 1):
            answer = str(item.get(f"response_{response_index}") or "").strip()
            if not answer:
                continue
            safe = bool(item.get(f"is_response_{response_index}_safe"))
            rows.append(
                {
                    "id": f"saferlhf_{split}_{index}_{response_index}",
                    "source": "PKU-SafeRLHF",
                    "language": "English",
                    "user_query": prompt,
                    "target_model_answer": answer,
                    "pair_fraud_label": "safe" if safe else "unsafe",
                    "gold_label": "safe" if safe else "unsafe",
                    "label_provenance": "external_public_saferlhf_official_response_safety",
                    "source_prompt_id": f"saferlhf_{split}_{prompt_hash}",
                    "metadata": {
                        "source_dataset": "PKU-Alignment/PKU-SafeRLHF",
                        "official_split": split,
                        "source_row_index": index,
                        "response_index": response_index,
                        "prompt_source": item.get("prompt_source"),
                        "response_source": item.get(f"response_{response_index}_source"),
                        "is_response_safe": safe,
                        "harm_category": item.get(f"response_{response_index}_harm_category"),
                        "harm_category_union": category_union,
                        "severity_level": item.get(f"response_{response_index}_severity_level"),
                        "safer_response_id": item.get("safer_response_id"),
                        "better_response_id": item.get("better_response_id"),
                        "license": "CC-BY-NC-4.0",
                    },
                }
            )
    return attach_semantic_components(rows)


def _category_union(item: dict) -> dict:
    result: dict[str, bool] = {}
    for field in ("response_0_harm_category", "response_1_harm_category"):
        category = item.get(field) or {}
        for key, value in category.items():
            result[key] = bool(result.get(key)) or bool(value)
    return result


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
