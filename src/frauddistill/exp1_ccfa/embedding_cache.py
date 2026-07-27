from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def cache_fingerprint(texts: list[str], config: dict) -> str:
    payload = {
        "model_id": config["model_id"],
        "revision": config.get("revision", "main"),
        "tokenizer_revision": config.get("tokenizer_revision", config.get("revision", "main")),
        "prefix": config.get("prefix", ""),
        "query_prefix": config.get("query_prefix", ""),
        "passage_prefix": config.get("passage_prefix", ""),
        "max_length": int(config.get("max_length", 256)),
        "pooling": config.get("pooling", "mean"),
        "normalize": bool(config.get("normalize", True)),
        "backend": config.get("backend", "transformers"),
        "dtype": config.get("dtype", "float32"),
        "text_sha256": [_sha256(text) for text in texts],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class FrozenEmbeddingCache:
    def __init__(self, cache_dir: str | Path, config: dict):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)

    def encode(self, texts: list[str], prefix: str | None = None) -> np.ndarray:
        config = dict(self.config)
        if prefix is not None:
            config["prefix"] = prefix
        fingerprint = cache_fingerprint(texts, config)
        path = self.cache_dir / f"{fingerprint}.npy"
        meta_path = self.cache_dir / f"{fingerprint}.json"
        if path.exists():
            return np.load(path)
        vectors = self._encode_uncached(texts, config)
        np.save(path, vectors)
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "config": config, "rows": len(texts)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return vectors

    def _encode_uncached(self, texts: list[str], config: dict) -> np.ndarray:
        import torch
        from tqdm import tqdm
        from transformers import AutoModel, AutoTokenizer

        model_id = config["model_id"]
        revision = config.get("revision", "main")
        tokenizer_revision = config.get("tokenizer_revision", revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
        model = AutoModel.from_pretrained(model_id, revision=revision)
        model.eval()
        model.to(torch.device("cpu"))
        prefix = str(config.get("prefix", ""))
        max_length = int(config.get("max_length", 256))
        batch_size = int(config.get("batch_size", 128))
        pooling = str(config.get("pooling", "mean"))
        if pooling != "mean":
            raise ValueError(f"unsupported pooling for FrozenEmbeddingCache: {pooling}")
        dtype = str(config.get("dtype", "float32"))
        if dtype != "float32":
            raise ValueError(f"unsupported dtype for FrozenEmbeddingCache: {dtype}")
        vectors = []
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc=f"encode {model_id}", leave=False):
                batch = [prefix + str(text) for text in texts[start : start + batch_size]]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                output = model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
                pooled = (output.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                if config.get("normalize", True):
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
