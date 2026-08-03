"""Content-hash request cache (guide section 13). Prompt version is part of the key."""
from __future__ import annotations

import hashlib
import json
import os
import threading


class RequestCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, prompt_version: str, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": model,
            "prompt_version": prompt_version,
            "system": system_prompt,
            "user": user_prompt,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def get(self, key: str):
        p = self.path(key)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.hits += 1
                return json.load(f)
        self.misses += 1
        return None

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        with self._lock:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}