from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_no} of {path} is not a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either a JSON array file or a JSONL file."""
    target = Path(path)
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return list(read_jsonl(target))
    if isinstance(loaded, list):
        if not all(isinstance(item, dict) for item in loaded):
            raise ValueError(f"{path} JSON array must contain objects")
        return loaded
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError(f"{path} must be a JSON object, JSON array, or JSONL file")
