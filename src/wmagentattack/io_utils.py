"""JSON and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = (
                record.model_dump(mode="json")
                if isinstance(record, BaseModel)
                else record
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

