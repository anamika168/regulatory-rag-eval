"""Reading and writing the derived artefacts in data/processed and results/."""

import json
from pathlib import Path
from typing import Iterable

from regrag.chunk import Chunk
from regrag.ingest import Page


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pages(path: Path, pages: list[Page]) -> None:
    write_jsonl(path, (vars(p) for p in pages))


def load_pages(path: Path) -> list[Page]:
    return [Page(**row) for row in read_jsonl(path)]


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    write_jsonl(path, (c.as_dict() for c in chunks))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
