#!/usr/bin/env python3
"""
Corpus provenance: record where each PDF in data/raw came from, and fetch missing ones.

data/raw/ is not version-controlled, so data/sources.yaml is what makes the corpus
reproducible: for each document the source URL and the SHA-256 of the exact file the
results were produced from. BaFin replaces documents under the same URL when a circular
is amended; a changed hash is how that shows up.

Run:
    python scripts/fetch_corpus.py --record   # add/refresh entries for the PDFs already in data/raw
    python scripts/fetch_corpus.py            # download entries whose file is missing; verify all hashes
"""
import argparse
import hashlib
import sys
import urllib.request

import yaml

from regrag import config


def sha256(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sources() -> dict:
    if not config.SOURCES_PATH.exists():
        return {"documents": []}
    return yaml.safe_load(config.SOURCES_PATH.read_text(encoding="utf-8")) or {"documents": []}


def save_sources(data: dict) -> None:
    header = config.SOURCES_PATH.read_text(encoding="utf-8").split("documents:")[0] \
        if config.SOURCES_PATH.exists() else ""
    body = yaml.safe_dump({"documents": data["documents"]}, allow_unicode=True, sort_keys=False)
    config.SOURCES_PATH.write_text(header + body, encoding="utf-8")


def record() -> int:
    data = load_sources()
    entries = {d["file"]: d for d in data["documents"]}
    for path in sorted(config.RAW_DIR.glob("*.pdf")):
        entry = entries.setdefault(path.name, {"file": path.name, "title": None, "url": None})
        entry["sha256"] = sha256(path)
        entry["bytes"] = path.stat().st_size
        print(f"recorded {path.name}  {entry['sha256'][:12]}  url: {entry.get('url') or '(fill in)'}")
    data["documents"] = list(entries.values())
    save_sources(data)
    return 0


def fetch() -> int:
    failures = 0
    for doc in load_sources()["documents"]:
        path = config.RAW_DIR / doc["file"]
        if not path.exists():
            if not doc.get("url"):
                print(f"missing  {doc['file']}: no url recorded")
                failures += 1
                continue
            print(f"download {doc['file']} <- {doc['url']}")
            config.RAW_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(doc["url"], path)
        actual = sha256(path)
        if doc.get("sha256") and actual != doc["sha256"]:
            print(f"CHANGED  {doc['file']}: sha256 {actual[:12]} != recorded {doc['sha256'][:12]}")
            failures += 1
        else:
            print(f"ok       {doc['file']}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    return record() if args.record else fetch()


if __name__ == "__main__":
    sys.exit(main())
