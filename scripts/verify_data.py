"""Verify checksums of the versioned derived data and optional raw Excel."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",action="store_true")
    args=parser.parse_args()
    folder=ROOT/("data/raw" if args.raw else "data/derived")
    manifest=ROOT/("data/manifest.json" if args.raw else "data/derived/provenance.json")
    entries=json.loads(manifest.read_text(encoding="utf-8"))["files"]
    for item in entries:
        path=folder/item["path"]
        if not path.is_file() or path.stat().st_size!=item["size_bytes"]:
            raise SystemExit(f"Missing or wrong file size: {item['path']}")
        if not item.get("sha256"):
            raise SystemExit(f"Expected SHA-256 missing from manifest: {item['path']}")
        if digest(path)!=item["sha256"]:
            raise SystemExit(f"Checksum mismatch: {item['path']}")
        print("OK",item["path"])
    print(f"Verified {len(entries)} files.")

if __name__=="__main__":main()
