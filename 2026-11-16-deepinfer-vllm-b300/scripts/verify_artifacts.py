#!/usr/bin/env python3
"""Verify a SHA256SUMS tree without trusting paths outside that tree."""
import argparse
import hashlib
from pathlib import Path


def verify(root, expected):
    root = root.resolve()
    sums = root / "SHA256SUMS"
    if hashlib.sha256(sums.read_bytes()).hexdigest() != expected:
        raise ValueError("SHA256SUMS identity mismatch")
    count = 0
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        p = (root / rel).resolve()
        if root not in p.parents or not p.is_file():
            raise ValueError(f"missing or unsafe manifest member: {rel}")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(b)
        if h.hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {rel}")
        count += 1
    if not count:
        raise ValueError("empty manifest")
    print(f"verified_files={count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    verify(a.root, a.manifest_sha256)
