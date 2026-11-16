#!/usr/bin/env python3
"""Make a deterministic offline dataset layer; never include local credentials.

Includes only the verified standard-v1 tree, prepared text requests and explicitly
supplied model metadata/encoding. Does not include weights, home dirs or caches.
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from verify_artifacts import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--model-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.catalog, "a49ef30dab14921a7b1cc804c847f65f721cf44aa5d59eddc47561b868e2f602")
    manifest = json.loads((a.prepared / "manifest.json").read_text())
    for f, entry in manifest["files"].items():
        if hashlib.sha256((a.prepared / f).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"prepared checksum mismatch: {f}")
    selected = []
    for root, dest in [(a.catalog, "datasets/standard-v1"), (a.prepared, "datasets/b300-paired-text-v1"),
                       (a.model_metadata, "datasets/model-metadata")]:
        for f in sorted(root.rglob("*")):
            if f.is_symlink():
                raise ValueError(f"symlink not allowed: {f.relative_to(root)}")
            if f.is_file():
                if root == a.model_metadata and f.relative_to(root).as_posix() not in (
                    "config.json", "tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
                    continue
                selected.append((f, str(Path(dest) / f.relative_to(root))))
    with a.output.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for f, name in selected:
                info = tar.gettarinfo(str(f), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with f.open("rb") as src:
                    tar.addfile(info, src)
    h = hashlib.sha256()
    with a.output.open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    result = {"files":len(selected), "bytes":a.output.stat().st_size, "sha256":h.hexdigest(),
              "standard_manifest_sha256":manifest["source_catalog_sha256"],
              "prepared_manifest_sha256":hashlib.sha256((a.prepared/"manifest.json").read_bytes()).hexdigest()}
    a.output.with_suffix(a.output.suffix+".json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
