#!/usr/bin/env python3
"""Copy verified amd64 manifests to SWR, then verify the destination digest."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    for image in json.loads(a.lock.read_text())["images"]:
        if image.get("copy_upstream") is False:
            print(f"distribution artifact (no upstream copy): {image['name']}")
            continue
        if not image.get("source_digest"):
            print(f"not_ready: {image['name']}")
            continue
        source = image["source"] + "@" + image["source_digest"]
        target = image["destination_tag"]
        cmd = ["crane", "copy", "--platform", "linux/amd64", "--no-clobber", source, target]
        if not a.execute:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)
        digest = subprocess.check_output(["crane", "digest", "--platform", "linux/amd64", target], text=True).strip()
        if digest != image["source_digest"]:
            raise ValueError(f"destination mismatch: {image['name']}")
        print(f"verified: {target.rsplit(':',1)[0]}@{digest}")


if __name__ == "__main__":
    main()
