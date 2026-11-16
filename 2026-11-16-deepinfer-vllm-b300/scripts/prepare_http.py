#!/usr/bin/env python3
"""Create candidate HTTP payloads from the same frozen messages as native IDs.

Does not claim parity with the server encoder: final wire bodies and server token
counts must pass G2 against the native manifest before measurements are compared.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--prepared-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", default="deepseek-v4.1-flash-1017")
    a = p.parse_args()
    d = json.loads(a.prepared_manifest.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    result = {"status":"candidate_http_not_wire_validated", "files":{}}
    for name, record in d["files"].items():
        source = a.catalog / record["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
            raise ValueError("source SHA mismatch")
        out = a.output / name
        with out.open("x") as target:
            for line in source.open():
                if not line.strip():
                    continue
                body = json.loads(line)
                body.update(model=a.model_name, temperature=1.0, top_p=1.0, seed=0,
                            stream=True, stream_options={"include_usage":True},
                            chat_template_kwargs={"thinking":True,"reasoning_effort":75})
                target.write(json.dumps(body, ensure_ascii=False)+"\n")
        result["files"][name] = {"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
            "n":record["n"], "native_reference_sha256":record["sha256"]}
    (a.output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
