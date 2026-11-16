#!/usr/bin/env python3
"""Create a new, immutable token-ID request set from historical source bytes.

No downloads; tokenizers==0.21.4 is required. The official model encoder is
loaded from the explicitly supplied, hash-recorded model directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

FILES = {
    "s1k": "s1k-e75.jsonl", "s8k": "s8k-e75.jsonl",
    "s32k": "s32k-e75.jsonl", "s128k": "s128k-e75.jsonl",
    "longout": "slongout-e75.jsonl", "sharegpt": "sharegpt-natural-e75-16384.jsonl",
    "rag": "rag-natural-e75.jsonl",
    "warm-s1k": "warm-s1k-e75.jsonl", "warm-s8k": "warm-s8k-c1-e75.jsonl",
    "warm-s32k": "warm-s32k-c1-e75.jsonl", "warm-s128k": "warm-s128k-c1-e75.jsonl",
    "warm-longout": "warm-slongout-c1-e75.jsonl",
    "warm-sharegpt": "warm-sharegpt-natural-c1-e75-16384.jsonl",
    "warm-rag": "warm-rag-natural-c1-e75.jsonl",
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.model / "tokenizer.json"))
    spec = importlib.util.spec_from_file_location("frozen_encoder", a.model / "encoding/encoding.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    # Refuse to replace an existing prepared set, including partial output.
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "kind": "b300-paired-text-v1", "status": "prepared_not_engine_validated",
                "source_catalog_sha256": sha(a.catalog / "SHA256SUMS"),
                "encoder_sha256": sha(a.model / "encoding/encoding.py"),
                "tokenizer_sha256": sha(a.model / "tokenizer.json"),
                "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 0},
                "thinking_mode": "thinking", "reasoning_effort": 75,
                "note": "Historical messages reused; sampling is NEW, not historical A100 performance.",
                "files": {}}
    for name, filename in FILES.items():
        src = a.catalog / "historical-a100-20261031" / filename
        source_sha = sha(src)
        rows, seen, lengths = [], set(), []
        for i, line in enumerate(src.open()):
            if not line.strip():
                continue
            body = json.loads(line)
            prompt = encoder.encode_messages(body["messages"], thinking_mode="thinking", reasoning_effort=75)
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            identity = hashlib.sha256(json.dumps(body["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if identity in seen:
                raise ValueError(f"duplicate source messages in {filename}:{i+1}")
            seen.add(identity)
            # Explicitly preserve the source's output budget. Natural workloads
            # require a budget/finish_reason preflight before formal comparison.
            rows.append({"id": f"{name}:{i:04d}", "source_sha256": source_sha,
                         "source_row": i, "messages_sha256": identity,
                         "prompt_token_ids": ids, "actual_isl": len(ids),
                         "max_tokens": body["max_tokens"],
                         "ignore_eos": body.get("ignore_eos", False),
                         "temperature": 1.0, "top_p": 1.0, "seed": 0})
            lengths.append(len(ids))
        expected_n = 3 if name.startswith("warm-") else 128
        if len(rows) != expected_n:
            raise ValueError(f"{filename}: expected {expected_n} distinct source records")
        out = a.output / f"{name}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["files"][out.name] = {"sha256": sha(out), "source": str(src.relative_to(a.catalog)),
            "source_sha256": source_sha, "n": len(rows), "isl_min": min(lengths), "isl_max": max(lengths),
            "max_tokens": sorted({r["max_tokens"] for r in rows}),
            "max_input_plus_budget": max(r["actual_isl"] + r["max_tokens"] for r in rows)}
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
