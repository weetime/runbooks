#!/usr/bin/env python3
"""Audit native text point accounting; this does not approve output quality or gates."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(args):
    point = json.loads((args.point / "point.json").read_text())
    require(args.count > 0 and args.batch_size > 0, "count and batch size must be positive")
    source = rows(args.requests)[:args.count]
    outputs = rows(args.point / "outputs.jsonl")
    batches = rows(args.point / "batches.jsonl")
    require(len(source) == len(outputs) == args.count, "missing requests or outputs")
    require(len({r["id"] for r in source}) == args.count, "duplicate source IDs")
    require([r["id"] for r in source] == [r["id"] for r in outputs], "source/output order differs")
    require(point["requests_sha256"] == hashlib.sha256(args.requests.read_bytes()).hexdigest(), "request fingerprint mismatch")
    require(point["recipe_sha256"] == hashlib.sha256(args.recipe.read_bytes()).hexdigest(), "recipe fingerprint mismatch")
    require(point["image_manifest_ref"] == args.image and "@sha256:" in args.image, "image fingerprint mismatch")
    require(point["engine"] == json.loads(args.recipe.read_text())["engine"], "engine mismatch")
    require(point["client_sha256"] == hashlib.sha256(Path(__file__).with_name("native_batch.py").read_bytes()).hexdigest(), "client fingerprint mismatch")
    require(point["n"] == args.count and point["batch_size"] == args.batch_size, "point shape mismatch")
    require(point["measurement_mode"] == ("instrumented_native_g3" if args.g3 else "native_batch"), "measurement mode mismatch")
    fixed = all(r["ignore_eos"] for r in source)
    require(fixed if args.fixed else all(not r["ignore_eos"] for r in source), "EOS policy differs from requested audit")
    total = 0
    for request, output in zip(source, outputs):
        tokens = output["token_ids"]
        require(request["actual_isl"] == len(request["prompt_token_ids"]) == output["input_tokens"], "ISL mismatch")
        require(tokens and all(type(t) is int and t >= 0 for t in tokens), "invalid generated token IDs")
        require(len(tokens) <= request["max_tokens"], "output exceeds budget")
        require(not fixed or len(tokens) == request["max_tokens"], "fixed output shorter than budget")
        require(output["at_budget"] == (len(tokens) == request["max_tokens"]), "at_budget mismatch")
        require(output.get("finish_reason") not in (None, "", "missing"), "missing finish reason")
        total += len(tokens)
    expected_batches = [outputs[i:i+args.batch_size] for i in range(0, args.count, args.batch_size)]
    require(len(batches) == len(expected_batches), "batch count mismatch")
    for batch, expected in zip(batches, expected_batches):
        require(batch["ids"] == [r["id"] for r in expected] and batch["n"] == len(expected), "batch membership mismatch")
        require(batch["seconds"] > 0 and math.isfinite(batch["seconds"]), "invalid batch time")
        require(batch["actual_generated_tokens"] == sum(len(r["token_ids"]) for r in expected), "batch token mismatch")
    seconds = sum(b["seconds"] for b in batches)
    require(point["actual_generated_tokens"] == total, "total token mismatch")
    for key, value in (("seconds", seconds), ("output_tokens_per_second", total/seconds), ("output_tokens_per_second_per_gpu", total/seconds/8)):
        require(math.isclose(point[key], value, rel_tol=1e-9), key + " accounting mismatch")
    if args.g3:
        report = json.loads((args.point / "g3.json").read_text())
        require(args.count == 24 and args.batch_size == 1 and report["n"] == 24, "G3 shape mismatch")
        require(report["numeric_gate_passed"] is True, "G3 numeric stop line reached")
        require([r["id"] for r in report["requests"]] == [r["id"] for r in source], "G3 IDs differ")
    return {"accounting": "passed", "quality": "requires_human_review", "n": args.count,
            "batch_size": args.batch_size, "fixed_output": fixed,
            "actual_isl_min": min(r["actual_isl"] for r in source),
            "actual_isl_max": max(r["actual_isl"] for r in source),
            "actual_osl_min": min(len(r["token_ids"]) for r in outputs),
            "actual_osl_max": max(len(r["token_ids"]) for r in outputs),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in outputs)),
            "at_budget": sum(r["at_budget"] for r in outputs),
            "seconds": seconds, "output_tok_s": total/seconds, "output_tok_s_per_gpu": total/seconds/8}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--point", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--g3", action="store_true")
    policy = p.add_mutually_exclusive_group(required=True)
    policy.add_argument("--fixed", action="store_true")
    policy.add_argument("--natural", action="store_true")
    print(json.dumps(audit(p.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
