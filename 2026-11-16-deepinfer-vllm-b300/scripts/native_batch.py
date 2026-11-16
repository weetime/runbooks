#!/usr/bin/env python3
"""Candidate paired offline batch harness; optional native G3 chunk tracing.

The external recipe supplies all engine kwargs. DeepInfer requires DP1 for this
single-coordinator harness; EP8/CP8 is a documented candidate, pending approval
and real GPU validation. A batch size here is NOT closed-loop concurrency.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


class ChunkTrace:
    """Host-visible native output chunks, never HTTP packets or token-level ITL."""
    def __init__(self):
        self.events, self.counts = {}, {}

    def add(self, request_id, timestamp, count, cumulative=False):
        if cumulative:
            previous = self.counts.get(request_id, 0)
            if count < previous:
                raise ValueError("non-cumulative native output; G3 trace unsupported")
            self.counts[request_id] = count
            count -= previous
        if count:
            self.events.setdefault(request_id, []).append({"time": timestamp, "tokens": count})

    def summary(self, request_id, generated, started):
        events = self.events.get(request_id, [])
        if not events or sum(e["tokens"] for e in events) != generated:
            raise ValueError("native trace/token reconciliation failed")
        first, last = events[0], events[-1]
        denominator = generated - first["tokens"]
        if denominator <= 0 or last["time"] <= first["time"]:
            raise ValueError("no post-first-chunk interval; cannot pass G3")
        tpot = 1000 * (last["time"] - first["time"]) / denominator
        return {"boundary": "native host output availability, not HTTP", "chunks": len(events),
                "first_chunk_tokens": first["tokens"], "generated_tokens": generated,
                "ttft_ms": 1000 * (first["time"] - started),
                "chunk_boundary_tpot_ms": tpot, "decode_tokens_per_second": 1000 / tpot}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["deepinfer", "vllm"], required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--gates-record", type=Path, required=True)
    p.add_argument("--g3", action="store_true", help="24 distinct 8K/512 requests, B1, three groups of eight; native chunk timing")
    p.add_argument("--probe", action="store_true", help="minimal recipe-specific startup/text probe; requires G0, never marks G1/G2 passed")
    a = p.parse_args()
    if min(a.batch_size, a.limit) < 1 or a.start < 0:
        p.error("batch size / limit must be positive; start must be nonnegative")
    if a.g3 and (a.batch_size != 1 or a.limit != 24 or a.start != 0):
        p.error("G3 requires --batch-size 1 --limit 24 --start 0")
    if a.probe and (a.g3 or a.batch_size != 1 or a.limit > 2):
        p.error("probe is separate from G3 and limited to at most two serial requests")
    recipe_bytes = a.recipe.read_bytes()
    fingerprint = hashlib.sha256(recipe_bytes).hexdigest()
    recipe = json.loads(recipe_bytes)
    gates = json.loads(a.gates_record.read_text())
    if gates.get("recipe_sha256") != fingerprint or gates.get("engine") != a.engine:
        raise ValueError("gate evidence is for a different recipe/engine")
    required_gates = ("resource_authorized", "recipe_approved", "G0") if a.probe else ("resource_authorized", "recipe_approved", "G0", "G1", "G2")
    if not all(gates.get(k) is True for k in required_gates):
        raise ValueError("resource, recipe or correctness gate not passed")
    if recipe.get("engine") != a.engine:
        raise ValueError("recipe engine mismatch")
    request_sha = hashlib.sha256(a.requests.read_bytes()).hexdigest()
    if request_sha not in gates.get("allowed_request_sha256", []):
        raise ValueError("request set absent from gate evidence")
    for k, v in recipe.get("env", {}).items():
        os.environ[k] = str(v)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    kwargs = dict(recipe["engine_kwargs"])
    model = Path(kwargs["model"])
    if not model.is_absolute() or not (model / "config.json").is_file():
        raise ValueError("model must be an existing local checkpoint")
    rows = [json.loads(l) for l in a.requests.open() if l.strip()]
    rows = rows[a.start:a.start+a.limit]
    if len(rows) != a.limit or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("insufficient or duplicate source IDs; never loop samples")
    if a.g3 and any(not (7000 <= r["actual_isl"] <= 9000 and r["max_tokens"] == 512 and r["ignore_eos"]) for r in rows):
        raise ValueError("G3 requires the frozen approximate-8K/512 fixed-length shape")
    window = kwargs["max_model_len"]
    if any(r["actual_isl"] != len(r["prompt_token_ids"]) or r["actual_isl"] + r["max_tokens"] > window for r in rows):
        raise ValueError("token count/window gate failed")
    sentinel_row = None
    if not a.probe:
        sentinel_path = a.requests.parent / "warm-sharegpt.jsonl"
        if hashlib.sha256(sentinel_path.read_bytes()).hexdigest() not in gates.get("allowed_request_sha256", []):
            raise ValueError("quality sentinel set absent from gate evidence")
        sentinel_row = next(json.loads(line) for line in sentinel_path.read_text().splitlines() if line.strip())
        if (sentinel_row["ignore_eos"] or sentinel_row["actual_isl"] != len(sentinel_row["prompt_token_ids"])
                or sentinel_row["actual_isl"] + sentinel_row["max_tokens"] > window):
            raise ValueError("quality sentinel must be a valid, natural-EOS request")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    if not torch.cuda.is_available() or any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(torch.cuda.device_count())):
        raise ValueError("this candidate requires B300 SM103; no A100 execution")
    if torch.cuda.device_count() != 8:
        raise ValueError("recipe requires exactly eight isolated visible devices")
    if a.engine == "deepinfer":
        if kwargs.get("data_parallel_size") != 1 or kwargs.get("context_parallel_size") != 8:
            raise ValueError("native harness only implements the EP8/CP8/DP1 candidate")
        from deepinfer.entrypoints.llm import LLM
        from deepinfer.sampling_params import SamplingParams
    else:
        from vllm import LLM, SamplingParams
        if a.g3:
            from vllm.sampling_params import RequestOutputKind
            class NativeTraceLLM(LLM):
                # v0.32's OfflineInferenceMixin forces FINAL_ONLY. This client
                # adapter preserves its add_request path but asks for cumulative
                # native output. No engine/model implementation is patched.
                def _add_request(self, prompt, params, lora_request=None, priority=0):
                    params.output_kind = RequestOutputKind.CUMULATIVE
                    request_id = str(next(self.request_counter))
                    return self.llm_engine.add_request(request_id, prompt, params,
                        lora_request=lora_request, priority=priority)
            LLM = NativeTraceLLM
    llm = LLM(**kwargs)
    trace = ChunkTrace()
    depth = [0]
    if a.g3:
        def wrap(original):
            def step(*args, **kwargs):
                outer = depth[0] == 0
                depth[0] += 1
                try:
                    outputs = original(*args, **kwargs)
                    timestamp = time.perf_counter()  # before parsing content
                finally:
                    depth[0] -= 1
                if outer:
                    if a.engine == "deepinfer":
                        from deepinfer._C.engine import RequestOutput
                        grouped = {}
                        for ptr in outputs:
                            rid = RequestOutput.get_request_id(ptr)
                            if RequestOutput.get_token_id(ptr) >= 0:
                                grouped[rid] = grouped.get(rid, 0) + 1
                        for rid, n in grouped.items(): trace.add(rid, timestamp, n)
                    else:
                        for result in outputs:
                            if len(result.outputs) != 1:
                                raise ValueError("native G3 expects one completion")
                            trace.add(result.request_id, timestamp, len(result.outputs[0].token_ids), cumulative=True)
                return outputs
            return step
        llm.llm_engine.step = wrap(llm.llm_engine.step)
        if a.engine == "deepinfer":
            llm.llm_engine.flush_inflight_reqs = wrap(llm.llm_engine.flush_inflight_reqs)

    def generate(batch):
        common = [{"temperature": r["temperature"], "top_p": r["top_p"],
                   "max_tokens": r["max_tokens"], "ignore_eos": r["ignore_eos"]} for r in batch]
        params = [SamplingParams(**d, **({"seq_seed": r["seed"]} if a.engine == "deepinfer" else {"seed": r["seed"]})) for r, d in zip(batch, common)]
        ids = [r["prompt_token_ids"] for r in batch]
        if a.engine == "deepinfer":
            return llm.generate(prompt_token_ids=ids, sampling_params=params, use_tqdm=False)
        return llm.generate(prompts=[{"prompt_token_ids": x} for x in ids], sampling_params=params, use_tqdm=False)

    def quality_sentinel(stage):
        if a.probe:
            return
        result, = generate([sentinel_row])
        if a.engine == "deepinfer":
            ids = [int(item.token_id) for item in result]
            text = llm.get_tokenizer().decode(ids)
            finish = str(result[-1].finish_reason) if result else "missing"
        else:
            if len(result.outputs) != 1:
                raise ValueError("quality sentinel expects one completion")
            completion, = result.outputs
            ids, text, finish = list(completion.token_ids), completion.text, completion.finish_reason
        accounting_ok = (bool(ids) and all(t >= 0 for t in ids)
                         and len(ids) < sentinel_row["max_tokens"] and finish not in (None, "", "missing"))
        with (a.output / "quality-sentinels.jsonl").open("a") as record:
            record.write(json.dumps({"stage": stage, "id": sentinel_row["id"],
                "input_tokens": sentinel_row["actual_isl"], "max_tokens": sentinel_row["max_tokens"],
                "token_ids": ids, "text": text, "finish_reason": finish,
                "accounting_passed": accounting_ok, "quality": "requires_human_review"}, ensure_ascii=False)+"\n")
        if not accounting_ok:
            raise ValueError("quality sentinel empty/invalid/at budget; preserve evidence and stop")

    quality_sentinel("before")
    # Warmup must be a separately frozen set, not measured inputs. Gate evidence
    # records its hash; output and initial startup effects remain separate.
    warm = Path(gates["warmup_path"])
    if hashlib.sha256(warm.read_bytes()).hexdigest() != gates["warmup_sha256"]:
        raise ValueError("warmup identity mismatch")
    warm_rows = [json.loads(l) for l in warm.open() if l.strip()]
    if not warm_rows or any(r["actual_isl"] + r["max_tokens"] > window for r in warm_rows):
        raise ValueError("empty or oversized warmup")
    if not a.probe and {tuple(r["prompt_token_ids"]) for r in warm_rows} & {tuple(r["prompt_token_ids"]) for r in rows}:
        raise ValueError("warmup overlaps measured requests")
    if a.probe:
        (a.output / "warmup.json").write_text(json.dumps({"skipped":"startup probe; first request retained, not a performance point"})+"\n")
    else:
        t0 = time.perf_counter()
        first = generate(warm_rows)
        (a.output / "warmup.json").write_text(json.dumps({"seconds": time.perf_counter()-t0, "n":len(first), "discarded":True})+"\n")
    trace.events.clear(); trace.counts.clear()
    total_tokens, duration, native_timings = 0, 0.0, []
    with (a.output / "outputs.jsonl").open("x") as out, (a.output / "batches.jsonl").open("x") as timing:
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start:start+a.batch_size]
            torch.cuda.synchronize()
            started_unix = time.time()
            begin = time.perf_counter()
            outputs = generate(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            ended_unix = time.time()
            if len(outputs) != len(batch):
                raise ValueError("output count mismatch")
            tokens = 0
            for r, result in zip(batch, outputs):
                if a.engine == "deepinfer":
                    ids = [int(x.token_id) for x in result]
                    text = llm.get_tokenizer().decode(ids)
                    # Preserve the exposed engine reason; normalization and
                    # natural-completion claims still require correctness gates.
                    finish = str(result[-1].finish_reason) if result else "missing"
                    drafted = sum(int(x.draft_tok) for x in result)
                    accepted = sum(int(x.accept_tok) for x in result)
                    internal_id = result[0].request_id if result else "missing"
                else:
                    if len(result.outputs) != 1:
                        raise ValueError("expected exactly one completion")
                    completion = result.outputs[0]
                    ids, text, finish = list(completion.token_ids), completion.text, completion.finish_reason
                    drafted = accepted = None
                    internal_id = result.request_id
                if not ids or any(t < 0 for t in ids) or len(ids) > r["max_tokens"] or (r["ignore_eos"] and len(ids) != r["max_tokens"]):
                    raise ValueError("empty/invalid token count; preserve failed point")
                native = trace.summary(internal_id, len(ids), begin) if a.g3 else None
                if native: native_timings.append({"id": r["id"], **native})
                out.write(json.dumps({"id":r["id"], "input_tokens":r["actual_isl"], "token_ids":ids,
                    "text":text, "finish_reason":finish, "at_budget":len(ids)==r["max_tokens"],
                    "native_timing": native,
                    "raw_draft_tok_sum":drafted, "raw_accept_tok_sum":accepted}, ensure_ascii=False)+"\n")
                tokens += len(ids)
            timing.write(json.dumps({"ids":[r["id"] for r in batch], "n":len(batch), "seconds":elapsed,
                                     "started_unix":started_unix, "ended_unix":ended_unix,
                                     "actual_generated_tokens":tokens})+"\n")
            out.flush(); timing.flush()
            total_tokens += tokens
            duration += elapsed
    point = {"engine":a.engine, "recipe_sha256":fingerprint, "requests_sha256":request_sha,
             "image_manifest_ref":os.getenv("BENCH_IMAGE_MANIFEST_REF", "not_supplied"),
             "client_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "measurement_mode":"recipe_probe" if a.probe else ("instrumented_native_g3" if a.g3 else "native_batch"),
             "status":"partial", "reason":"review native outputs, startup logs, counters and applicable gates; HTTP metrics are not measured",
             "measurement":"offline synchronous generate wall time; excludes load, warmup and external output serialization",
             "batch_size":a.batch_size, "n":len(rows), "seconds":duration,
             "actual_generated_tokens":total_tokens, "output_tokens_per_second":total_tokens/duration,
             "output_tokens_per_second_per_gpu":total_tokens/duration/8,
             "ttft":"not_measured", "tpot":"not_measured", "goodput":"not_measured"}
    (a.output / "point.json").write_text(json.dumps(point, indent=2)+"\n")
    measured_chunks = {key: list(events) for key, events in trace.events.items()}
    quality_sentinel("after")
    if a.engine == "deepinfer":
        llm.close()
    if a.g3:
        groups = [statistics.median(x["decode_tokens_per_second"] for x in native_timings[i:i+8]) for i in (0, 8, 16)]
        median = statistics.median(groups)
        spread = (max(groups)-min(groups))/median
        reference = gates.get("g3_reference_decode_p50")
        floor_ok = median >= (0.8*reference if reference is not None else 15)
        passed = floor_ok and spread <= 0.10
        report = {"numeric_gate_passed":passed, "round_decode_p50_tok_s":groups,
                  "round_range_over_median":spread, "reference_decode_p50":reference,
                  "n":len(native_timings), "requests":native_timings,
                  "note":"Native host chunk timing, not HTTP or GPU kernel time. Review output quality and evidence before setting G3=true."}
        (a.output/"g3.json").write_text(json.dumps(report,indent=2)+"\n")
        (a.output/"native-chunks.json").write_text(json.dumps(trace.events)+"\n")
        if not passed:
            raise SystemExit("G3 numeric stop line reached; preserve evidence and do not run performance expansion")


if __name__ == "__main__":
    main()
