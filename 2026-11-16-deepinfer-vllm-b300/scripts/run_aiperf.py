#!/usr/bin/env python3
"""Run in the dedicated AIPerf client Pod after wire/protocol gates pass.

No credentials in argv. API authorization, if needed, must be injected through
the environment/secret mechanism supported by the frozen client version.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="deepseek-v4.1-flash-1017")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.count < 1 or a.concurrency < 1:
        p.error("count and concurrency must be positive")
    digest = hashlib.sha256(a.input.read_bytes()).hexdigest()
    rows = [json.loads(l) for l in a.input.open() if l.strip()]
    identities = {json.dumps(r["messages"],sort_keys=True,ensure_ascii=False) for r in rows}
    if a.count > len(rows) or len(identities) != len(rows):
        raise ValueError("insufficient or repeated input; refuse cycling")
    cmd = ["aiperf","profile","--model",a.model,"--url",a.url,"--endpoint-type","chat",
           "--streaming","--custom-dataset-type","raw_payload","--input-file",str(a.input),
           "--dataset-sampling-strategy","sequential","--concurrency",str(a.concurrency),
           "--request-count",str(a.count),"--tokenizer",a.tokenizer,"--use-server-token-count",
           "--export-outputs-json","--export-level","raw","--export-http-trace",
           "--output-artifact-dir",str(a.output/"aiperf")]
    if not a.execute:
        print(json.dumps({"command":cmd,"input_sha256":digest,"state":"dry_run"},indent=2))
        return
    gate = json.loads(a.gates.read_text())
    if gate.get("url") != a.url or digest not in gate.get("allowed_request_sha256",[]):
        raise ValueError("endpoint/request identity differs from reviewed gate evidence")
    for key in ["resource_authorized","recipe_approved","client_013_verified","G0","G1","G2_http","G3","budget_checked","sentinel_before_passed","metrics_sampler_running"]:
        if gate.get(key) is not True:
            raise ValueError(f"missing gate: {key}")
    if a.concurrency not in gate.get("allowed_concurrency",[]):
        raise ValueError("expansion not approved by previous point/stop-line review")
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"point.json").write_text(json.dumps({"status":"planned","url":a.url,
        "input_sha256":digest,"n":a.count,"concurrency":a.concurrency,
        "note":"review actual tokens, reasoning, sentinels, counters and errors before valid"},indent=2)+"\n")
    with (a.output/"client.log").open("x") as log:
        result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    point=json.loads((a.output/"point.json").read_text())
    point.update(status="partial" if result.returncode==0 else "failed",exit_code=result.returncode)
    (a.output/"point.json").write_text(json.dumps(point,indent=2)+"\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
