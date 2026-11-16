#!/usr/bin/env python3
"""Sample server metrics and whole-node GPU telemetry at 1 Hz until interrupted.

Run telemetry on the server node outside inference containers. Before/after
counter snapshots remain separate from these in-flight gauge samples.
"""
import argparse
import json
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-url",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (a.output/"device.jsonl").open("x") as dev:
        while running:
            begin=time.monotonic();stamp=time.time_ns()
            record={"timestamp_ns":stamp}
            try:
                with urllib.request.urlopen(a.metrics_url,timeout=0.8) as r:
                    (a.output/f"metrics-{stamp}.prom").write_bytes(r.read())
            except Exception as e:
                record["metrics_error"]=type(e).__name__
            try:
                r=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw,temperature.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=0.8,check=True)
                record["whole_node_gpu_csv"]=r.stdout
            except Exception as e:
                record["device_error"]=type(e).__name__
            record["collection_seconds"]=time.monotonic()-begin
            dev.write(json.dumps(record)+"\n");dev.flush()
            time.sleep(max(0,1-(time.monotonic()-begin)))


if __name__ == "__main__":
    main()
