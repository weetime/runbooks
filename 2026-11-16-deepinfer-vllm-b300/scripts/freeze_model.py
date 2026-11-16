#!/usr/bin/env python3
"""Hash local checkpoint and verify every indexed weight shard exists.

No downloads or inference. The provenance of the supplied revision must be
verified separately; a SHA manifest proves byte identity, not model accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError("refuse manifest overwrite")
    root=a.model.resolve()
    index=json.loads((root/"model.safetensors.index.json").read_text())
    referenced=set(index["weight_map"].values())
    if not referenced:
        raise ValueError("empty weight index")
    files={"model.safetensors.index.json","config.json","tokenizer.json","tokenizer_config.json","encoding/encoding.py"}|referenced
    for folder in ["encoding","inference"]:
        if (root/folder).is_dir():
            files.update(str(x.relative_to(root)) for x in (root/folder).rglob("*") if x.is_file())
    result={"model_id":"deepseek-ai/DeepSeek-V4.1-Flash-1017","revision":a.revision,
            "status":"byte_inventory_not_inference_validation","indexed_shards":len(referenced),"files":{}}
    for name in sorted(files):
        f=root/name
        if Path(name).is_absolute() or ".." in Path(name).parts or not f.is_file():
            raise ValueError(f"missing or unsafe model member: {name}")
        h=hashlib.sha256()
        with f.open("rb") as stream:
            for b in iter(lambda:stream.read(8*1024*1024),b""):
                h.update(b)
        result["files"][name]={"sha256":h.hexdigest(),"bytes":f.stat().st_size}
    with a.output.open("x") as out:
        json.dump(result,out,indent=2);out.write("\n")
    print(json.dumps({"indexed_shards":len(referenced),"hashed_files":len(files),"status":result["status"]}))


if __name__ == "__main__":
    main()
