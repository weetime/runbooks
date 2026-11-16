#!/usr/bin/env python3
"""Package only the exact text sources/requests needed by the first paired run.

The full standard-v1 archive remains separate. This subset gets its own identity
and checksum list; it must never masquerade as the full 1,042-file catalog.
"""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--http',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    origin='a49ef30dab14921a7b1cc804c847f65f721cf44aa5d59eddc47561b868e2f602'
    if sha((a.catalog/'SHA256SUMS').read_bytes())!=origin:
        raise ValueError('wrong standard-v1 manifest identity')
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    http_manifest=json.loads((a.http/'manifest.json').read_text())
    source_files={}
    members={}
    for name,entry in manifest['files'].items():
        src=(a.catalog/entry['source']).read_bytes()
        if sha(src)!=entry['source_sha256']:
            raise ValueError('source identity mismatch')
        source_files[entry['source']]=src
        prepared=(a.prepared/name).read_bytes()
        if sha(prepared)!=entry['sha256']:
            raise ValueError('prepared identity mismatch')
        members['datasets/b300-paired-text-v1/'+name]=prepared
        http=(a.http/name).read_bytes()
        if sha(http)!=http_manifest['files'][name]['sha256']:
            raise ValueError('HTTP identity mismatch')
        members['datasets/b300-http-v1/'+name]=http
    sums=''.join(f'{sha(b)}  {n}\n' for n,b in sorted(source_files.items())).encode()
    source_files['SHA256SUMS']=sums
    source_files['CATALOG.json']=(json.dumps({'version':'b300-source-subset-v1',
        'upstream_version':'standard-v1','upstream_manifest_sha256':origin,
        'selected_source_files':len(source_files)-1,
        'selection':'exact source files referenced by b300-paired-text-v1 manifest; no CC/Agent full corpus',
        'subset_manifest_sha256':sha(sums)},indent=2)+'\n').encode()
    for name,b in source_files.items():
        members['datasets/b300-source-subset-v1/'+name]=b
    for prefix,folder in [('b300-paired-text-v1',a.prepared),('b300-http-v1',a.http)]:
        members[f'datasets/{prefix}/manifest.json']=(folder/'manifest.json').read_bytes()
    for name in ['config.json','tokenizer.json','tokenizer_config.json','encoding/encoding.py']:
        members['datasets/model-metadata/'+name]=(a.metadata/name).read_bytes()
    with a.output.open('xb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=6,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for name,b in sorted(members.items()):
                info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(b))
    result={'version':'b300-text-offline-v1','files':len(members),'bytes':a.output.stat().st_size,
        'sha256':sha(a.output.read_bytes()),'source_subset_manifest_sha256':sha(sums),
        'upstream_full_manifest_sha256':origin,'contains_full_standard_v1':False,
        'contains_weights':False,'contains_cc_agent_corpora':False}
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
