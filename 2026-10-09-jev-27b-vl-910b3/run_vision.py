"""Capability-only image/no-image ablation; timings are not performance benchmarks."""
import base64, concurrent.futures, hashlib, json, math, os, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/os.environ.get('JEV_VISION_OUT','vision-results-v2');OUT.mkdir(exist_ok=False)
records=[]
for name in ['vision-requests.jsonl','mind2web-requests.jsonl']:
    records += [json.loads(l) for l in (ROOT/name).read_text().splitlines()]
jobs=[(row,mode) for row in records for mode in ['image','no-image']]
def run(job):
    row,mode=job
    body=dict(row['body']);state=body.get('state','Look at the supplied chart.')
    if mode=='image':
        image=(ROOT/row['image']).read_bytes()
        assert hashlib.sha256(image).hexdigest()==row['image_sha256']
        mime='image/png' if row['image'].endswith('.png') else 'image/jpeg'
        body['state']=[state,{'image':f'data:{mime};base64,'+base64.b64encode(image).decode()}]
    else:
        body['state']=state
    t=time.perf_counter()
    result={k:v for k,v in row.items() if k!='body'}
    result.update(mode=mode,body_sha256=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
    try:
        req=urllib.request.Request('http://127.0.0.1:8127/v1/decide',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Connection':'close'})
        with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
        p=r['probabilities'];assert len(p)==len(body['options']) and all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<0.01
        assert r['protocol']=='jev27-bare-v1' and not r.get('thinking')
        result.update(status='ok',response=r,correct=r['choice_index']==row['gold_index'],elapsed_ms=(time.perf_counter()-t)*1000)
    except Exception as e:
        result.update(status='error',error=repr(e),elapsed_ms=(time.perf_counter()-t)*1000)
    return result
summary={}
with (OUT/'raw.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,r in enumerate(pool.map(run,jobs)):
        f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        k=r['suite']+'/'+r['mode'];s=summary.setdefault(k,dict(decisions=0,correct=0,errors=0));s['decisions']+=1;s['correct']+=r.get('correct',False);s['errors']+=r['status']!='ok'
        if (i+1)%32==0:print('VISION',i+1,'/',len(jobs),flush=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'limitations.json').write_text(json.dumps(dict(status='complete',timings='capability replay only; mixed concurrent load possible, not latency/capacity benchmark',protocol='adapted public subsets; not leaderboard reproduction'),indent=2))
print(summary)
