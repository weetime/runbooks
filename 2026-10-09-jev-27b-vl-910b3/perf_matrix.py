"""Closed-loop single-judgment matrix; independent process restarts are external."""
import base64, concurrent.futures, datetime, hashlib, io, json, math, time, urllib.request
from pathlib import Path
from PIL import Image

URL='http://127.0.0.1:8127'
def request(item):
    started=time.perf_counter()
    body=item['body']
    result=dict(id=item['id'],started=started,
                request_sha256=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
    try:
        req=urllib.request.Request(URL+'/v1/decide',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Connection':'close'})
        with urllib.request.urlopen(req,timeout=180) as r:data=json.load(r)
        p=data['probabilities'];assert len(p)==len(body['options']) and all(math.isfinite(v) and 0<=v<=1 for v in p) and abs(sum(p)-1)<0.01
        assert data['num_model_requests']==1 and data['usage']['completion_tokens']==1 and not data.get('thinking')
        result.update(status='ok',response=data)
    except Exception as e:
        result.update(status='error',error=repr(e))
    result.update(finished=time.perf_counter(),elapsed_ms=(time.perf_counter()-started)*1000)
    return result

def percentile(values,q):
    return sorted(values)[math.ceil(len(values)*q)-1]

def metrics(path):
    try:
        with urllib.request.urlopen(URL+'/metrics',timeout=10) as r:path.write_bytes(r.read())
    except Exception as e:
        path.write_text('FETCH_ERROR '+repr(e))

def run(root,out):
    root,out=Path(root),Path(out)
    text=[json.loads(l) for l in (root/'perf-text.jsonl').read_text().splitlines()]
    vision=[]
    for l in (root/'vision-requests.jsonl').read_text().splitlines()[:16]:
        row=json.loads(l);im=Image.open(root/row['image']).convert('RGB');im.thumbnail((448,448))
        b=io.BytesIO();im.save(b,format='PNG');body=dict(row['body'])
        body['state']=['Look at the supplied chart.',{'image':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()}]
        vision.append(dict(id=row['id'],body=body,image_size=im.size))
    inputs={'text':text,'vision':vision}
    (out/'perf-inputs.json').write_text(json.dumps(inputs,ensure_ascii=False))
    summaries=[]
    for workload,items in inputs.items():
        for item in items:
            r=request(item);assert r['status']=='ok',r
        for concurrency in [1,4,8,16,32,64]:
            label=f'{workload}-c{concurrency}';target=768 if concurrency==8 else 128
            metrics(out/(label+'-metrics-before.txt'))
            point_start=time.perf_counter();started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[];counter=0
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending={pool.submit(request,items[i%len(items)]) for i in range(concurrency)};counter=concurrency
                while pending:
                    done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        rows.append(future.result())
                        if counter<target or time.perf_counter()-point_start<5:
                            pending.add(pool.submit(request,items[counter%len(items)]));counter+=1
            elapsed=time.perf_counter()-point_start;finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
            raw=out/(label+'-raw.jsonl');raw.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            good=[r for r in rows if r['status']=='ok'];lat=[r['elapsed_ms'] for r in good]
            tokens=[r['response']['usage']['prompt_tokens'] for r in good]
            s=dict(workload=workload,concurrency=concurrency,requests=len(rows),successful_requests=len(good),errors=len(rows)-len(good),duration_s=elapsed,
                   requests_per_s=len(good)/elapsed,decisions_per_s=len(good)/elapsed,started_utc=started_utc,finished_utc=finished_utc,
                   p50_ms=percentile(lat,.5) if lat else None,p90_ms=percentile(lat,.9) if len(lat)>=128 else None,
                   p95_ms=percentile(lat,.95) if len(lat)>=128 else None,p99_ms=percentile(lat,.99) if len(lat)>=662 else None,
                   prompt_tokens_min=min(tokens) if tokens else None,prompt_tokens_max=max(tokens) if tokens else None,
                   raw_file=raw.name,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            summaries.append(s);(out/'perf-summary.json').write_text(json.dumps(summaries,indent=2))
            metrics(out/(label+'-metrics-after.txt'));print('PERF',label,json.dumps(s),flush=True)
            if s['errors']:
                raise RuntimeError('Invalid responses or failures: retain point and stop matrix')
    return summaries
