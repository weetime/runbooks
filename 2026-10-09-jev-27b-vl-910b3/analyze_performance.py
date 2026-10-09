"""Recompute all points from raw calls; aggregate independent restart medians."""
import collections,datetime,hashlib,json,math,re,statistics,sys
from pathlib import Path
root=Path(__file__).resolve().parent
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else root/'runtime'
def epoch(s):return datetime.datetime.fromisoformat(s).timestamp()
def nearest(v,q):return sorted(v)[math.ceil(len(v)*q)-1]
def npu_parse(stdout):
    lines=stdout.splitlines();cards={}
    for i,line in enumerate(lines[:-1]):
        a=line.split('|')
        if len(a)<5 or not re.fullmatch(r'\s*\d+\s+910B3\s*',a[1]):continue
        nxt=lines[i+1].split('|')
        if len(nxt)<5:continue
        card=int(a[1].split()[0]);head=a[3].split();tail=nxt[3].split()
        cards[card]=dict(power_w=float(head[0]),temperature_c=float(head[1]),aicore_pct=float(tail[0]))
        hbm=re.findall(r'(\d+)\s*/\s*(\d+)',nxt[3])
        cards[card]['hbm_mb']=int(hbm[-1][0])
    return cards
def metric_total(path,name):
    values=[]
    for line in path.read_text().splitlines():
        if line.startswith(name+'{'):
            values.append(float(line.rsplit(' ',1)[1]))
    assert values, ('Missing engine metric',name,path)
    return sum(values)
points=[]
reference_sources=None;reference_runtime=None;reference_inputs=None;reference_pod=None;server_pids=set()
for rep in [1,2,3]:
    folder=runtime/f'perf-{rep}'
    assert json.loads((folder/'status.json').read_text())['status']=='performance-complete'
    assert json.loads((folder/'gate.json').read_text())['status']=='passed'
    server=json.loads((folder/'server-process.json').read_text())
    assert server['pid'] not in server_pids, 'Server process reused across supposed restarts'
    server_pids.add(server['pid'])
    api_pids=set(re.findall(r'APIServer pid=(\d+)',(folder/'server.log').read_text(errors='replace')))
    assert api_pids=={str(server['pid'])}, 'API startup PID does not match this restart launcher'
    sources=json.loads((folder/'source-fingerprints.json').read_text())
    environment=json.loads((folder/'runtime-fingerprint.json').read_text())
    inputs=json.loads((folder/'perf-inputs.json').read_text())
    pod=json.loads((runtime/f'perf-{rep}-pod-fingerprint.json').read_text())
    if reference_sources is None:
        reference_sources,reference_runtime,reference_inputs,reference_pod=sources,environment,inputs,pod
    else:
        assert sources==reference_sources, 'Source drift across independent restarts'
        assert environment==reference_runtime, 'Runtime/source implementation drift'
        assert inputs==reference_inputs, 'Input drift across independent restarts'
        assert pod==reference_pod, 'Pod/node/image/allocation/restart drift'
    for name,sha in sources.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha, ('Archived source mismatch',name)
    raw_monitor=[json.loads(l) for l in (folder/'npu-smi.jsonl').read_text().splitlines()]
    summaries=json.loads((folder/'perf-summary.json').read_text());assert len(summaries)==12
    for s in summaries:
        f=folder/s['raw_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==s['raw_sha256']
        rows=[json.loads(l) for l in f.read_text().splitlines()];assert len(rows)==s['requests'] and all(r['status']=='ok' for r in rows)
        lat=[r['elapsed_ms'] for r in rows]
        for r in rows:
            response=r['response'];assert response['num_model_requests']==1 and response['usage']['completion_tokens']==1 and not response.get('thinking')
        workload_inputs=inputs[s['workload']]
        allowed_hashes={hashlib.sha256(json.dumps(item['body'],ensure_ascii=False,sort_keys=True).encode()).hexdigest() for item in workload_inputs}
        assert all(row['request_sha256'] in allowed_hashes for row in rows)
        assert len(rows)>=128
        recomputed=dict(p50_ms=nearest(lat,.5),p90_ms=nearest(lat,.9),p95_ms=nearest(lat,.95),p99_ms=nearest(lat,.99) if len(lat)>=662 else None)
        for k,v in recomputed.items():assert s[k]==v,(k,v,s[k])
        assert abs(s['requests_per_s']-len(rows)/s['duration_s'])<1e-8
        start,end=epoch(s['started_utc']),epoch(s['finished_utc'])
        samples=[r for r in raw_monitor if 'utc' in r and start<=epoch(r['utc'])<=end and r.get('code')==0]
        parsed=[(r,npu_parse(r['stdout'])) for r in samples]
        assert parsed and all(set(range(8))<=set(c) for _,c in parsed)
        resources={}
        for card in range(8):
            values=[c[card] for _,c in parsed]
            resources[str(card)]={k:{'mean':statistics.mean(x[k] for x in values),'max':max(x[k] for x in values)} for k in ['aicore_pct','hbm_mb','power_w']}
        times=[epoch(r['utc']) for r in samples];gaps=[b-a for a,b in zip(times,times[1:])]
        cpu=[];rss=[]
        for r in samples:
            info=r.get('cpu',{});procs=info.get('processes',{})
            if procs:rss.append(sum(x['rss_pages'] for x in procs.values())*info['page_bytes']/1024**3)
        for a,b in zip(samples,samples[1:]):
            ca,cb=a.get('cpu',{}),b.get('cpu',{});pa,pb=ca.get('processes',{}),cb.get('processes',{})
            common=pa.keys()&pb.keys();dt=epoch(b['utc'])-epoch(a['utc'])
            if common and dt>0:
                ticks=sum(pb[k]['utime_ticks']+pb[k]['stime_ticks']-pa[k]['utime_ticks']-pa[k]['stime_ticks'] for k in common)
                cpu.append(ticks/cb['clock_ticks_per_s']/dt)
        label=f"{s['workload']}-c{s['concurrency']}"
        before,after=folder/(label+'-metrics-before.txt'),folder/(label+'-metrics-after.txt')
        count=metric_total(after,'vllm:time_to_first_token_seconds_count')-metric_total(before,'vllm:time_to_first_token_seconds_count')
        total=metric_total(after,'vllm:time_to_first_token_seconds_sum')-metric_total(before,'vllm:time_to_first_token_seconds_sum')
        assert count==len(rows), ('Engine TTFT counter window mismatch',label,count,len(rows))
        engine_ttft_mean_ms=1000*total/count
        points.append(dict(**s,restart=rep,engine_ttft_mean_ms=engine_ttft_mean_ms,input_tokens_per_s=sum(r['response']['usage']['prompt_tokens'] for r in rows)/s['duration_s'],
            resources=resources,monitor_samples=len(samples),monitor_max_gap_s=max(gaps) if gaps else None,
            cpu_cores_mean=statistics.mean(cpu) if cpu else None,cpu_cores_max=max(cpu) if cpu else None,rss_gib_max=max(rss) if rss else None))
aggregate=[]
for workload in ['text','vision']:
    for c in [1,4,8,16,32,64]:
        group=[s for s in points if s['workload']==workload and s['concurrency']==c];assert len(group)==3
        row=dict(workload=workload,concurrency=c,restarts=3,total_requests=sum(s['requests'] for s in group),errors=0)
        for key in ['requests_per_s','p50_ms','p90_ms','p95_ms','p99_ms','input_tokens_per_s','engine_ttft_mean_ms']:
            v=[s[key] for s in group if s[key] is not None]
            row[key+'_median']=statistics.median(v) if len(v)==3 else None
            row[key+'_min']=min(v) if v else None;row[key+'_max']=max(v) if v else None
        aggregate.append(row)
out=dict(points=points,aggregate=aggregate,independent_server_pids=sorted(server_pids),units={'throughput':'one typed judgment per request; 2 NPU cards','latency':'end-to-end nonstreaming HTTP ms','cpu':'sum of process-tree CPU tick deltas / wall time; RSS includes shared pages per process'},limitations=['Closed loop; not sustainable arrival-rate capacity or a production SLO guarantee.','Input tokens include image processing tokens; image processor cache is warm for repeated inputs.','Single completion token: TPOT is undefined, output token/s equals decision/s in this protocol.','Engine TTFT means use before/after Prometheus sum/count deltas, verified against exact timed request count; no per-request TTFT percentiles inferred from nonstreaming HTTP latency.','Do not directly compare these request/s to the previous bundled three-question native SDK runs.'])
(runtime/'performance-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(aggregate,indent=2))
