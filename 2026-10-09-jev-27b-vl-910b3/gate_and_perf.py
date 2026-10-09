"""Isolated NPU launch, correctness gate, and complete frozen text replay."""
import base64, concurrent.futures, datetime, hashlib, io, json, math, os
import signal, subprocess, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('JEV_ATTEMPT', 'attempt-01')
OUT.mkdir(exist_ok=False)
BASE = 'http://127.0.0.1:8127'
stop = threading.Event()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def call(body):
    t = time.perf_counter()
    req = urllib.request.Request(BASE+'/v1/decide', data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={'Content-Type':'application/json','Connection':'close'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    p = data['probabilities']
    assert len(p)==len(data['options']) and len(p)>1
    assert all(math.isfinite(v) and 0<=v<=1 for v in p)
    assert abs(sum(p)-1)<0.01
    assert data['protocol']=='jev27-bare-v1'
    assert not data.get('thinking'), 'System 2 unexpectedly activated'
    return dict(response=data, elapsed_ms=(time.perf_counter()-t)*1000)

def cpu_snapshot(root_pid):
    all_procs={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            st=(proc/'stat').read_text().rsplit(')',1)[1].split()
            all_procs[int(proc.name)]=dict(ppid=int(st[1]),utime_ticks=int(st[11]),stime_ticks=int(st[12]),rss_pages=int(st[21]),threads=int(st[17]))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    selected={root_pid};changed=True
    while changed:
        before=len(selected)
        selected.update(pid for pid,st in all_procs.items() if st['ppid'] in selected)
        changed=len(selected)>before
    return dict(clock_ticks_per_s=os.sysconf('SC_CLK_TCK'),page_bytes=os.sysconf('SC_PAGE_SIZE'),processes={str(pid):all_procs[pid] for pid in selected if pid in all_procs})

def monitor():
    with (OUT/'npu-smi.jsonl').open('w') as f:
        while not stop.is_set():
            t=time.monotonic()
            try:
                r=subprocess.run(['npu-smi','info'], capture_output=True, text=True, timeout=10)
                row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout=r.stdout,stderr=r.stderr,code=r.returncode,cpu=cpu_snapshot(server.pid))
            except Exception as e:
                row=dict(error=str(e))
            f.write(json.dumps(row)+'\n');f.flush()
            stop.wait(max(0,1-(time.monotonic()-t)))

save('source-fingerprints.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
    [ROOT/'protocol.json',ROOT/'performance-protocol.json',ROOT/'perf_matrix.py',ROOT/'start_server.sh',ROOT/'serve_decide_ascend.py',Path(__file__)]})
import importlib.metadata, platform
save('runtime-fingerprint.json',dict(python=platform.python_version(),hostname=platform.node(),
 packages={n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu','transformers']},
 implementation_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [
  Path('/vllm-workspace/vllm/vllm/v1/worker/gpu_input_batch.py'),
  Path('/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py')]}))
log=(OUT/'server.log').open('wb',buffering=0)
server=subprocess.Popen(['bash',str(ROOT/'start_server.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
save('server-process.json',dict(pid=server.pid,pgid=server.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
mon=threading.Thread(target=monitor,daemon=True);mon.start()
try:
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        if server.poll() is not None:
            raise RuntimeError(f'Engine startup failed: exit {server.returncode}; see server.log')
        try:
            with urllib.request.urlopen(BASE+'/v1/decide/info',timeout=2) as r: info=json.load(r)
            break
        except Exception:
            time.sleep(2)
    else:
        raise TimeoutError('Engine startup exceeded 1200 seconds')
    save('decide-info.json',info)
    with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as r:models=json.load(r)
    save('models.json',models)
    assert any(x['id']=='jev-decision' for x in models['data']), 'Decision LoRA missing'
    # Readout gate: all requested tokens must be returned after masking,
    # with finite logprobs and no unrelated vocabulary entries.
    raw_probes=[]
    for ids in ([32,33,34], list(range(32,52)), [3721,1802], list(range(15,21))):
        body=dict(model='jev-decision',prompt='[kind] choice\n[state] Pick a suitable option.\n[question] Which option is most suitable?\n[options]\nA) a\nB) b\nC) c\n[decision]:',
                  max_tokens=1,temperature=1.0,top_p=1.0,top_k=0,logprobs=len(ids),allowed_token_ids=ids,
                  return_tokens_as_token_ids=True,add_special_tokens=False)
        req=urllib.request.Request(BASE+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response: raw=json.load(response)
        lp=raw['choices'][0]['logprobs']['top_logprobs'][0]
        returned={int(k.split(':')[1]):v for k,v in lp.items()}
        raw_probes.append(dict(request=body,response=raw))
        save('readout-completeness.json',raw_probes)
        assert set(returned)==set(ids), f'Missing or unrelated candidate logprobs: {returned}'
        assert all(math.isfinite(v) for v in returned.values())
        assert abs(sum(math.exp(v) for v in returned.values())-1)<0.001
    from PIL import Image
    smoke=[]
    for label,rgb in [('red',(230,20,20)),('blue',(20,20,230)),('green',(20,180,20)),('yellow',(230,230,20))]:
        b=io.BytesIO();Image.new('RGB',(256,256),rgb).save(b,format='PNG')
        image=b.getvalue();(OUT/f'probe-{label}.png').write_bytes(image)
        body=dict(kind='choice',state=[{'image':'data:image/png;base64,'+base64.b64encode(image).decode()}],
                  question='What is the dominant color in the image?',options=['red','blue','green','yellow'],thinking='off')
        row=dict(id=label,body=body,expected=label,**call(body))
        smoke.append(row);save('smoke-vision.json',smoke)
        assert row['response']['choice']==label, f'Image correctness gate failed: {label}'
    probes=[dict(kind='choice',state='The request failed with HTTP 500 and users cannot pay.',question='Which team handles this?',
                 options=['backend engineering','graphic design','human resources'],thinking='off'),
            dict(kind='noul',state='The customer was charged twice for one purchase.',question='Does the customer have a billing problem?',thinking='off')]
    reference=[call(b) for b in probes]
    assert reference[0]['response']['choice_index']==0 and reference[1]['response']['choice']=='true'
    save('smoke-text.json',reference)
    # Same inputs with concurrent submissions: verify distributions, not just a CLI flag.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        parity=list(pool.map(call,[probes[i%2] for i in range(16)]))
    delta=max(abs(a-b) for i,r in enumerate(parity) for a,b in zip(r['response']['probabilities'],reference[i%2]['response']['probabilities']))
    changes=sum(r['response']['choice_index']!=reference[i%2]['response']['choice_index'] for i,r in enumerate(parity))
    save('batch-parity.json',dict(max_probability_delta=delta,argmax_changes=changes,raw=parity))
    assert delta<=0.05 and changes==0, 'Batch parity gate failed'
    # Heterogeneous mixed-input reorder gate: distinct candidate sets,
    # varying prefill lengths, and image/text inputs share the same queue.
    mixed_bodies=[probes[0],probes[1]]+[x['body'] for x in smoke]
    for n,length in [(3,20),(5,800),(8,1800),(16,3000),(20,100)]:
        mixed_bodies.append(dict(kind='choice',state='Context. '+('neutral background. '*((length+19)//20))[:length],
            question='Which option describes the request?',options=['option '+str(i) for i in range(n)],thinking='off'))
    mixed_reference=[call(b) for b in mixed_bodies]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        mixed=list(pool.map(call,[mixed_bodies[i%len(mixed_bodies)] for i in range(132)]))
    mixed_delta=max(abs(a-b) for i,r in enumerate(mixed) for a,b in
        zip(r['response']['probabilities'],mixed_reference[i%len(mixed_bodies)]['response']['probabilities']))
    save('heterogeneous-parity.json',dict(max_probability_delta=mixed_delta,inputs=mixed_bodies,references=mixed_reference,raw=mixed))
    assert mixed_delta<=0.05, 'Heterogeneous batch probability gate failed'
    save('gate.json',dict(status='passed',vision_correct=4,text_correct=2,max_probability_delta=delta,
                         heterogeneous_calls=132,heterogeneous_probability_delta=mixed_delta))
    records=[]  # Capability replay already archived independently.
    def replay(record):
        try:
            row=dict(**record,**call(record['body']),status='ok')
            row['predicted']=record['keys'][row['response']['choice_index']]
            row['correct']=row['predicted']==str(record['gold'])
            return row
        except Exception as e:
            return dict(**record,status='error',error=repr(e))
    results=[]
    with (OUT/'text-raw.jsonl').open('w') as f, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(replay,records):
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();results.append(row)
            if len(results)%64==0:print('TEXT',len(results),'/',len(records),flush=True)
    suites={}
    for row in results:
        s=suites.setdefault(row['suite'],dict(decisions=0,errors=0,correct=0))
        s['decisions']+=1;s['errors']+=row['status']!='ok';s['correct']+=row.get('correct',False)
    save('text-summary.json',suites)
    from perf_matrix import run as run_perf
    points=run_perf(ROOT,OUT)
    save('status.json',dict(status='performance-complete',points=len(points)))
except Exception as e:
    save('status.json',dict(status='failed',error=repr(e)))
    raise
finally:
    if server.poll() is None:
        assert os.getpgid(server.pid)==server.pid
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
    stop.set();mon.join(timeout=12);log.close()
