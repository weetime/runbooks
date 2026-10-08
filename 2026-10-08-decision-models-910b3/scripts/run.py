def run_jevbench():
    """Frozen JevBench public requests, native distributions, no retry or test fitting."""
    import argparse,datetime,hashlib,importlib.metadata,json,os,random,sys,time,math
    from pathlib import Path
    R=Path(__file__).resolve().parent/'data/jevbench'
    sys.path.insert(0,str(R/'upstream'))
    from jevbench.tasks import Task
    from jevbench.adapters.base import build_question
    from jevbench.scoring import score_task
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--device',default='npu:0');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    sha=lambda b:hashlib.sha256(b).hexdigest()
    manifest=json.loads((R/'data/manifest.json').read_text());blob=(R/'data/cases.jsonl').read_bytes();assert sha(blob)==manifest['cases_sha256']
    cases=[json.loads(x) for x in blob.splitlines()]
    for c in cases:
     t=Task.from_dict(c['task']);assert c['request']=={'state':t.state,'questions':{'decision':build_question(t)}}
     assert sha(json.dumps(c['request'],ensure_ascii=False,separators=(',',':')).encode())==c['request_sha256']
    for item in manifest['source_files']:assert sha((R/item['path']).read_bytes())==item['sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    meta={'model':a.model,'device':a.device,'smoke_only':a.smoke,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases_sha256':sha(blob),'protocol_sha256':sha((R/'data/manifest.json').read_bytes()),'runner_sha256':sha(Path(__file__).read_bytes()),'official_commit':manifest['upstream_commit'],'warmups':[],'diagnostic_timing_only':True}
    agent=None;torch=None;secret='';client=None
    if a.model=='jev-1.13.0':
     import httpx
     secret=os.environ['TYPESAFE_API_KEY'].strip();assert secret
     client=httpx.Client(timeout=120,headers={'Authorization':'Bearer '+secret,'User-Agent':'JevBench/1.2 (+https://github.com/fstandhartinger/jevbench)'})
     meta['runtime']={'transport':'Mac HTTPS','model':a.model};meta['versions']={'httpx':importlib.metadata.version('httpx')}
    else:
     sys.path.insert(0,str(Path(__file__).resolve().parent))
     from loader import load
     infer,agent,runtime,torch=load(a.model,a.device);torch.set_num_threads(8)
     meta['runtime']={k:v for k,v in runtime.items() if k!='sdpa_mask_expansions'}
     meta['loader_sha256']=sha((Path(__file__).resolve().parent/'loader.py').read_bytes())
     meta['versions']={x:importlib.metadata.version(x) for x in ['torch','torch-npu','transformers','decider-ai','laya']}
     if a.model.startswith('laya'):meta['laya_config']={k:agent.cfg.get(k) for k in ['max_len','head_max_len','temperature','temperature_by_options']}
     devices=[0,1] if len(runtime.get('parameter_devices',[]))>1 else [int(a.device.split(':')[1])]
    def safe_dump(obj):
     s=json.dumps(obj,ensure_ascii=False,allow_nan=False)
     return s.replace(secret,'[REDACTED]') if secret else s
    def save():
     (out/'metadata.json').write_text(safe_dump(meta))
    def diagnostics(req):
     if not a.model.startswith('laya'):return {'truncation_checked':False}
     from laya.common import build_sequence,serialize_state,render_options
     question=req['questions']['decision'];q=agent._to_internal(question)
     seq,_=build_sequence(agent.tok,req['state'],q,agent.cfg['max_len'],agent.cfg['head_max_len'])
     empty,_=build_sequence(agent.tok,'',q,agent.cfg['max_len'],agent.cfg['head_max_len'])
     full=len(agent.tok(serialize_state(req['state']),add_special_tokens=False)['input_ids'])
     head=len(agent.tok('%s question: %s'%(q['t'],str(q['ins']).replace(agent.tok.mask_token,' ')),add_special_tokens=False)['input_ids'])
     retained=seq.index(agent.tok.sep_token_id)-1
     opts=[len(agent.tok(' '+o,add_special_tokens=False)['input_ids']) for o in render_options(q)]
     return {'truncation_checked':True,'state_tokens':full,'retained_state_tokens':len(seq)-len(empty),'instruction_tokens':head,'retained_instruction_tokens':retained,'option_tokens':opts,'state_truncated':len(seq)-len(empty)<full,'instruction_truncated':retained<head,'options_over_48':sum(n>48 for n in opts)}
    def call(c):
     req=c['request'];row={k:c[k] for k in ['id','tier','request_sha256']};row['family']=c['task']['family'];row['status_code']=None;row['infrastructure_error']=False
     try:
      if torch is not None:
       for d in devices:torch.npu.synchronize(d)
      started=time.perf_counter()
      if client is not None:
       response=client.post('https://api.typesafe.ai/v1/systemone',json={**req,'model':a.model});row['status_code']=response.status_code
       if response.status_code!=200:
        row['infrastructure_error']=response.status_code!=422
        raise RuntimeError('API HTTP '+str(response.status_code))
       obj=response.json();assert obj.get('model')==a.model,'API version mismatch'
      else:obj=infer(req['state'],req['questions'])
      if torch is not None:
       for d in devices:torch.npu.synchronize(d)
      row['elapsed_ms']=(time.perf_counter()-started)*1000
      row['response']=obj
      ans=obj['answers']['decision'];qtype=c['task']['question']['type'];assert ans['type']==qtype
      if qtype=='noul':
       val=ans['noul'];assert not isinstance(val,bool) and isinstance(val,(int,float)) and math.isfinite(val) and 0<=val<=1
       probs={'yes':val,'no':1-val}
      else:
       probs=ans['probabilities'];assert isinstance(probs,dict)
       if qtype=='choice':assert ans['choice'] in c['task']['labels']
      scored=score_task(probs,Task.from_dict(c['task']))
      row.update(status='ok',probs_as_returned=probs,scored=scored,diagnostics=diagnostics(req))
     except Exception as exc:
      row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:400])
      if client is not None and row['status_code'] is None:row['infrastructure_error']=True
     return row
    save()
    for tier in ['easy','standard','hard']:
     c=next(c for c in cases if c['tier']==tier);row=call(c);meta['warmups'].append(row);save()
     assert row['status']=='ok' and row['scored']['valid'], 'warmup failed: '+row.get('error',row.get('scored',{}).get('error','invalid'))
    if a.smoke:
     meta['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('SMOKE PASSED',a.model,flush=True);sys.exit(0)
    random.Random(13).shuffle(cases);infra_errors=0;completed=0
    with (out/'raw.jsonl').open('x') as f:
     for i,c in enumerate(cases,1):
      row=call(c);f.write(safe_dump(row)+'\n');f.flush();completed=i
      infra_errors=infra_errors+1 if row['infrastructure_error'] else 0
      if i%20==0:print('PROGRESS',a.model,i,'/231',flush=True);save()
      if row['status_code'] in (401,403,429) or infra_errors>=3:
       meta['stop_reason']='access/rate limit or3 consecutive infrastructure errors';break
    meta['completed_requests']=completed;meta['complete']=completed==len(cases);meta['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    if client is not None:client.close()
    print('FINISHED',a.model,completed,'complete',meta['complete'],flush=True)

import sys
if '--benchmark' in sys.argv:
    i=sys.argv.index('--benchmark')
    benchmark=sys.argv[i+1]
    del sys.argv[i:i+2]
    if benchmark != 'jevbench': raise ValueError('Unknown benchmark')
    run_jevbench()
    raise SystemExit()

"""Same frozen requests for local SDKs and Jev. No labels/factors enter model inputs."""
import argparse,json,time,datetime,hashlib,random,os,sys,traceback,importlib.metadata
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--device',default='npu:0');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
manifest=json.loads((R/'data/manifest.json').read_text());blob=(R/'data/cases.jsonl').read_bytes();assert hashlib.sha256(blob).hexdigest()==manifest['cases_sha256'];cases=[json.loads(x) for x in blob.splitlines()]
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
meta={'model':a.model,'device':a.device,'protocol_sha256':hashlib.sha256((R/'data/manifest.json').read_bytes()).hexdigest(),'cases_sha256':manifest['cases_sha256'],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'smoke_only':a.smoke,'warmups':[],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'concurrent_other_local_models_possible':True}
agent=None;torch=None
if a.model.startswith('jev'):
 import httpx
 client=httpx.Client(timeout=90,headers={'Authorization':'Bearer '+os.environ['TYPESAFE_API_KEY'].strip()})
 def infer(state,questions):
  response=client.post('https://api.typesafe.ai/v1/systemone',json={'state':state,'questions':questions,'model':a.model})
  if response.status_code!=200:raise RuntimeError('API HTTP status '+str(response.status_code))
  obj=response.json();return {k:obj[k] for k in ['model','answers','usage'] if k in obj}
 meta['runtime']={'client':'Mac HTTPS','pure_inference_timing':False}
else:
 from loader import load
 infer,agent,runtime,torch=load(a.model,a.device);torch.set_num_threads(8)
 meta['runtime']=runtime;meta['versions']={x:importlib.metadata.version(x) for x in ['torch','torch-npu','transformers','decider-ai','laya']}
 if not a.model.startswith('decider'):meta['laya_config']={k:agent.cfg.get(k) for k in ['max_len','head_max_len','temperature','temperature_by_options']}
def save():
 runtime=meta.get('runtime',{});exp=runtime.pop('sdpa_mask_expansions',None)
 if exp is not None:meta['sdpa_mask_expansion_count']=len(exp)
 (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
def validate(req,resp):
 answers=resp.get('answers',{});assert set(answers)==set(req['questions']), 'answer key mismatch'
 for qid,q in req['questions'].items():
  z=answers[qid]
  if q['type']=='noul':
   value=z.get('noul',z.get('probability_true'));assert isinstance(value,(int,float)) and 0<=value<=1,(qid,'invalid noul')
  else:
   keys=list(q['criteria']) if q['type']=='choice' else [str(i) for i in range(len(q['criteria']))]
   v=z.get('probabilities');values=[v[k] for k in keys] if isinstance(v,dict) else v
   assert isinstance(values,list) and len(values)==len(keys),(qid,'probability shape')
   assert all(isinstance(x,(int,float)) and 0<=x<=1 for x in values),(qid,'invalid probabilities')
   assert abs(sum(values)-1)<.02,(qid,'invalid probability sum')
def diagnostics(req):
 if a.model.startswith(('decider','jev')):return {'truncation_checked':False}
 from laya.common import build_sequence,serialize_state,render_options
 full=len(agent.tok(serialize_state(req['state']),add_special_tokens=False)['input_ids']);qs={}
 for qid,question in req['questions'].items():
  q=agent._to_internal(question);seq,_=build_sequence(agent.tok,req['state'],q,agent.cfg['max_len'],agent.cfg['head_max_len']);empty,_=build_sequence(agent.tok,'',q,agent.cfg['max_len'],agent.cfg['head_max_len'])
  head=len(agent.tok('%s question: %s'%(q['t'],str(q['ins']).replace(agent.tok.mask_token,' ')),add_special_tokens=False)['input_ids'])
  qs[qid]={'state_tokens':full,'retained_state_tokens':len(seq)-len(empty),'instruction_tokens':head,'retained_instruction_tokens':seq.index(agent.tok.sep_token_id)-1,'option_tokens':[len(agent.tok(' '+o,add_special_tokens=False)['input_ids']) for o in render_options(q)]}
 return {'truncation_checked':True,'questions':qs}
def call(c):
 req=c['request']
 if torch is not None:
  devices=[0,1] if len(meta['runtime'].get('parameter_devices',[]))>1 else [int(a.device.split(':')[1])]
  for d in devices:torch.npu.synchronize(d)
 t=time.perf_counter();response=infer(req['state'],req['questions'])
 if torch is not None:
  for d in devices:torch.npu.synchronize(d)
 ms=(time.perf_counter()-t)*1000;validate(req,response)
 return response,ms,diagnostics(req)
save()
try:
 for suite in manifest['suites']:
  c=next(c for c in cases if c['suite']==suite);resp,ms,diag=call(c)
  meta['warmups'].append({'id':c['id'],'suite':suite,'elapsed_ms':ms,'response':resp,'diagnostics':diag});save()
except Exception as exc:
 meta['warmup_failure']={'type':type(exc).__name__,'error':str(exc)};save();raise
if a.smoke:
 meta['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('SMOKE PASSED',a.model,flush=True);sys.exit(0)
random.Random(13).shuffle(cases)
with (out/'raw.jsonl').open('w') as f:
 for i,c in enumerate(cases):
  row={k:c[k] for k in ['id','suite','workflow','request_sha256']}
  try:
   resp,ms,diag=call(c);row.update(status='ok',response=resp,elapsed_ms=ms,diagnostics=diag)
  except Exception as e:
   row.update(status='error',error_type=type(e).__name__,error=str(e));print('ERROR',a.model,c['id'],type(e).__name__,str(e),flush=True)
  f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
  if (i+1)%25==0:print('PROGRESS',a.model,i+1,'/',len(cases),flush=True);save()
meta['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('FINISHED',a.model,flush=True)
