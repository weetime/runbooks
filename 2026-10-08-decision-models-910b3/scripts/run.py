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
