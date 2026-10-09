"""Recompute frozen exact-match labels; never trust the runner's naive aliases."""
import base64, collections, hashlib, json, math, sys
from pathlib import Path
root=Path(__file__).resolve().parent
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else root/'runtime'
text=runtime/'attempt-07';vision=runtime/'vision-results-v3'
assert not (text/'INVALIDATED.json').exists(), 'Invalidated run'
assert json.loads((text/'status.json').read_text())['status']=='text-complete'
assert json.loads((text/'gate.json').read_text())['status']=='passed'
text_fixture=root/'data/text-requests.jsonl'
assert hashlib.sha256(text_fixture.read_bytes()).hexdigest()==json.loads((root/'data/text-manifest.json').read_text())['sha256']
expected=[json.loads(l) for l in text_fixture.read_text().splitlines()]
rows=[json.loads(l) for l in (text/'text-raw.jsonl').read_text().splitlines()]
assert len(rows)==len(expected)==2895
summary={};canonical=[]
for src,row in zip(expected,rows):
    for field in ['id','suite','question_id','body','gold','keys']:assert src[field]==row[field], field
    gold=str(row['gold'])
    if row['body']['kind']=='noul':gold={'yes':'true','no':'false'}.get(gold,gold)
    assert gold in row['keys'], (row['id'],gold,row['keys'])
    predicted=None;correct=False
    if row['status']=='ok':
        response=row['response'];prob=response['probabilities']
        assert len(prob)==len(row['keys']) and all(math.isfinite(x) and 0<=x<=1 for x in prob)
        assert abs(sum(prob)-1)<0.001 and not response.get('thinking')
        predicted=row['keys'][response['choice_index']];correct=predicted==gold
    else:assert row['status']=='error' and row.get('error')
    key=row['suite'];s=summary.setdefault(key,dict(decisions=0,correct=0,errors=0,adaptations={}))
    s['decisions']+=1;s['correct']+=correct;s['errors']+=row['status']!='ok'
    a=row['adaptation'];s['adaptations'][a]=s['adaptations'].get(a,0)+1
    canonical.append(dict(id=row['id'],suite=key,question_id=row['question_id'],gold=gold,predicted=predicted,correct=correct))
for s in summary.values():s['accuracy']=s['correct']/s['decisions']
# Retain original case-level all-judgments pass rate for bundled workloads.
case_scores={};question_scores={}
for row in canonical:
    key=(row['suite'],row['id']);case_scores.setdefault(key,[]).append(row['correct'])
    key=(row['suite'],row['question_id']);question_scores.setdefault(key,[]).append(row['correct'])
for suite,s in summary.items():
    grouped=[values for (name,_),values in case_scores.items() if name==suite]
    s['cases']=len(grouped);s['fully_correct_cases']=sum(all(values) for values in grouped)
    s['case_pass_rate']=s['fully_correct_cases']/len(grouped)
    s['per_question']={question:dict(decisions=len(v),correct=sum(v),accuracy=sum(v)/len(v))
        for (name,question),v in question_scores.items() if name==suite}
assert summary['typed-decisions']['cases']==400
assert all(len(v)==5 for (suite,_),v in case_scores.items() if suite=='typed-decisions')
vrows=[json.loads(l) for l in (vision/'raw.jsonl').read_text().splitlines()]
assert len(vrows)==256
fixtures=[]
for name,manifest in [('vision-requests.jsonl','vision-manifest.json'),('mind2web-requests.jsonl','mind2web-manifest.json')]:
    file=root/'data'/name
    declared=json.loads((root/'data'/manifest).read_text())
    assert hashlib.sha256(file.read_bytes()).hexdigest()==declared['sha256']
    fixtures.extend(json.loads(l) for l in file.read_text().splitlines())
expected_jobs=[(src,mode) for src in fixtures for mode in ['image','no-image']]
assert len(expected_jobs)==len(vrows)
seen=set();vs={}
for (src,mode),row in zip(expected_jobs,vrows):
    assert row['mode']==mode
    for k,v in src.items():
        if k!='body':assert row[k]==v,k
    image=(root/'data'/src['image']).read_bytes()
    assert hashlib.sha256(image).hexdigest()==src['image_sha256']
    body=dict(src['body']);state=body.get('state','Look at the supplied chart.')
    if mode=='image':
        mime='image/png' if src['image'].endswith('.png') else 'image/jpeg'
        body['state']=[state,{'image':f'data:{mime};base64,'+base64.b64encode(image).decode()}]
    else:body['state']=state
    expected_sha=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert row['body_sha256']==expected_sha
    key=(row['suite'],row['id'],mode);assert key not in seen;seen.add(key)
    correct=False
    if row['status']=='ok':
        response=row['response'];prob=response['probabilities']
        assert len(prob)==len(body['options']) and all(math.isfinite(x) and 0<=x<=1 for x in prob)
        assert abs(sum(prob)-1)<0.001 and not response.get('thinking')
        correct=response['choice_index']==src['gold_index']
    else:assert row['status']=='error' and row.get('error')
    s=vs.setdefault(row['suite']+'/'+mode,dict(decisions=0,correct=0,errors=0))
    s['decisions']+=1;s['correct']+=correct;s['errors']+=row['status']!='ok'
for s in vs.values():assert s['decisions']==64;s['accuracy']=s['correct']/64
mind=[src for src in fixtures if src['suite'].startswith('mind2web')]
vision_scope=dict(mind2web_steps=len(mind),mind2web_tasks=len({src['annotation_id'] for src in mind}),charxiv_questions=64)
strata={}
for row in vrows:
    if row['suite']!='charxiv-counts-adapted':continue
    key=row['group']+'/'+row['mode']
    s=strata.setdefault(key,dict(decisions=0,correct=0,errors=0))
    s['decisions']+=1;s['errors']+=row['status']!='ok'
    s['correct']+=row['status']=='ok' and row['response']['choice_index']==row['gold_index']
for s in strata.values():s['accuracy']=s['correct']/s['decisions']
out=dict(text=summary,vision=vs,vision_scope=vision_scope,charxiv_strata=strata,scoring='top1 exact-match with frozen noul yes/no aliases',limitations=['Adapted Score uses Choice; not native Score equivalence.','MASSIVE uses 20 labels beyond trained A-P head.','Visual subsets and oracle-included webpage choices are adapted; not official benchmark scores.','Replay latency is mixed-load capability data, not a performance benchmark.','Mind2Web: 64 steps from 14 tasks, oracle candidate inclusion, not official task/step success.','CharXiv: 32 numeric and 32 Not Applicable questions; report strata, not pure vision interpretation of pooled accuracy.','132 heterogeneous gate calls are repeats of 11 inputs, Choice/Noul only; no concurrent native Score truth validation.','Typed: synthetic teacher reference labels; all-five match is not executed workflow success.'],source_sha256={str(f.relative_to(runtime)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [text/'text-raw.jsonl',vision/'raw.jsonl']})
(runtime/'capability-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(runtime/'canonical-text-scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in canonical))
print(json.dumps(out,ensure_ascii=False,indent=2))
