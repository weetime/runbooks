def analyze_jevbench():
    """Audit frozen JevBench artifacts and recompute scores; never calls models."""
    import argparse,collections,hashlib,json,math,sys
    from pathlib import Path
    R=Path(__file__).resolve().parent/'data/jevbench';sys.path.insert(0,str(R/'upstream'))
    from jevbench.tasks import Task
    from jevbench.scoring import score_task
    from jevbench.metrics import brier_score,ece_top_label,ordinal_mae
    sha=lambda b:hashlib.sha256(b).hexdigest()
    def stats(rows):
     valid=[x for x in rows if x['valid']];conf=[(max(x['probs'].values()),x['correct']) for x in valid]
     accepted=[x for x in valid if x['diagnostic_normalized_confidence']>=.8]
     incorrect=[x for x in valid if not x['correct']]
     return {'n':len(rows),'correct':sum(x['correct'] for x in rows),'accuracy':sum(x['correct'] for x in rows)/len(rows) if rows else None,'valid':len(valid),'invalid':len(rows)-len(valid),'brier_hard_label':sum(x['brier'] for x in valid)/len(valid) if valid else None,'ece_10':ece_top_label(conf,10)['ece'] if valid else None,'score_mae':ordinal_mae([(x['expected_level'],x['ordinal_ev']) for x in valid if 'ordinal_ev' in x]),'confidence_ge_08':{'accepted':len(accepted),'coverage':len(accepted)/len(rows) if rows else None,'correct':sum(x['correct'] for x in accepted),'wrong':sum(not x['correct'] for x in accepted),'accuracy':sum(x['correct'] for x in accepted)/len(accepted) if accepted else None},'wrong_mean_confidence':sum(x['diagnostic_normalized_confidence'] for x in incorrect)/len(incorrect) if incorrect else None,'state_truncated':sum(x.get('diagnostics',{}).get('state_truncated',False) for x in rows),'instruction_truncated':sum(x.get('diagnostics',{}).get('instruction_truncated',False) for x in rows),'options_over_48':sum(x.get('diagnostics',{}).get('options_over_48',0) for x in rows)}
    def main():
     p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
     manifest=json.loads((R/'data/manifest.json').read_text());blob=(R/'data/cases.jsonl').read_bytes();assert sha(blob)==manifest['cases_sha256'];assert sha((Path(__file__).resolve().parent/'run.py').read_bytes())==manifest['runner_sha256']
     for f,h in manifest['upstream_code_hashes'].items():assert sha((R/f).read_bytes())==h
     cases={c['id']:c for c in map(json.loads,blob.splitlines())};models=[]
     for name in manifest['models']:
      folder=R/'results'/name
      if not (folder/'metadata.json').exists() or not (folder/'raw.jsonl').exists():
       assert a.partial,name+' missing';continue
      meta=json.loads((folder/'metadata.json').read_text());assert not meta['smoke_only'];assert meta['model']==name;assert meta['cases_sha256']==manifest['cases_sha256'];assert meta['protocol_sha256']==sha((R/'data/manifest.json').read_bytes());assert meta['runner_sha256']==manifest['runner_sha256'];assert len(meta['warmups'])==3
      rows=[json.loads(s) for s in (folder/'raw.jsonl').read_text().splitlines()];assert len({x['id'] for x in rows})==len(rows)
      complete=meta.get('complete') and len(rows)==len(cases) and set(x['id'] for x in rows)==set(cases)
      if not a.partial:assert complete,name+' incomplete'
      decisions=[];failures=[]
      for row in rows:
       c=cases[row['id']];assert row['tier']==c['tier'];assert row['request_sha256']==c['request_sha256'];task=Task.from_dict(c['task'])
       z={'id':c['id'],'tier':c['tier'],'family':task.family,'type':task.question['type'],'expected':str(task.expected),'valid':False,'correct':False,'diagnostics':row.get('diagnostics',{})}
       if row['status']=='ok':
        if name=='jev-1.13.0':assert row['response']['model']==name
        ans=row['response']['answers']['decision'];assert ans['type']==task.question['type']
        if task.question['type']=='noul':
         val=ans['noul'];probs={'yes':val,'no':1-val}
        else:probs=ans['probabilities']
        assert row['probs_as_returned']==probs
        scored=score_task(probs,task);assert scored==row['scored'];z.update({k:v for k,v in scored.items() if k not in ('error',)})
        if scored['valid']:
         z['brier']=brier_score(scored['probs'],str(task.expected),task.labels)
         total=sum(scored['probs'].values());z['diagnostic_normalized_confidence']=max(scored['probs'].values())/total
         if task.question['type']=='score':z['expected_level']=task.expected
       else:failures.append({'id':c['id'],'tier':c['tier'],'error_type':row['error_type'],'error':row['error'],'status_code':row['status_code']})
       decisions.append(z)
      result={'model':name,'complete':bool(complete),'requests':len(rows),'successful_requests':sum(x['status']=='ok' for x in rows),'failed_requests':len(failures),'errors':failures,'overall':stats(decisions),'tiers':{t:stats([x for x in decisions if x['tier']==t]) for t in manifest['tiers']},'families':{t:stats([x for x in decisions if x['family']==t]) for t in sorted(set(x['family'] for x in decisions))},'types':{t:stats([x for x in decisions if x['type']==t]) for t in sorted(set(x['type'] for x in decisions))},'raw_sha256':sha((folder/'raw.jsonl').read_bytes()),'metadata_sha256':sha((folder/'metadata.json').read_bytes())}
      (folder/'quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(folder/'decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in decisions));models.append(result)
      print(name,'COMPLETE' if complete else 'PARTIAL',len(rows),{t:round(x['accuracy']*100,2) if x['n'] else None for t,x in result['tiers'].items()},'errors',len(failures))
     report={'protocol':manifest,'complete':len(models)==len(manifest['models']) and all(m['complete'] for m in models),'models':models,'calibration_note':'Brier hard onehot reference and ECE10 use official scored.probs, whose strict-valid mass is left unchanged. Coverage diagnostic alone normalizes any residual mass before applying fixed0.8 threshold. Not comparable with old soft-teacher Brier/ECE15.'}
     (R/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('ALL COMPLETE',report['complete'])
    if __name__=='__main__':main()

import sys
if '--benchmark' in sys.argv:
    i=sys.argv.index('--benchmark')
    benchmark=sys.argv[i+1]
    del sys.argv[i:i+2]
    if benchmark != 'jevbench': raise ValueError('Unknown benchmark')
    analyze_jevbench()
    raise SystemExit()

"""Independently audit raw outputs and derive quality metrics; never call a model."""
import argparse, collections, hashlib, json, math
from pathlib import Path

R = Path(__file__).resolve().parent
def sha(b): return hashlib.sha256(b).hexdigest()
def compact(o): return json.dumps(o, ensure_ascii=False, separators=(',', ':')).encode()
def mean(v): return sum(v) / len(v) if v else None

def metrics(rows):
    valid = [r for r in rows if r['valid']]
    ece = 0.
    for i in range(15):
        selected = [r for r in valid if min(14, int(r['confidence'] * 15)) == i]
        if selected:
            ece += len(selected) / len(valid) * abs(mean([r['confidence'] for r in selected]) - mean([r['correct'] for r in selected]))
    return {'decisions': len(rows), 'valid': len(valid), 'invalid': len(rows)-len(valid),
            'correct': sum(r.get('correct', False) for r in rows),
            'accuracy': sum(r.get('correct', False) for r in rows) / len(rows) if rows else None,
            'valid_accuracy': mean([r['correct'] for r in valid]),
            'brier': mean([r['brier'] for r in valid]), 'kl': mean([r['kl'] for r in valid]),
            'ece_15': ece if valid else None,
            'score_mae': mean([r['score_error'] for r in valid if 'score_error' in r])}

def probabilities(q, answer):
    if q['type'] == 'noul':
        value = answer.get('noul', answer.get('probability_true'))
        keys, values = ['false', 'true'], [1-value, value]
    else:
        keys = list(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]
        p = answer['probabilities']
        if isinstance(p, dict):
            assert set(p) == set(keys), 'probability key mismatch'
            values = [p[k] for k in keys]
        else: values = p
    assert len(keys) == len(values) and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values)
    total = sum(values)
    assert abs(total-1) < .02, 'probability mass'
    return keys, [v/total for v in values], total

def analyze(model, cases, manifest, partial):
    folder = R/'results'/model
    meta = json.loads((folder/'metadata.json').read_text())
    assert not meta.get('smoke_only')
    assert meta['cases_sha256'] == manifest['cases_sha256']
    assert meta['protocol_sha256'] == sha((R/'data/manifest.json').read_bytes())
    assert meta['runner_sha256'] == sha((R/'run.py').read_bytes()), 'runner changed'
    raw = [json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in raw}) == len(raw), 'duplicate case'
    complete = bool(meta.get('finished_at')) and len(raw) == len(cases)
    if not partial: assert complete, model+' unfinished'
    by_id = {c['id']: c for c in cases}
    decisions, errors, lengths = [], [], []
    for row in raw:
        c = by_id[row['id']]
        assert all(row[k] == c[k] for k in ['suite', 'workflow', 'request_sha256'])
        req = c['request']; assert sha(compact(req)) == row['request_sha256']
        if row['status'] == 'ok': assert set(row['response']['answers']) == set(req['questions'])
        else: errors.append({'id': row['id'], 'error': row.get('error'), 'suite': c['suite']})
        if row['status']=='ok' and model.startswith('jev'):
            assert row['response']['model']==model, 'API model version mismatch'
        for qid, q in req['questions'].items():
            d = {'id': row['id'], 'qid': qid, 'suite': c['suite'], 'workflow': c['workflow'], 'type': q['type'], 'valid': False}
            if row['status'] == 'ok':
                keys, p, mass = probabilities(q, row['response']['answers'][qid])
                g = c['gold'][qid]; gp = [g['probabilities'][k] for k in keys]
                assert abs(sum(gp)-1) < .0001
                gp = [v/sum(gp) for v in gp]
                label = keys[max(range(len(p)), key=lambda i: p[i])]
                d.update(valid=True, prediction=label, reference=g['label'], probabilities=dict(zip(keys,p)),
                         confidence=max(p), correct=label == g['label'], original_probability_mass=mass,
                         brier=sum((x-y)**2 for x,y in zip(p,gp)),
                         kl=sum(y*math.log(y/max(x,1e-12)) for x,y in zip(p,gp) if y>0))
                if q['type'] == 'score':
                    score = sum(i*v for i,v in enumerate(p))
                    d.update(expected_score=score, reference_score=g['score'], score_error=abs(score-g['score']))
                diag = row.get('diagnostics', {})
                if diag.get('truncation_checked'):
                    x=diag['questions'][qid]
                    lengths.append(dict(id=c['id'], qid=qid, suite=c['suite'], **x,
                                        state_truncated=x['retained_state_tokens']<x['state_tokens'],
                                        instruction_truncated=x['retained_instruction_tokens']<x['instruction_tokens'],
                                        options_over_48=any(v>48 for v in x['option_tokens'])))
            decisions.append(d)
    suites = {s: metrics([r for r in decisions if r['suite']==s]) for s in manifest['suites']}
    workflows = {w: metrics([r for r in decisions if r['suite']=='typed-decisions' and r['workflow']==w]) for w in manifest['workflow_counts']}
    types = {t: metrics([r for r in decisions if r['suite']=='typed-decisions' and r['type']==t]) for t in ['choice','noul','score']}
    lengths_by_suite = {}
    for s in manifest['suites']:
        selected=[r for r in lengths if r['suite']==s]
        if selected:
            lengths_by_suite[s]={'audited_decisions':len(selected), **{k:sum(r[k] for r in selected) for k in ['state_truncated','instruction_truncated','options_over_48']},
                                'max_state_tokens':max(r['state_tokens'] for r in selected)}
    summary={'model':model,'complete':complete,'requests':len(raw),'successful_requests':len(raw)-len(errors),
             'failed_requests':len(errors),'errors':errors,'suites':suites,'typed_workflows':workflows,'typed_types':types,
             'length_audit':lengths_by_suite,'raw_sha256':sha((folder/'raw.jsonl').read_bytes()),
             'runner_sha256':meta['runner_sha256'], 'metadata_sha256':sha((folder/'metadata.json').read_bytes())}
    (folder/'quality.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (folder/'decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in decisions))
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--partial',action='store_true'); a=p.parse_args()
    manifest=json.loads((R/'data/manifest.json').read_text())
    for item in manifest['files']:
        b=(R/'data'/item['path']).read_bytes();assert len(b)==item['bytes'] and sha(b)==item['sha256'],item['path']
    cases=[json.loads(x) for x in (R/'data/cases.jsonl').read_text().splitlines()]
    summaries=[]
    for model in manifest['models']:
        if not (R/'results'/model/'raw.jsonl').exists():
            assert a.partial, model+' missing';continue
        summary=analyze(model,cases,manifest,a.partial);summaries.append(summary)
        print(model,'complete' if summary['complete'] else 'PARTIAL',summary['requests'],
              {s:round(z['accuracy']*100,2) for s,z in summary['suites'].items()},'failures',summary['failed_requests'])
    out={'protocol':manifest,'models':summaries,'quality_note':'typed-decisions measures agreement with soft synthetic teacher references; Laya Typed trained on train split; Chinese probes are first300 subsets.',
         'probability_normalization':'normalize model mass and rounded reference mass to1 before distribution metrics; accuracy uses returned probabilities argmax',
         'timing_note':manifest['parallelism'],'complete':len(summaries)==len(manifest['models']) and all(s['complete'] for s in summaries)}
    (R/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
