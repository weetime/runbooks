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
