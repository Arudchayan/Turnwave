"""Read-only paired audit of semantic matrices, including live partial runs.

Preserves exact input snapshots in a new experiment. Repeated deterministic
replications are not independent evidence: row-level McNemar is descriptive;
the complete-run fact-cluster sign-flip calculation is exploratory sensitivity.
Never starts, resets or stops any model process or writes into its inputs.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def recalls(data):
    lines=data.splitlines(keepends=True)
    partial=bool(lines and not lines[-1].endswith(b'\n'))
    if partial:lines=lines[:-1]
    rows={}
    for line in lines:
        item=json.loads(line)
        if not item.get('recall'):continue
        key=item['run']
        if key in rows:raise ValueError('Duplicate recall run: '+key)
        if not isinstance(item.get('passed'),bool):raise ValueError('Recall grade is not boolean: '+key)
        for field in ['fact','voice','framing','rep']:item[field]
        rows[key]=item
    return rows,partial


def mcnemar(gains,losses):
    n=gains+losses
    return min(1.,2*sum(math.comb(n,i) for i in range(min(gains,losses)+1))/2**n) if n else 1.


def validate_protocols(base,candidate,base_protocol,candidate_protocol):
    rosters=[]
    for protocol in [base_protocol,candidate_protocol]:
        roster={item['id']:item for item in protocol['runs']}
        if len(roster)!=len(protocol['runs']):raise ValueError('Duplicate protocol run ID')
        rosters.append(roster)
    if rosters[0]!=rosters[1]:raise ValueError('Protocol matrix conditions differ')
    if set(base)!=set(rosters[0]):raise ValueError('Baseline matrix is incomplete')
    for rows in [base,candidate]:
        for key,item in rows.items():
            if key not in rosters[0] or any(item[f]!=rosters[0][key][f] for f in ['fact','voice','framing','rep']):
                raise ValueError('Result does not match protocol: '+key)
    for field in ['grading_sha256','rubric_version']:
        if base_protocol[field]!=candidate_protocol[field]:raise ValueError('Grading contracts differ')


def compare(base,candidate):
    unexpected=sorted(set(candidate)-set(base))
    if unexpected:raise ValueError('Unexpected recall IDs: '+str(unexpected))
    ids=sorted(set(base)&set(candidate))
    for key in ids:
        if any(base[key][field]!=candidate[key][field] for field in ['fact','voice','framing','rep']):
            raise ValueError('Mismatched paired condition: '+key)
    complete=bool(base) and set(base)==set(candidate)
    gains=sum(not base[k]['passed'] and candidate[k]['passed'] for k in ids)
    losses=sum(base[k]['passed'] and not candidate[k]['passed'] for k in ids)
    report=dict(expected_recall_turns=len(base),observed_recall_turns=len(candidate),complete=complete,
        missing_ids=sorted(set(base)-set(candidate)),paired_observed=len(ids),
        baseline_pass_on_observed=sum(base[k]['passed'] for k in ids),candidate_pass_on_observed=sum(candidate[k]['passed'] for k in ids),
        gains=gains,losses=losses,transport_errors=sum(bool(candidate[k].get('transport_error')) for k in ids),
        interpretation='Complete paired descriptive comparison' if complete else 'INCOMPLETE: observed subset only; no final efficacy conclusion')
    groups=defaultdict(list)
    for key in ids:groups[(base[key]['fact'],base[key]['voice'],base[key]['framing'])].append(key)
    repeated=[keys for keys in groups.values() if len(keys)>1]
    report['replication_groups']=dict(groups=len(groups),
        baseline_consistent_multiple_reps=sum(len({base[k]['passed'] for k in keys})==1 for keys in repeated),
        candidate_consistent_multiple_reps=sum(len({candidate[k]['passed'] for k in keys})==1 for keys in repeated),
        groups_with_multiple_reps=len(repeated))
    report['per_voice']={v:dict(paired=sum(base[k]['voice']==v for k in ids),
        baseline_pass=sum(base[k]['passed'] for k in ids if base[k]['voice']==v),
        candidate_pass=sum(candidate[k]['passed'] for k in ids if base[k]['voice']==v)) for v in sorted({base[k]['voice'] for k in ids})}
    if complete:
        report['row_mcnemar_descriptive']=dict(exact_p=mcnemar(gains,losses),note='Repeated rows share facts/prompts; this is not independent-replication proof.')
        facts=defaultdict(list)
        for key in ids:facts[base[key]['fact']].append(key)
        deltas={fact:Fraction(sum(int(candidate[k]['passed'])-int(base[k]['passed']) for k in keys),len(keys)) for fact,keys in facts.items()}
        nonzero=[d for d in deltas.values() if d]
        if len(nonzero)>20:raise ValueError('Too many nonzero fact clusters for a bounded exact audit')
        observed=abs(sum(nonzero));denominator=2**len(nonzero)
        extreme=sum(abs(sum(sign*d for sign,d in zip(signs,nonzero)))>=observed for signs in itertools.product([-1,1],repeat=len(nonzero)))
        report['fact_cluster_sensitivity']=dict(facts=len(facts),nonzero_facts=len(nonzero),
            mean_delta_pp=float(sum(deltas.values())/len(deltas)*100),exact_sign_flip_p=extreme/denominator,
            deltas_pp={k:float(v*100) for k,v in deltas.items()},
            note='Exploratory paired fact-level sign-flip sensitivity. Assumes exchangeable signs across fact clusters; fixed selected facts do not establish benchmark/generalization performance.')
    return report


def run(folder,baseline,candidate):
    if folder.resolve() in [baseline.resolve(),candidate.resolve()]:raise ValueError('Use a separate audit folder')
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    snapshots={}
    for label,origin in [('baseline',baseline),('candidate',candidate)]:
        data=(origin/'results.jsonl').read_bytes();(folder/(label+'-results.jsonl')).write_bytes(data)
        rows,partial=recalls(data);snapshots[label]=dict(rows=rows,partial=partial)
        (folder/(label+'-provenance.json')).write_text(json.dumps(dict(origin=str(origin.resolve()),
            sha256=hashlib.sha256(data).hexdigest(),partial_final_line_ignored=partial),indent=2))
        for name in ['protocol.json','run-manifest.json']:
            if (origin/name).exists():(folder/(label+'-'+name)).write_bytes((origin/name).read_bytes())
    validate_protocols(snapshots['baseline']['rows'],snapshots['candidate']['rows'],
        json.loads((folder/'baseline-protocol.json').read_text()),json.loads((folder/'candidate-protocol.json').read_text()))
    report=compare(snapshots['baseline']['rows'],snapshots['candidate']['rows'])
    report['deployed']=False;report['scope']='Offline audit of saved text grades only, not voice quality or live backend validation.'
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
    try:run(a.folder,a.baseline,a.candidate)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
