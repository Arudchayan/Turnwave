"""Apply explicit, hash-bound contextual annotations to unresolved saved answers.

Annotations are reviewer judgements, not independent model measurements. They
cannot replace audio/transport evidence or alter automatically known outcomes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.checkpoint_experiment import checkpoint,digest


def adjudicate(report,label):
    if report['outcome']!='manual':raise ValueError('Only unresolved outcomes may be adjudicated')
    if label['run']!=report['run'] or label['fact']!=report['fact']:
        raise ValueError('Annotation identity differs from source')
    if hashlib.sha256(report['text'].encode()).hexdigest()!=label['text_sha256']:
        raise ValueError('Annotation answer text differs from source')
    if not label.get('reason'):raise ValueError('Annotation needs an explicit reason')
    report=dict(report)
    report['automatic']={k:report[k] for k in ['content_pass','attribution_pass','instructions_pass','outcome']}
    for key in ['content_pass','attribution_pass','instructions_pass']:
        value=label[key]
        if value is not None and type(value) is not bool:raise ValueError('Review dimensions must be boolean or null')
        report[key]=value
    report['semantic_review_required']=report['content_pass'] is None or report['attribution_pass'] is None
    gates=[report[k] for k in ['content_pass','attribution_pass','instructions_pass','transport_pass']]
    report['outcome']='fail' if False in gates else 'pass' if all(v is True for v in gates) else 'manual'
    report['adjudication_reason']=label['reason']
    return report


def run(origin,labels_path,folder):
    labels=json.loads(labels_path.read_text());summary=json.loads((origin/'summary.json').read_text())
    if folder.exists():raise ValueError('Use a new adjudication folder')
    prepared={}
    for arm,contract in labels['arms'].items():
        path=origin/(arm+'-review.jsonl')
        if digest(path)!=contract['review_sha256']:raise ValueError('Automatic review changed')
        if digest(origin/(arm+'-results.jsonl'))!=contract['results_sha256']:raise ValueError('Input rows changed')
        if digest(origin/(arm+'-protocol.json'))!=contract['protocol_sha256']:raise ValueError('Input context changed')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        by_id={r['run']:r for r in rows}
        if len(by_id)!=len(rows):raise ValueError('Duplicate automatic result IDs')
        ids=[r['run'] for r in contract['labels']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate annotation IDs')
        for annotation in contract['labels']:
            by_id[annotation['run']]=adjudicate(by_id[annotation['run']],annotation)
        prepared[arm]=list(by_id.values())
    if set(prepared)!=set(summary['arms']):raise ValueError('Both saved arms must be reviewed')
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,folder/'adjudication-driver.py');shutil.copy2(labels_path,folder/'annotations.json')
    report=dict(version=labels['version'],reviewer=labels['reviewer'],limitations=labels['limitations'],
        origin=str(origin.resolve()),annotations_sha256=digest(labels_path),arms={})
    for arm,rows in prepared.items():
        (folder/(arm+'-review.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report['arms'][arm]=dict(outcomes=dict(Counter(r['outcome'] for r in rows)),
            rows=len(rows),adjudicated_rows=sum('adjudication_reason' in r for r in rows),
            unresolved_semantic_dimensions=sum(r['semantic_review_required'] for r in rows),
            unresolved_outcomes=sum(r['outcome']=='manual' for r in rows),
            automatic_origin_sha256=digest(origin/(arm+'-review.jsonl')))
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('origin',type=Path);p.add_argument('labels',type=Path);p.add_argument('folder',type=Path);a=p.parse_args()
    if a.folder.exists():raise SystemExit('Use a new adjudication folder')
    try:run(a.origin,a.labels,a.folder)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
