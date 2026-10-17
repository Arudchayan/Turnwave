"""Versioned supplemental review; preserves original grades, never infers truth from keywords.

Strict format measures the explicit 'Just ...' instruction on this fixed
protocol. Negation is a review flag, not an automatic contradiction judgement.
Non-directed factual truth still requires manual review against the setup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.read_aloud_rubrics import normalize
from synth.checkpoint_experiment import checkpoint

VERSION='supplemental-format-and-negation-1'
ANSWERS={'OWN-1':['Maya'],'OWN-2':['Alex'],'OWN-3':['green'],'OWN-4':['Coventry'],
         'INST-1':['seven','7'],'INST-2':['sunset garden'],
         'INST-3':['Friday at noon','Friday noon'],'INST-4':['coffee']}


def review(row, fact):
    directed=fact['directed'][row['framing']]
    if directed and row['fact'] not in ANSWERS:raise ValueError('No registered exact-format contract')
    accepted=ANSWERS.get(row['fact'],[]) if directed else []
    return dict(run=row['run'],fact=row['fact'],text=row['text'],historical_passed=row['passed'],
        directed=directed,accepted_format_forms=accepted,
        strict_format_pass=normalize(row['text']) in {normalize(s) for s in accepted} if directed else None,
        negation_review=bool(re.search(r"\b(?:not|never|no|cannot|can't|isn't|wasn't|doesn't|don't|didn't|won't)\b",row['text'],re.I)),
        note='Format is not factual truth. Negation flags require context review; historical grade unchanged.')


def run(folder,arms):
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,folder/'review-driver.py')
    shutil.copy2(ROOT/'src/synth/read_aloud_rubrics.py',folder/'normalization-source.py')
    summary=dict(version=VERSION,scope='Saved-text supplemental checks; no new inference, no historical rescore.',arms={})
    for index,origin in enumerate(arms):
        label=f'arm{index}';data=(origin/'results.jsonl').read_bytes()
        (folder/(label+'-results.jsonl')).write_bytes(data)
        protocol=json.loads((origin/'protocol.json').read_text())
        (folder/(label+'-protocol.json')).write_bytes((origin/'protocol.json').read_bytes())
        facts={f['id']:f for f in protocol['facts']}
        lines=data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
        rows=[json.loads(line) for line in lines];recalls=[r for r in rows if r.get('recall')]
        if len({r['run'] for r in recalls})!=len(recalls):raise ValueError('Duplicate recall IDs')
        reports=[review(r,facts[r['fact']]) for r in recalls]
        (folder/(label+'-review.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in reports))
        summary['arms'][label]=dict(origin=str(origin.resolve()),results_sha256=hashlib.sha256(data).hexdigest(),
            observed_recalls=len(reports),expected_recalls=len(protocol['runs']),
            complete=len(reports)==len(protocol['runs']),
            historical_passed=sum(r['historical_passed'] is True for r in reports),
            directed_observed=sum(r['directed'] for r in reports),
            strict_format_passed=sum(r['strict_format_pass'] is True for r in reports),
            historical_pass_but_format_fail=[r['run'] for r in reports if r['historical_passed'] is True and r['strict_format_pass'] is False],
            negation_review_ids=[r['run'] for r in reports if r['negation_review']])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('arms',type=Path,nargs='+');a=p.parse_args()
    if a.folder.exists():raise SystemExit('Use a new review folder')
    try:run(a.folder,a.arms)
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
