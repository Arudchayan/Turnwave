"""Build a separate, provenance-preserving projection of complete semantic pairs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from synth.checkpoint_experiment import checkpoint,digest
from synth.snapshot_live_matrix import completed_files
from synth.resume_semantic_matrix import read_rows,missing_runs


def complete_pairs(roster,original,continuation):
    # Validation accepts orphan SETUPs but never selects them.
    pending=missing_runs(roster,original)
    missing_runs(pending,continuation)
    original_done={r['run'] for r in original if r.get('recall')}
    continued_done={r['run'] for r in continuation if r.get('recall')}
    if original_done & continued_done or original_done | continued_done != {r['id'] for r in roster}:
        raise ValueError('Combined matrix is incomplete or has overlapping completed recalls')
    selected=[];origins={}
    for run in roster:
        label='original' if run['id'] in original_done else 'continuation'
        pair=[r for r in (original if label=='original' else continuation) if r.get('run')==run['id']]
        for row in pair:
            if any(row[k]!=run[k] for k in ('fact','fact_type','framing','voice','rep')):
                raise ValueError('Pair metadata differs from protocol')
        selected.extend(pair);origins[run['id']]=label
    return selected,origins


def run(original,continuation,folder):
    if folder.exists():raise ValueError('Use a new projection folder')
    if not (continuation/'completion.json').is_file() or not (continuation/'server-exit.json').is_file():
        raise ValueError('Continuation has not completed and stopped its server')
    contract=json.loads((continuation/'continuation.json').read_text())
    if digest(original/'results.jsonl')!=contract['original_results_sha256']:
        raise ValueError('Original results changed after continuation preparation')
    if (original/'protocol.json').read_bytes()!=(continuation/'protocol.json').read_bytes():
        raise ValueError('Protocol changed across restart')
    roster=json.loads((original/'protocol.json').read_text())['runs']
    before=read_rows(original/'results.jsonl');after=read_rows(continuation/'results.jsonl')
    selected,origins=complete_pairs(roster,before,after)
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,folder/'combine-driver.py')
    shutil.copy2(original/'protocol.json',folder/'protocol.json')
    shutil.copytree(original/'inputs',folder/'inputs')
    for label,origin in [('original',original),('continuation',continuation)]:
        pairs=[r for r in selected if origins[r['run']]==label]
        for relative in completed_files(pairs):
            dest=folder/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(origin/relative,dest)
        record=folder/'provenance'/label;record.mkdir(parents=True)
        for name in ['results.jsonl','run-manifest.json','server-process.json','server-exit.json','continuation.json','completion.json']:
            if (origin/name).is_file():shutil.copy2(origin/name,record/name)
    gates=[r for r in before if 'fixture_check_pass' in r]
    (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gates+selected),encoding='utf-8')
    manifest=dict(complete=True,recalls=len(roster),process_restart=True,deployed=False,
        scope='Complete matrix projection across two processes; original orphan SETUP excluded. Original rubric retained.',
        source_folders=dict(original=str(original.resolve()),continuation=str(continuation.resolve())),pair_origins=origins)
    (folder/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='pair_origins'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['original','continuation','folder']:p.add_argument(name,type=Path)
    a=p.parse_args()
    if a.folder.exists():raise SystemExit('Projection already exists; use a new folder.')
    try:run(a.original,a.continuation,a.folder)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
