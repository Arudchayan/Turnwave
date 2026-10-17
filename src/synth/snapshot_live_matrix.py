"""Snapshot completed matrix turns without touching an active experiment.

Only request/reply files whose completed result row is already present are
included. The current in-flight WAV is deliberately excluded. This is durable
partial evidence, not a completed experiment or an automatic resume format.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def completed_files(rows):
    files=set()
    for row in rows:
        if not row.get('run'):continue
        key=row['run']
        if not re.fullmatch(r'[A-Za-z0-9-]+',key):raise ValueError('Unsafe matrix run ID')
        if row.get('recall') is True:pos,kind=1,'RECALL'
        elif row.get('case')=='SETUP':pos,kind=0,'SETUP'
        else:raise ValueError('Unknown completed matrix row')
        stem=f'{key}-turn{pos}-{kind}'
        files.update([stem+'-request.json','reply-audio/'+stem+'.wav'])
    return sorted(files)


def run(folder,origin):
    origin=origin.resolve();folder=folder.resolve()
    if folder==origin or origin in folder.parents:raise ValueError('Snapshot must be outside its source experiment')
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    data=(origin/'results.jsonl').read_bytes();lines=data.splitlines(keepends=True)
    partial=bool(lines and not lines[-1].endswith(b'\n'))
    if partial:lines=lines[:-1]
    rows=[json.loads(line) for line in lines]
    snapshot=folder/'partial-experiment';snapshot.mkdir()
    (snapshot/'results.jsonl').write_bytes(b''.join(lines))
    files=set(completed_files(rows))
    files.update(p.relative_to(origin).as_posix() for p in (origin/'inputs').rglob('*') if p.is_file())
    files.update(p.name for p in origin.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ['protocol.json','run-manifest.json','prompt-diff.json','server-process.json','launcher-process.json','local-asr-manifest.json','local-asr-summary.json']))
    manifest=[]
    for relative in sorted(files):
        source=origin/relative;content=source.read_bytes();target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        manifest.append(dict(path=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    report=dict(origin=str(origin),complete=False,completed_recall_rows=sum(r.get('recall') is True for r in rows),
        completed_setup_rows=sum(r.get('case')=='SETUP' for r in rows),partial_final_line_ignored=partial,
        original_results_sha256=hashlib.sha256(data).hexdigest(),files=manifest,
        note='Read-only partial snapshot. No live server logs or unfinished reply included; does not stop or resume the source runner.')
    (folder/'snapshot.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('origin',type=Path);a=p.parse_args()
    try:run(a.folder,a.origin)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
