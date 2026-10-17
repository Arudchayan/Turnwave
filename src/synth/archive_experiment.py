"""Preserve experiment context and optionally watch the existing UI read-only.

Archives source, patch, reports, logs and UI transcript/metrics. Does not record
raw microphone audio, call inference, restart the app, reset or unmute anything.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request
ROOT=Path(__file__).resolve().parents[2]


def capture_files(destination):
    destination.mkdir(parents=True,exist_ok=True)
    files=[]
    for folder in ['src','tests','docs','patches','outputs/logs','outputs/maintenance']:
        parent=ROOT/folder
        if parent.exists():files.extend(p for p in parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    files.extend(p for p in (ROOT/'outputs').glob('*.json'))
    files.extend(p for p in ROOT.glob('requirements*.txt'))
    files.extend(ROOT/p for p in ['AGENTS.md','README.md','.gitattributes','.gitignore'] if (ROOT/p).is_file())
    manifest=[];read_errors=[]
    for source in sorted(set(files)):
        relative=source.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        # Snapshot a live file's bytes once. It may grow afterward; the final
        # snapshot captures later bytes without modifying the original file.
        try:
            data=source.read_bytes()
        except OSError as exc:
            read_errors.append(dict(path=relative.as_posix(),error=str(exc)))
            continue
        target.write_bytes(data)
        manifest.append(dict(path=relative.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (destination/'files.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (destination/'read-errors.json').write_text(json.dumps(read_errors,indent=2),encoding='utf-8')
    for label,arguments in [('working-tree.patch',['diff','--binary','HEAD']),('git-status.txt',['status','--short']),('git-head.txt',['rev-parse','HEAD'])]:
        result=subprocess.run(['git',*arguments],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
        (destination/label).write_bytes(result.stdout)


def watch(folder,minutes):
    seen=set();revision=None;deadline=time.monotonic()+minutes*60;last_error=None
    with (folder/'ui-history.jsonl').open('a',encoding='utf-8') as history, (folder/'events.jsonl').open('a',encoding='utf-8') as events:
        while time.monotonic()<deadline and not (folder/'STOP').exists():
            now=time.time()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/api/state',timeout=2) as response:state=json.load(response)
                signature=(state.get('transcript_revision'),state.get('mic_on'),state.get('status'),state.get('error'),state.get('interrupted_at'))
                if signature!=revision:
                    history.write(json.dumps(dict(observed_ts=now,state=state),ensure_ascii=True)+'\n');history.flush();revision=signature
                with urllib.request.urlopen('http://127.0.0.1:8000/api/diagnostics',timeout=2) as response:recent=json.load(response)
                for event in recent:
                    key=json.dumps(event,sort_keys=True)
                    if key not in seen:events.write(key+'\n');seen.add(key)
                events.flush();last_error=None
            except (OSError,ValueError) as exc:
                error=str(exc)
                if error!=last_error:
                    history.write(json.dumps(dict(observed_ts=now,observer_error=error))+'\n');history.flush();last_error=error
            time.sleep(.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='snapshot')
    parser.add_argument('--watch-minutes',type=float,default=0)
    parser.add_argument('--backup-root',type=Path,help='Default: home/LocalVoice-Experiment-Backups')
    args=parser.parse_args()
    if not 0<=args.watch_minutes<=120:parser.error('Watch duration must be 0-120 minutes')
    label=re.sub('[^a-zA-Z0-9_-]+','-',args.label).strip('-')[:64] or 'snapshot'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=ROOT/'outputs/experiments'/(stamp+'-'+label)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'experiment.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),label=label,watch_minutes=args.watch_minutes,script='docs/READ_ALOUD_TEST.md',raw_microphone_audio=False,note='Source is a worktree snapshot; running processes may predate some edits. Live file snapshots can end in a partial final line.'),indent=2),encoding='utf-8')
    capture_files(folder/'before');print('Experiment:',folder,flush=True)
    if args.watch_minutes:
        print('Read-only observation active. Create STOP in this folder to finish early.',flush=True)
        try:watch(folder,args.watch_minutes)
        finally:
            capture_files(folder/'after')
            (folder/'finished.json').write_text(json.dumps(dict(finished_utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')
    from checkpoint_experiment import checkpoint
    print('Verified backup:',checkpoint(folder,args.backup_root),flush=True)
    print('Saved:',folder,flush=True)


if __name__=='__main__':main()
