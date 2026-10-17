"""Bounded regression loop; safe offline checks by default, no device audio.

Backend checks require the UI process to be stopped because the runner has one
shared conversation context. This command never restarts or unmutes the app.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def app_running():
    try:
        with socket.create_connection(('127.0.0.1',8000),timeout=.5):return True
    except OSError:return False


def run_check(folder, cycle, name, arguments, timeout, artifact=None):
    """Archive only an artifact actually updated by this invocation."""
    source = ROOT/'outputs'/artifact if artifact else None
    before = source.stat().st_mtime_ns if source is not None and source.exists() else None
    start = time.monotonic()
    try:
        result = subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        output, code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as exc:
        output, code = (exc.stdout or b'')+b'\nTIMED OUT\n', 124
    logfile = folder/f'{cycle+1}-{name}.log'
    logfile.write_bytes(output)
    row = dict(round=cycle+1,check=name,returncode=code,passed=code==0,
               elapsed_s=round(time.monotonic()-start,3),log=logfile.name)
    if source is not None:
        fresh = source.exists() and source.stat().st_mtime_ns != before
        row['artifact_state'] = 'updated' if fresh else 'missing_or_stale'
        if fresh:
            destination = folder/f'{cycle+1}-{artifact}'
            destination.write_bytes(source.read_bytes())
            row['artifact'] = destination.name
        else:
            row['passed'] = False
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--backend',action='store_true',help='Run shared GPU tests; stop the UI first')
    args=parser.parse_args()
    if not 1<=args.rounds<=10:parser.error('Use 1-10 bounded rounds')
    if args.backend and app_running():parser.error('Stop the UI before backend tests; its conversation context is shared.')
    folder=ROOT/'outputs/maintenance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    all_rows=[]
    for cycle in range(args.rounds):
        commands=[('diagnostics',['-u','src/synth/analyze_logs.py'],30,'log_summary.json'),
                  ('regression',['-m','pytest','-q','tests'],120,None),
                  ('acoustics',['-u','src/synth/acoustic_stress.py'],120,'acoustic_stress.json')]
        if args.backend:
            commands += [('conversation',['-u','src/synth/conversation_suite.py','--rounds','3'],180,'conversation_suite.json'),
                         ('application',['-u','src/synth/app_smoke.py'],90,'app_smoke.json'),
                         ('context',['-u','src/synth/context_soak.py','--turns','24'],180,'context_soak.json')]
        for name,arguments,timeout,artifact in commands:
            if args.backend and app_running():
                raise RuntimeError('UI started; refusing further backend tests')
            row = run_check(folder,cycle,name,arguments,timeout,artifact)
            all_rows.append(row)
            print(json.dumps(row),flush=True)
    quality_path=ROOT/'outputs/conversation_suite.json'
    known_quality=json.loads(quality_path.read_text(encoding='utf-8')) if quality_path.exists() else None
    digest=hashlib.sha256()
    for source in sorted((ROOT/'src').rglob('*.py')):
        digest.update(str(source.relative_to(ROOT)).encode());digest.update(source.read_bytes())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest.hexdigest(),
                scope='offline plus GPU' if args.backend else 'offline only; no GPU, microphone capture or speaker playback',
                checks=all_rows,checks_passed=all(row['passed'] for row in all_rows),
                last_semantic_suite_passed=known_quality.get('passed') if known_quality else None,
                product_ready=False,note='Passing this bounded run does not establish sustained real-device quality. Semantic evidence may be from an earlier run.')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    from checkpoint_experiment import checkpoint
    print('Verified backup:',checkpoint(folder),flush=True)
    print('Report:',folder/'report.json',flush=True)
    return 0 if report['checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
