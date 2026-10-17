"""Continue missing two-turn semantic contexts in a new, independently saved run.

Original grades and prompts are retained. Previously completed fixture gates
are reused by hash, not misrepresented as new ASR measurements. Never mutates
the original experiment or carries over an interrupted model context.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from backend_session import backend_session
from client import ServerClient, Turn
from synth.checkpoint_experiment import checkpoint, digest
from synth.native_abort_integration import port_free, stop_owned
from synth.native_abort_probe import wsl_path
from synth.snapshot_live_matrix import run as snapshot_completed


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def missing_runs(runs, rows):
    known = {r['id'] for r in runs}
    recalls = [r['run'] for r in rows if r.get('recall')]
    if len(set(recalls)) != len(recalls) or set(recalls) - known:
        raise ValueError('Duplicate or unknown completed recall IDs')
    for run_id in recalls:
        pairs = [r for r in rows if r.get('run') == run_id]
        if len(pairs) != 2 or pairs[0].get('case') != 'SETUP' or not pairs[1].get('recall'):
            raise ValueError('Completed recall lacks a unique ordered setup: ' + run_id)
    return [r for r in runs if r['id'] not in set(recalls)]


def load_runner(path):
    # Executing definitions under a non-main name does not run the experiment.
    # Avoid importlib's cache writes inside the historical evidence directory.
    module = types.ModuleType('saved_semantic_runner')
    module.__file__ = str(path.resolve())
    exec(compile(path.read_bytes(), str(path.resolve()), 'exec'), module.__dict__)
    return module


def prepare(origin, folder):
    # Must be called only after the original process has stopped and been
    # checkpointed. An exclusive folder prevents accidental repeated execution.
    rows = read_rows(origin / 'results.jsonl')
    module = load_runner(origin / 'run_r1.py')
    pending = missing_runs(module.RUNS, rows)
    if not pending:
        raise ValueError('No missing recalls')
    protocol = json.loads((origin / 'protocol.json').read_text())
    if digest(ROOT / 'src/synth/read_aloud_rubrics.py') != protocol['grading_sha256']:
        raise ValueError('Original grading implementation has changed')
    expected = [{k: r[k] for k in ('id', 'fact', 'fact_type', 'framing', 'voice', 'rep')} for r in module.RUNS]
    if protocol['runs'] != expected:
        raise ValueError('Runner roster differs from original protocol')
    gates = {r['key']: r for r in rows if 'fixture_check_pass' in r}
    local = {r['key']: r for r in read_rows(origin / 'local-asr.jsonl')}
    for check in protocol['input_checks']:
        key = check['key']
        sha = digest(origin / 'inputs' / (key + '.wav'))
        if not (sha == check['sha256'] == local[key]['sha256'] == gates[key]['filing_sha']
                and local[key]['local_gate_pass'] and gates[key]['fixture_check_pass']):
            raise ValueError('Historical fixture gate/hash mismatch: ' + key)
    prompts = set()
    for row in rows:
        if row.get('case') == 'SETUP':
            payload = json.loads((origin / (row['run'] + '-turn0-SETUP-request.json')).read_text())
            prompts.add(payload['messages'][0]['content'])
            if payload['max_tokens'] != module.MAX_TOKENS or not payload['reset_context']:
                raise ValueError('Historical request configuration differs')
    if len(prompts) != 1:
        raise ValueError('Historical system prompt is inconsistent')
    folder.mkdir(parents=True, exist_ok=False)
    for name in ['run_r1.py', 'protocol.json', 'run-manifest.json', 'local-asr.jsonl']:
        shutil.copy2(origin / name, folder / name)
    shutil.copytree(origin / 'inputs', folder / 'inputs')
    shutil.copy2(__file__, folder / 'resume-driver.py')
    shutil.copytree(ROOT / 'src', folder / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    manifest = dict(origin=str(origin.resolve()), original_results_sha256=digest(origin / 'results.jsonl'),
                    original_runner_sha256=digest(origin / 'run_r1.py'), completed_recalls=len(module.RUNS)-len(pending),
                    pending_ids=[r['id'] for r in pending], system_prompt=next(iter(prompts)),
                    fixture_gates='Reused original local and server ASR evidence, SHA verified; not rerun.',
                    context='Each pending ID starts a fresh SETUP; orphaned original setup is excluded.',
                    process_restart=True, physical_audio=False, deployed=False)
    (folder / 'continuation.json').write_text(json.dumps(manifest, indent=2))
    (folder / 'historical-server-gates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gates.values()))
    return module, pending, manifest


def result_row(m, run, pos, key, turn):
    rubric = m.grade_setup(run['fact'], turn.text) if pos == 0 else m.grade_recall(run['fact'], run['recall_script_id'], turn.text)
    row = {k: run[k] for k in ('fact', 'fact_type', 'framing', 'voice', 'rep')}
    row.update(run=run['id'], key=key, case='SETUP' if pos == 0 else run['recall_script_id'],
               text=turn.text, error=turn.error, finish_reason=turn.finish_reason,
               first_audio_s=round(turn.ttft, 3) if turn.ttft is not None else None,
               audio_s=round(turn.audio_samples / 24000, 3), truncated=turn.truncated,
               transport_error=bool(turn.error or not turn.audio_samples or turn.truncated),
               rubric=rubric, recall=bool(pos),
               passed=bool(rubric['passed'] and not turn.error and turn.audio_samples > 0) if rubric['passed'] is not None else None)
    if pos:
        blob = m.normalize(turn.text)
        row.update(stance=m.stance(turn.text), has_claim=bool(re.search(m.CLAIM_RE[run['fact']], turn.text.lower())),
                   has_deflect=bool(re.search(m.DEFLECT_RE, turn.text.lower())),
                   stale_markers=[s for s in m.STALE_POOL if s in blob and s not in set(m.REQUIRED[run['fact']])])
    return row


def run(origin, folder):
    with backend_session('http://127.0.0.1:8081', test=True):
        if not port_free():
            raise RuntimeError('8081 occupied; leave its owner alone')
        m, pending, manifest = prepare(origin, folder)
        command = ['wsl.exe', '-e', 'python3', wsl_path(ROOT/'src/synth/serve_abort_variant_linux.py'),
                   m.PRIVATE, m.VERSION_NAME, wsl_path(folder)]
        (folder/'reply-audio').mkdir()
        with (folder/'server.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                client = ServerClient(m.ENDPOINT, timeout=120)
                deadline = time.monotonic()+120
                while not client.health():
                    if process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('CPU continuation server failed startup')
                    time.sleep(.5)
                actual = json.loads((folder/'server-process.json').read_text())
                prior = json.loads((origin/'server-process.json').read_text())
                if any(actual[k] != prior[k] for k in ('binary_sha256', 'models', 'command', 'cuda_visible_devices')):
                    raise ValueError('Server/model configuration differs from original')
                with (folder/'results.jsonl').open('x', encoding='utf-8') as results:
                    for case in pending:
                        for pos, key in enumerate([case['setup_input'], case['recall_input']]):
                            label = case['id']+f'-turn{pos}-'+('SETUP' if pos == 0 else 'RECALL')
                            payload = client._payload(manifest['system_prompt'], client._audio_input(
                                (folder/'inputs'/(key+'.wav')).read_bytes()), pos == 0, m.MAX_TOKENS)
                            (folder/(label+'-request.json')).write_text(json.dumps(payload, indent=2))
                            turn = Turn()
                            with wave.open(str(folder/'reply-audio'/(label+'.wav')), 'wb') as writer:
                                writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24000)
                                def receive(audio):
                                    writer.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
                                client._stream(payload, turn, receive)
                            if not turn.error and not turn.audio_samples:
                                turn.error = 'The reply reached its length limit before producing audio.' if turn.truncated else 'The server returned no audio. Please try again.'
                            row = result_row(m, case, pos, key, turn)
                            results.write(json.dumps(row)+'\n'); results.flush(); os.fsync(results.fileno())
                            print(json.dumps(row), flush=True)
                            if turn.error or turn.cancel.is_set() or turn.truncated:
                                raise RuntimeError('Turn failed; stop without recalling a broken setup: '+label)
                        # A completed pair is also durable on the second drive.
                        snapshot = folder.parent / (folder.name+'-durable-'+case['id'])
                        snapshot_completed(snapshot, folder)
                        # Include the continuation contract and reused gates.
                        for name in ['continuation.json', 'local-asr.jsonl', 'historical-server-gates.jsonl']:
                            shutil.copy2(folder/name, snapshot/'partial-experiment'/name)
                        print('Verified pair checkpoint:', checkpoint(snapshot), flush=True)
            finally:
                stop_owned(process, folder, m.PRIVATE+'/'+m.VERSION_NAME+'/llama-liquid-audio-server')
                (folder/'server-exit.json').write_text(json.dumps(dict(exit_code=process.poll())))
        (folder/'completion.json').write_text(json.dumps(dict(completed_ids=[r['id'] for r in pending],
            scope='Continuation subset only; combine with original completed pairs for full-matrix analysis.')))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('origin',type=Path); p.add_argument('folder',type=Path); a=p.parse_args()
    if a.folder.exists():
        raise SystemExit('Continuation destination already exists; preserve it and use a new folder.')
    try:
        run(a.origin.resolve(),a.folder.resolve())
    except Exception:
        if a.folder.exists(): (a.folder/'continuation-failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists(): print('Verified backup:',checkpoint(a.folder),flush=True)
