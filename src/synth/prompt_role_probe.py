"""Compare prompts using independently verified speech in an isolated context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from client import CHAT_SYS, CHAT_SYS_BASE, ServerClient, Turn
from backend_session import backend_session, canonical_base
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.read_aloud_rubrics import grade

ROLE_PROMPT = (CHAT_SYS_BASE + ' You are the assistant, and the user is a separate person.'
    ' When the user says I or my, those words refer to the user.'
    ' In your replies, refer to the user\'s facts with you or your.'
    ' Keep identities and preferences attributed to their original speaker.'
    ' Answer briefly and follow the user\'s requested format.')


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    capture_files(folder / 'source')
    origin = ROOT / 'outputs/experiments/synthetic-ad-v1'
    verification = {row['key']: row for row in map(json.loads, (origin/'independent-asr.jsonl').read_text().splitlines())}
    protocol = json.loads((origin / 'protocol.json').read_text())
    cases = [row for row in protocol['cases'] if row['id'] in ['B1', 'B2', 'B3', 'B8']]
    inputs = folder / 'inputs'; inputs.mkdir()
    checks = []
    for voice in protocol['voices']:
        for case in cases:
            key = voice['id']+'-'+case['id']; data = (origin/'inputs'/(key+'.wav')).read_bytes()
            check = verification[key]
            if not check['input_check_pass'] or hashlib.sha256(data).hexdigest() != check['sha256']:
                raise RuntimeError('Unverified or changed synthetic input: '+key)
            (inputs/(key+'.wav')).write_bytes(data); checks.append(check)
    (folder/'protocol.json').write_text(json.dumps(dict(cases=cases, voices=protocol['voices'],
        prompts={'baseline': CHAT_SYS, 'role': ROLE_PROMPT}, input_checks=checks,
        scope='Short B1/B2/B3/B8 sequence in two voices; not a full A-D or long-memory benchmark.'), indent=2))
    client = ServerClient('http://127.0.0.1:8081', timeout=120)
    if canonical_base(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080')) == client.base:
        raise RuntimeError('A separate CPU backend is required')
    rows = []
    with backend_session(client.base), (folder/'results.jsonl').open('x') as results:
        deadline = time.monotonic()+60
        while not client.health():
            if time.monotonic() > deadline: raise TimeoutError('CPU server did not become ready')
            time.sleep(.5)
        for label, system in [('baseline', CHAT_SYS), ('role', ROLE_PROMPT)]:
            for voice in protocol['voices']:
                for index, case in enumerate(cases):
                    key = voice['id']+'-'+case['id']; turn = Turn()
                    payload = client._payload(system, client._audio_input((inputs/(key+'.wav')).read_bytes()), index == 0, 192)
                    (folder/(label+'-'+key+'-request.json')).write_text(json.dumps(payload, indent=2))
                    with wave.open(str(folder/(label+'-'+key+'-reply.wav')), 'wb') as writer:
                        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24000)
                        def receive(audio): writer.writeframes((np.clip(audio, -1, 1)*32767).astype('<i2').tobytes())
                        client._stream(payload, turn, on_audio=receive)
                    rubric = grade(case['id'], turn.text)
                    row = dict(prompt=label, key=key, case=case['id'], text=turn.text, error=turn.error,
                               finish_reason=turn.finish_reason, audio_samples=turn.audio_samples,
                               audio_s=turn.audio_samples/24000, fixture_check_pass=True,
                               rubric=rubric, passed=bool(rubric['passed'] and not turn.error and turn.audio_samples and not turn.truncated))
                    rows.append(row); results.write(json.dumps(row)+'\n'); results.flush(); os.fsync(results.fileno())
                    print(json.dumps(row), flush=True)
    report = {label: dict(strict_passed=sum(r['passed'] for r in rows if r['prompt']==label and not r['rubric'].get('setup_only')),
                          strict_total=6, setup_responses_passed=sum(r['passed'] for r in rows if r['prompt']==label and r['rubric'].get('setup_only')))
              for label in ['baseline', 'role']}
    (folder/'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path); args = parser.parse_args()
    if args.folder.exists(): parser.error('Use a new experiment folder')
    try:
        run(args.folder)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists(): print('Verified backup:', checkpoint(args.folder), flush=True)
