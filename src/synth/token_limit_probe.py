"""Isolated token-budget termination and continuation controls; no device audio."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import client as client_module
from backend_session import backend_session, canonical_base
from client import CHAT_SYS, ServerClient, Turn
from synth.checkpoint_experiment import checkpoint
from synth.archive_experiment import capture_files


def run(folder, server='http://127.0.0.1:8081'):
    folder.mkdir(parents=True, exist_ok=False)
    capture_files(folder / 'source')
    client = ServerClient(server, timeout=120)
    shared_backend = canonical_base(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080')) == client.base
    original = client_module.sse_events
    finish = []
    def observed(response, cancel=None):
        for event in original(response, cancel):
            for choice in event.get('choices', []):
                if choice.get('finish_reason'):
                    finish.append(choice['finish_reason'])
            yield event
    client_module.sse_events = observed
    rows = []
    with backend_session(client.base, test=shared_backend), (folder / 'results.jsonl').open('x') as stream:
        deadline = time.monotonic() + 60
        while not client.health():
            if time.monotonic() >= deadline:
                raise TimeoutError('The isolated runner did not become ready')
            time.sleep(.5)
        for budget in (1, 8, 32, 128):
            protocol = [('setup', True, 'The secret word is cedar. Remember it and then count from one to twenty.', budget),
                        ('recall', False, 'What is the secret word? Answer with only that word.', 128)]
            for label, first, text, limit in protocol:
                finish.clear()
                payload = client._payload(CHAT_SYS, text, first, limit)
                key = f'{budget}-{label}'
                (folder / (key + '-request.json')).write_text(json.dumps(payload, indent=2))
                turn = Turn(); chunks = []
                client._stream(payload, turn, on_audio=chunks.append)
                if chunks:
                    with wave.open(str(folder / (key + '-reply.wav')), 'wb') as writer:
                        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24000)
                        writer.writeframes((np.clip(np.concatenate(chunks), -1, 1)*32767).astype('<i2').tobytes())
                row = dict(budget=budget, case=label, text=turn.text, error=turn.error,
                           audio_samples=turn.audio_samples, finish_reasons=list(finish),
                           first_audio_s=turn.ttft, reset_context=first)
                rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
    transport_ok = all(not row['error'] for row in rows)
    limits_ok = all(row['finish_reasons'] == ['length'] for row in rows
                    if row['case'] == 'setup' and row['budget'] in (1, 8))
    recalls_ok = all(row['finish_reasons'] == ['stop'] and row['audio_samples'] and
                     row['text'].strip().lower().strip('.') == 'cedar'
                     for row in rows if row['case'] == 'recall')
    natural_stop_seen = any(row['case'] == 'setup' and row['finish_reasons'] == ['stop']
                            for row in rows)
    passed = transport_ok and limits_ok and recalls_ok and natural_stop_seen
    (folder / 'summary.json').write_text(json.dumps(dict(passed=bool(passed), cases=rows, server=server,
        checks=dict(transport=transport_ok, short_budget_length_reporting=limits_ok,
                    fact_recall=recalls_ok, natural_completion_control=natural_stop_seen),
        note='Termination reporting and cached-context continuation, not general memory or live-room quality.'), indent=2))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--server', choices=['http://127.0.0.1:8080', 'http://127.0.0.1:8081'],
                        default='http://127.0.0.1:8081')
    args = parser.parse_args()
    if args.folder.exists(): parser.error('Use a new experiment folder')
    try:
        passed = run(args.folder, args.server)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists(): print('Verified backup:', checkpoint(args.folder), flush=True)
    raise SystemExit(0 if passed else 1)
