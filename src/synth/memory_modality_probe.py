"""Isolate memory failures by crossing text/audio input and text/audio output.

Uses a separately launched CPU server, never the live app's default backend.
Every request and reply is retained; this is a diagnostic, not a passing gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from backend_session import backend_session, canonical_base
from client import CHAT_SYS, ServerClient, Turn
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder, server):
    folder.mkdir(parents=True, exist_ok=False)
    capture_files(folder / 'source')
    fixtures = ROOT / 'data/synth/quality'
    protocol = [('name', 'My name is Alex. Please remember my name.'),
                ('recall', 'What is my name?')]
    inputs = folder / 'inputs'
    inputs.mkdir()
    manifest = []
    for name, text in protocol:
        data = (fixtures / (name + '.wav')).read_bytes()
        (inputs / (name + '.wav')).write_bytes(data)
        manifest.append(dict(id=name, text=text, sha256=hashlib.sha256(data).hexdigest()))
    (folder / 'protocol.json').write_text(json.dumps(dict(server=server, inputs=manifest,
        modes=['text_to_text', 'text_to_speech', 'audio_to_text', 'audio_to_speech'],
        system=CHAT_SYS, max_tokens=192, scope='Independent CPU context; diagnostic outputs only, no hardware audio.'), indent=2))
    client = ServerClient(server, timeout=120)
    assert client.health(), 'Independent server is unavailable'
    errors = 0
    rows = []
    # Use test=False only for this explicitly isolated backend. The OS lock
    # still prevents two probes from sharing its conversation context.
    with backend_session(server, test=False), (folder / 'results.jsonl').open('x', encoding='utf-8') as results:
        for audio_input in (False, True):
            for audio_output in (False, True):
                mode = ('audio' if audio_input else 'text') + '_to_' + ('speech' if audio_output else 'text')
                for index, (name, text) in enumerate(protocol):
                    content = client._audio_input((inputs / (name + '.wav')).read_bytes()) if audio_input else text
                    payload = client._payload(CHAT_SYS, content, index == 0, 192, audio=audio_output)
                    (folder / (mode + '-' + name + '-request.json')).write_text(json.dumps(payload, indent=2))
                    turn = Turn()
                    chunks = []
                    begin = time.perf_counter()
                    client._stream(payload, turn, on_audio=chunks.append)
                    if chunks:
                        audio = np.concatenate(chunks)
                        with wave.open(str(folder / (mode + '-' + name + '-reply.wav')), 'wb') as writer:
                            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24000)
                            writer.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
                    row = dict(mode=mode, case=name, reset_context=index == 0, text=turn.text,
                               error=turn.error, first_audio_s=turn.ttft, first_event_s=turn.first_event,
                               audio_samples=turn.audio_samples, elapsed_s=round(time.perf_counter()-begin, 3),
                               name_present='alex' in turn.text.lower())
                    errors += int(bool(turn.error))
                    row['requested_modalities'] = payload['modalities']
                    row['unexpected_audio'] = not audio_output and bool(turn.audio_samples)
                    rows.append(row)
                    results.write(json.dumps(row) + '\n'); results.flush(); os.fsync(results.fileno())
                    print(json.dumps(row), flush=True)
    invalid = errors or any(row['unexpected_audio'] or not row['text'].strip() for row in rows)
    (folder / 'summary.json').write_text(json.dumps(dict(transport_errors=errors,
        valid_semantic_comparison=not invalid, requests=len(rows),
        unexpected_audio_requests=sum(row['unexpected_audio'] for row in rows),
        modes={mode:dict(valid=all(not row['error'] and row['text'].strip() and not row['unexpected_audio']
                                  for row in rows if row['mode']==mode))
               for mode in sorted({row['mode'] for row in rows})},
        note='Modes name requested modalities; output sample counts reveal unsupported or ignored requests. Name presence alone is not an ownership/recall pass.'), indent=2))
    return bool(invalid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8081')
    args = parser.parse_args()
    parsed = urlparse(args.server)
    if parsed.hostname not in ('localhost', '127.0.0.1') or parsed.port != 8081:
        parser.error('This probe requires the independent local CPU server on port 8081')
    if canonical_base(args.server) == canonical_base(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080')):
        parser.error('The probe backend must differ from the app backend')
    # Never write into or recopy a pre-existing experiment after a CLI error.
    if args.folder.exists():
        parser.error('Use a new experiment folder')
    try:
        errors = run(args.folder, args.server)
    except Exception:
        if args.folder.is_dir():
            import traceback
            (args.folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if args.folder.is_dir():
            print('Verified backup:', checkpoint(args.folder), flush=True)
    raise SystemExit(1 if errors else 0)
