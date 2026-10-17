"""Reproduce stale audio attachments after a failed request on isolated 8081."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from backend_session import backend_session, canonical_base
from client import ServerClient, Turn, CHAT_SYS
from synth.checkpoint_experiment import checkpoint


def main(folder, server='http://127.0.0.1:8081'):
    folder.mkdir(parents=True, exist_ok=False)
    client = ServerClient(server, timeout=120)
    app_backend = canonical_base(os.getenv('LOCALVOICE_SERVER', 'http://localhost:8080'))
    shared_backend = client.base == app_backend
    audio = client._audio_input((ROOT / 'data/synth/quality/name.wav').read_bytes())
    invalid = dict(type='input_audio', input_audio=dict(data=base64.b64encode(b'invalid WAV').decode(), format='wav'))
    rows = []
    # A probe against the app backend is allowed only with the UI stopped;
    # the same OS lease owns it through the entire malformed/recovery cycle.
    with backend_session(client.base, test=shared_backend), (folder / 'results.jsonl').open('x') as stream:
        for cycle in range(2):
            for case, content in [('healthy', 'Say only the word ready.'),
                                  ('invalid_second_attachment', audio + [invalid]),
                                  ('reset_text', 'Say only the word ready.'),
                                  ('reset_audio', audio)]:
                turn = Turn()
                payload = client._payload(CHAT_SYS, content, True, 64, audio=False)
                (folder / f'{cycle}-{case}-request.json').write_text(json.dumps(payload, indent=2))
                client._stream(payload, turn)
                expected_failure = case == 'invalid_second_attachment'
                passed = bool(turn.error) if expected_failure else bool(turn.text.strip()) and not turn.error
                row = dict(cycle=cycle, case=case, text=turn.text, error=turn.error,
                           expected_failure=expected_failure, passed=passed)
                rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
    passed = all(row['passed'] for row in rows)
    (folder / 'summary.json').write_text(json.dumps(dict(passed=passed, cases=rows, server=server,
        scope='Error rejection and fresh-context recovery only; does not grade reply semantics.'), indent=2))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--server', choices=['http://127.0.0.1:8080', 'http://127.0.0.1:8081'],
                        default='http://127.0.0.1:8081')
    args = parser.parse_args()
    if args.folder.exists(): parser.error('Use a new experiment folder')
    try:
        passed = main(args.folder, args.server)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists(): print('Verified backup:', checkpoint(args.folder), flush=True)
    raise SystemExit(0 if passed else 1)
