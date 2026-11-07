"""Compare local voice backends with a supplied synthetic 16 kHz WAV.

This is a completed-file diagnostic, not a microphone or intelligibility test.
Do not run it alongside the app: it acquires the backend's exclusive lease.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend_session import backend_session
from client import ServerClient, Turn
from lfm3_client import LFM3Client


def probe(client, wav, trials):
    rows = []
    for _ in range(trials):
        asr = Turn()
        start = time.perf_counter()
        try:
            transcript = client.asr_text(wav, turn=asr)
            asr_error = None
        except Exception as exc:
            transcript, asr_error = '', str(exc)
        asr_s = time.perf_counter() - start
        turn = Turn()
        start = time.perf_counter()
        first_audio = []
        def on_audio(_):
            if not first_audio:
                first_audio.append(time.perf_counter() - start)
        client.chat_turn(wav, turn, True, on_audio=on_audio)
        total = time.perf_counter() - start
        rows.append({'asr_text': transcript, 'asr_error': asr_error, 'asr_s': round(asr_s, 3),
                     'reply_text': turn.text, 'chat_error': turn.error,
                     'first_audio_s': round(first_audio[0], 3) if first_audio else None,
                     'total_s': round(total, 3), 'audio_samples': turn.audio_samples,
                     'finish_reason': turn.finish_reason})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('lfm2', 'lfm3'), required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--trials', type=int, default=3, choices=range(1, 11))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing result')
    wav = args.wav.read_bytes()
    client = (LFM3Client if args.backend == 'lfm3' else ServerClient)(args.url, timeout=90)
    try:
        with backend_session(args.url):
            if not client.health():
                raise SystemExit('Voice server is not ready')
            rows = probe(client, wav, args.trials)
    finally:
        if hasattr(client, 'close'):
            client.close()
    timings = [row['first_audio_s'] for row in rows if row['first_audio_s'] is not None]
    result = {'backend': args.backend, 'fixture_sha256': hashlib.sha256(wav).hexdigest(),
              'trials': rows, 'median_first_audio_s': statistics.median(timings) if timings else None,
              'scope': 'Completed synthetic WAV to decoded audio; excludes microphone, endpointing and playback.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    if any(row['asr_error'] or row['chat_error'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
