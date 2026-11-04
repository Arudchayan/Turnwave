"""Compare single-call and draft segmented TTS on fixed public text.

Uses a dedicated LFM3 backend and no audio device. Output timing covers local
synthesis only; it is not end-to-end latency or a listening-quality assessment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from app import Assistant
from backend_session import backend_session
from client import Turn
from lfm3_client import LFM3Client

TEXT = ('After work, take a few slow breaths and allow yourself a moment to settle. '
        'A short walk outside can help you change pace and enjoy some fresh air. '
        'Finally, choose a calming activity such as reading, stretching, or listening to music.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18086')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not callable(getattr(Assistant, '_speak_answer', None)):
        raise RuntimeError('This comparison requires the draft segmented synthesis implementation')
    rows = []
    with backend_session(args.url, test=True):
        voice = LFM3Client(args.url, timeout=180)
        assistant = object.__new__(Assistant)
        assistant.client = voice
        try:
            for mode in ('single', 'segmented', 'segmented', 'single'):
                turn = Turn()
                start = time.perf_counter()
                first = [None]
                samples = [0]
                def receive(audio):
                    if first[0] is None:
                        first[0] = time.perf_counter() - start
                    samples[0] += len(audio)
                if mode == 'single':
                    voice.tts(TEXT, turn, max_tokens=512, on_audio=receive)
                else:
                    assistant._speak_answer(TEXT, turn, receive)
                row = {'mode': mode,
                       'first_audio_s': round(first[0], 3) if first[0] is not None else None,
                       'total_s': round(time.perf_counter() - start, 3),
                       'audio_samples': samples[0], 'error': turn.error}
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            voice.close()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'scope': 'one fixed text, local synthesis, no playback',
                                  'order': 'ABBA', 'text': TEXT, 'rows': rows}, indent=2) + '\n', encoding='utf-8')
    if any(row['error'] or not row['audio_samples'] for row in rows):
        raise SystemExit('A synthesis request failed or returned no audio')


if __name__ == '__main__':
    main()
