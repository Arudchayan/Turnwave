"""Compare a contextual follow-up with and without an LFM3 session reset.

Run with the app stopped and an LFM3 backend running. Synthetic TTS is checked
with the same model's ASR; results are a narrow diagnostic, not a quality score.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from backend_session import backend_session
from client import Turn, wav16k_bytes
from lfm3_client import LFM3Client

LINES = ('Give me three ways to relax after work.', 'What about something else?')


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()
    with backend_session(args.url, test=True):
        client = LFM3Client(args.url)
        try:
            clips, transcripts = [], []
            for line in LINES:
                chunks, turn = [], Turn()
                client.tts(line, turn, on_audio=chunks.append)
                if turn.error or not chunks:
                    raise RuntimeError('Fixture synthesis failed: ' + str(turn.error))
                audio = np.concatenate(chunks)
                positions = np.arange(round(len(audio) * 2 / 3)) * 1.5
                wav = wav16k_bytes(np.interp(positions, np.arange(len(audio)), audio).astype(np.float32))
                transcript = client.asr_text(wav)
                if words(transcript) != words(line):
                    raise RuntimeError(f'Fixture did not transcribe as scripted: {transcript!r}')
                clips.append(wav)
                transcripts.append(transcript)
            rows = []
            for cycle in range(args.rounds):
                for condition in ('retained', 'reset', 'recovered', 'interrupted_recovered'):
                    first = Turn()
                    def interrupt_after_audio(_):
                        if first.audio_samples >= 24000:
                            first.abort()
                    client.chat_turn(clips[0], first, True,
                                     on_audio=interrupt_after_audio if condition == 'interrupted_recovered' else None)
                    followup = Turn()
                    started = time.perf_counter()
                    first_audio = []
                    def on_audio(_):
                        if not first_audio:
                            first_audio.append(time.perf_counter() - started)
                    client.chat_turn(clips[1], followup, condition != 'retained',
                                     recovery_wav=clips[0] if condition in ('recovered', 'interrupted_recovered') else None,
                                     recovery_reply=first.text if condition in ('recovered', 'interrupted_recovered') else '',
                                     on_audio=on_audio)
                    elapsed = time.perf_counter() - started
                    rows.append(dict(round=cycle + 1, condition=condition, first_reply=first.text,
                                     first_cancelled=first.cancel.is_set(),
                                     followup_reply=followup.text, first_error=first.error,
                                     followup_error=followup.error, followup_ttft_s=followup.ttft,
                                     request_to_first_audio_s=round(first_audio[0], 3) if first_audio else None,
                                     followup_total_s=round(elapsed, 3),
                                     followup_audio_s=followup.audio_samples / 24000))
            report = dict(scope='One synthetic voice and prompt pair; cancellation is simulated without device playback. ASR uses the same model.',
                          fixture_asr=transcripts, rows=rows)
            Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
        finally:
            client.close()


if __name__ == '__main__':
    main()
