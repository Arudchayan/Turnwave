"""Compare a contextual follow-up with and without an LFM3 session reset.

Run with the app stopped and an LFM3 backend running. Synthetic TTS is checked
with the same model's ASR; results are a narrow diagnostic, not a quality score.
"""
import argparse
import json
from pathlib import Path
import re
import sys

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
            for reset in (False, True):
                first = Turn()
                client.chat_turn(clips[0], first, True)
                followup = Turn()
                client.chat_turn(clips[1], followup, reset)
                rows.append(dict(reset_before_followup=reset, first_reply=first.text,
                                 followup_reply=followup.text, first_error=first.error,
                                 followup_error=followup.error, followup_ttft_s=followup.ttft,
                                 followup_audio_s=followup.audio_samples / 24000))
            report = dict(scope='One synthetic voice and prompt pair per condition; no live interruption. ASR uses the same model.',
                          fixture_asr=transcripts, rows=rows)
            Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
        finally:
            client.close()


if __name__ == '__main__':
    main()
