"""Compare resetting an interrupted LFM3 session with replaying previous audio.

Run with the app stopped and one LFM3 server running. This small diagnostic uses
one synthetic prompt pair and cancels after one second of generated audio; it
does not simulate device playback or estimate general conversational quality.
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


class ReplayProbeClient(LFM3Client):
    """Research-only replay: two audio segments in a single new user turn."""
    def __init__(self, base):
        super().__init__(base)
        self.replay = None

    def _append(self, sock, pcm, turn):
        replay, self.replay = self.replay, None
        if replay is not None:
            wav, draft = replay
            def item(text):
                sock.send(json.dumps({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user',
                    'content': [{'type': 'input_text', 'text': text}]}}))
            item('The following audio is the previous user request.')
            super()._append(sock, self._pcm(wav), turn)
            if turn.cancel.is_set():
                return
            sock.send(json.dumps({'type': 'input_audio_buffer.commit'}))
            item('The assistant response was interrupted. Its draft, which may not have been heard, was: '
                 + draft[:600] + '. Respond to the next audio from the user.')
        super()._append(sock, pcm, turn)


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    with backend_session(args.url, test=True):
        client = ReplayProbeClient(args.url)
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
                    raise RuntimeError('Fixture did not transcribe as scripted: ' + repr(transcript))
                clips.append(wav)
                transcripts.append(transcript)
            rows = []
            for cycle in range(args.rounds):
                for mode in ('reset', 'audio_replay'):
                    first = Turn()
                    def interrupt(_):
                        if first.audio_samples >= 24000:
                            first.abort()
                    client.chat_turn(clips[0], first, True, on_audio=interrupt)
                    if not first.cancel.is_set():
                        raise RuntimeError('First reply did not reach the cancellation point')
                    if mode == 'audio_replay':
                        client.replay = (clips[0], first.text)
                    followup, started, first_audio = Turn(), time.perf_counter(), []
                    def on_audio(_):
                        if not first_audio:
                            first_audio.append(time.perf_counter() - started)
                    client.chat_turn(clips[1], followup, True, on_audio=on_audio)
                    row = dict(round=cycle + 1, condition=mode, first_draft=first.text,
                               first_cancelled=first.cancel.is_set(), reply=followup.text,
                               error=followup.error, audio_samples=followup.audio_samples,
                               request_to_first_audio_s=round(first_audio[0], 3) if first_audio else None)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            report = dict(scope='One synthetic voice and prompt pair; fixed reset-then-replay order; model cancellation without device playback; ASR uses the same model.',
                          fixture_asr=transcripts, rows=rows)
            Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            client.close()


if __name__ == '__main__':
    main()
