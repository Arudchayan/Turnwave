"""Measure interruption and follow-up through the local app and real LFM3.

Uses 16 kHz generated mono PCM16 WAVs, a paced virtual microphone and
speaker, and an isolated backend. No physical devices or cloud calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import sys
import time
from urllib.parse import urlsplit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Assistant
from backend_session import backend_session
from diagnostics import Diagnostics
from lfm3_client import LFM3Client
from synth.echo_tail_probe import read_wav


class VirtualMic:
    device = -1
    native_rate = 16000

    def __init__(self):
        self.q = queue.Queue()

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        self.drain()


class PacedSpeaker:
    playing = False

    def set_gain(self, *_args):
        pass

    def stop(self):
        self.playing = False

    def play_chunks(self, chunks, cancel, on_start, on_play, on_written=None):
        try:
            for chunk in chunks:
                for start in range(0, len(chunk), 480):
                    if cancel.is_set():
                        return
                    block = chunk[start:start + 480]
                    if not self.playing:
                        self.playing = True
                        on_start()
                    on_play(block)
                    if cancel.wait(len(block) / 24000):
                        return
                    if on_written:
                        on_written(block)
        finally:
            self.playing = False


class MeasuredAssistant(Assistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.submitted = []

    def submit_audio(self, audio):
        job = super().submit_audio(audio)
        self.submitted.append(job)
        return job


def until(predicate, label, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise TimeoutError(label)


def send(mic, audio, silence=.0):
    tail = np.zeros(round(silence * 16000), dtype=np.float32)
    track = np.concatenate((audio, tail))
    origin = time.perf_counter()
    for offset in range(0, len(track), 320):
        frame = track[offset:offset + 320]
        if len(frame) < 320:
            frame = np.pad(frame, (0, 320 - len(frame)))
        mic.q.put(frame)
        time.sleep(max(0.0, origin + (offset + 320) / 16000 - time.perf_counter()))


def closed(job):
    return job.generation_closed.is_set() and job.playback_closed.is_set()


def describe(job):
    return {
        'submit_to_audio_s': round(job.first_play_at - job.t0, 3) if job.first_play_at else None,
        'model_first_audio_s': job.turn.ttft,
        'cancelled': job.turn.cancel.is_set(),
        'audio_s': round(job.played_samples / 24000, 3),
        'error': bool(job.turn.error),
        'text_chars': len(job.turn.text),
    }


def trial(url, mode, fixtures, directory):
    mic, speaker = VirtualMic(), PacedSpeaker()
    diagnostics = Diagnostics(directory / 'app.jsonl')
    app = MeasuredAssistant(mic=mic, player=speaker, client=LFM3Client(url),
                            diagnostics=diagnostics, floor_mode=mode)
    app.status = 'listening'
    app.front.warmup()
    app._spawn(app._loop)
    try:
        first = app.submit_audio(fixtures['initial'])
        until(lambda: first.first_play_at, 'initial playback missing')
        send(mic, fixtures['ack'], .5)
        ack_overlap_valid = not first.playback_closed.is_set() or first.turn.cancel.is_set()
        if not first.playback_closed.is_set():
            button_at = time.perf_counter()
            app.interrupt()
            until(first.playback_closed.is_set, 'button did not stop playback')
            stop_ms = round((time.perf_counter() - button_at) * 1000, 1)
        else:
            stop_ms = None
        until(lambda: closed(first), 'first job did not close')
        send(mic, np.zeros(0, dtype=np.float32), 1.35)
        ack_submissions = len(app.submitted) - 1
        if ack_submissions:
            until(lambda: closed(app.submitted[-1]), 'acknowledgment response did not close')

        # Always exercise a button cancellation with the same long prompt.
        app.reset()
        before = len(app.submitted)
        button_job = app.submit_audio(fixtures['initial'])
        until(lambda: button_job.first_play_at, 'button-arm playback missing')
        button_at = time.perf_counter()
        producing = not button_job.generation_closed.is_set()
        app.interrupt()
        until(button_job.playback_closed.is_set, 'button-arm playback did not stop')
        button_ms = round((time.perf_counter() - button_at) * 1000, 1)
        send(mic, fixtures['followup'], 1.25)
        follow = until(lambda: app.submitted[before + 1] if len(app.submitted) > before + 1 else None,
                       'post-button input missing')
        until(lambda: closed(follow), 'post-button response did not close')
        return {
            'mode': mode, 'ack_overlap_valid': ack_overlap_valid,
            'ack_extra_submissions': ack_submissions, 'first': describe(first),
            'ack_arm_button_stop_ms': stop_ms, 'button_stop_ms': button_ms,
            'button_generation_in_progress': producing, 'button_turn': describe(button_job),
            'followup': describe(follow),
            'followup_mentions_tomato': 'tomato' in follow.turn.text.casefold(),
        }
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Dedicated loopback LFM3 endpoint')
    parser.add_argument('--mode', choices=('manual', 'reversible'), required=True)
    for name in ('initial', 'ack', 'followup', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    endpoint = urlsplit(args.url)
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.port not in range(18100, 18200) or endpoint.netloc != f'127.0.0.1:{endpoint.port}'
            or endpoint.path or endpoint.query or endpoint.fragment):
        parser.error('Use an isolated loopback backend on a port from 18100 to 18199.')
    fixtures, hashes = {}, {}
    for name in ('initial', 'ack', 'followup'):
        path = getattr(args, name)
        audio, rate = read_wav(path)
        if rate != 16000 or not len(audio):
            parser.error('Provide nonempty 16 kHz generated WAV fixtures.')
        fixtures[name] = audio
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    with backend_session(args.url):
        result = trial(args.url, args.mode, fixtures, args.out)
    result['fixture_sha256'] = hashes
    payload = json.dumps(result, indent=2, allow_nan=False)
    (args.out / 'summary.json').write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
