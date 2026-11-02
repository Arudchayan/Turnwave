"""Real backend and VAD, with a simulated mic and optional real playback.

No microphone is recorded. By default no sound is played; --real-playback
uses the selected output device. This is not a live human listening check.
"""
import json
import argparse
import pathlib
import queue
import re
import sys
import threading
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from app import Assistant
from audio_io import Player
from client import ServerClient
from duplex import DuplexFrontend


class SyntheticMic:
    device = -1
    def __init__(self): self.q = queue.Queue(maxsize=32)
    def start(self): pass
    def stop(self): pass
    def drain(self):
        while not self.q.empty(): self.q.get_nowait()


class PacedSpeaker:
    def __init__(self): self.playing = False; self.gain = 1.0
    def stop(self): self.playing = False; self.gain = 1.0
    def set_gain(self, level): self.gain = max(0., min(1., float(level)))
    def play_chunks(self, chunks, cancel, on_start, on_play, on_written=None):
        started = False
        try:
            for chunk in chunks:
                for i in range(0, len(chunk), 480):
                    if cancel.is_set(): return
                    if not started:
                        self.playing = True
                        started = True
                        on_start()
                    piece = chunk[i:i + 480] * self.gain
                    on_play(piece)
                    if cancel.wait(len(piece) / 24000): return
                    if on_written is not None: on_written(piece)
        finally:
            self.playing = False


def wav(name):
    with wave.open(str(ROOT / 'data' / 'synth' / name), 'rb') as reader:
        assert reader.getframerate() == 16000
        return np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').astype(np.float32) / 32768


def assess_interruption(report):
    """A quick duck is insufficient if an interruption splits or loses words."""
    return (report['barge_duck_from_onset_s'] < .5
            and report['barge_cancel_after_fixture_s'] < 2
            and report['premature_cancel_s'] <= .12
            and report['user_turns'] == 2
            and not report.get('expected_words_missing'))


def main(args):
    from conversation_suite import ensure_idle
    ensure_idle()
    if args.backend == 'lfm3':
        from lfm3_client import LFM3Client
        client = LFM3Client(args.url, timeout=90)
    else:
        client = ServerClient(args.url, timeout=90)
    player = Player() if args.real_playback else PacedSpeaker()
    assistant = Assistant(mic=SyntheticMic(), player=player, front=DuplexFrontend(aec=not args.disable_aec), client=client)
    assistant.front.set_end_silence(args.pause_ms)
    submitted = []
    original_submit = assistant.submit_audio
    def capture_submission(audio):
        job = original_submit(audio)
        submitted.append(job)
        return job
    assistant.submit_audio = capture_submission
    assert assistant.client.health(), 'Start the local GPU server first'
    assistant.front.warmup()
    assistant.status = 'listening'
    def read_fixture(path, default):
        if path is None: return wav(default)
        with wave.open(str(path), 'rb') as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Fixtures must be 16 kHz mono PCM16 WAVs')
            return np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').astype(np.float32) / 32768
    greet = read_fixture(args.greet, 'user_greet.wav')
    correction = read_fixture(args.barge, 'user_barge.wav')
    report = {}
    def feed(audio):
        # Monotonic deadlines avoid accumulating 32 ms scheduling errors.
        deadline = time.perf_counter()
        for i in range(0, len(audio), 512):
            block = audio[i:i + 512]
            if len(block) < 512: block = np.pad(block, (0, 512 - len(block)))
            assistant.mic.q.put(block)
            deadline += .032
            time.sleep(max(0, deadline - time.perf_counter()))
    def await_condition(predicate, seconds=3):
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if predicate(): return
            feed(np.zeros(512, np.float32))
        raise AssertionError('Application did not reach expected state: ' + str(assistant.state()))
    try:
        assistant._spawn(assistant._loop)
        feed(np.zeros(4096, np.float32))
        feed(greet)
        input_end = time.perf_counter()
        await_condition(lambda: assistant.status == 'speaking')
        report['input_file_end_to_playback_s'] = round(time.perf_counter() - input_end, 3)
        first_job = assistant._active
        report['request_first_audio_s'] = round(first_job.turn.ttft, 3)
        feed(np.zeros(8000, np.float32))
        # Remove initial recorded silence, so onset timing means speech onset.
        nonquiet = np.flatnonzero(np.abs(correction) > .02)
        assert len(nonquiet)
        correction = correction[max(0, nonquiet[0] - 160):]
        voiced = np.flatnonzero(np.abs(correction) > .02)
        last_voiced_s = float(voiced[-1]) / 16000
        onset = time.perf_counter()
        ducked = None
        cancelled = None
        for i in range(0, len(correction), 512):
            feed(correction[i:i + 512])
            if ducked is None and any(event['event'] == 'barge_candidate' and event.get('turn_id') == first_job.id
                                       for event in assistant.diagnostics.recent(100)):
                ducked = time.perf_counter() - onset
            if cancelled is None and first_job.turn.cancel.is_set():
                cancelled = time.perf_counter() - onset
        input_end = time.perf_counter()
        assert ducked is not None, 'Speech did not duck the reply while it was playing'
        await_condition(first_job.turn.cancel.is_set, seconds=3)
        if cancelled is None:
            cancelled = time.perf_counter() - onset
        report['barge_duck_from_onset_s'] = round(ducked, 3)
        report['barge_cancel_after_fixture_s'] = round(max(0, onset + cancelled - input_end), 3)
        report['barge_cancel_from_onset_s'] = round(cancelled, 3)
        report['last_voiced_sample_s'] = round(last_voiced_s, 3)
        report['premature_cancel_s'] = round(max(0, last_voiced_s - cancelled), 3)
        await_condition(lambda: len(submitted) > 1 and submitted[-1].first_play_at, seconds=15)
        replacement = submitted[-1]
        report['replacement_request_first_audio_s'] = round(replacement.turn.ttft, 3)
        assert replacement is not first_job
        assert not assistant.error, assistant.error
        if args.verify_asr:
            await_condition(replacement.playback_closed.is_set, seconds=30)
            # This is a separate ASR request to the same backend, not ground truth.
            report['captured_asr'] = [client.asr_text(job.wav) for job in submitted[1:]]
            if args.expected_words:
                words = set(re.findall(r"[\w']+", ' '.join(report['captured_asr']).lower()))
                report['expected_words_missing'] = [word for word in args.expected_words
                                                    if word.lower() not in words]
        report['user_turns'] = sum(text.startswith('user:') for text in assistant.state()['transcript'])
        report['passed'] = assess_interruption(report)
        report['devices'] = ('simulated mic and real output device; real GPU inference and Silero/AEC'
                             if args.real_playback else 'paced simulated devices; real GPU inference and Silero/AEC')
        report['aec_enabled'] = not args.disable_aec
        report['pause_ms'] = args.pause_ms
        report['detector_events'] = [event for event in assistant.diagnostics.recent(100)
                                     if event['event'] in ('speech_start', 'speech_end',
                                                           'barge_candidate', 'barge_confirm', 'barge_restore')]
        if args.real_playback:
            report['output_device'] = player.device
        report['transcript'] = assistant.state()['transcript']
    finally:
        assistant.close()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    print(encoded, flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    from backend_session import backend_session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['lfm2', 'lfm3'], default='lfm2')
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--greet', type=pathlib.Path)
    parser.add_argument('--barge', type=pathlib.Path)
    parser.add_argument('--real-playback', action='store_true', help='write generated speech to the selected output device')
    parser.add_argument('--disable-aec', action='store_true', help='diagnostic comparison only; leave echo cancellation on for normal use')
    parser.add_argument('--pause-ms', type=int, choices=[480, 800, 1120], default=480)
    parser.add_argument('--verify-asr', action='store_true', help='transcribe captured interrupting turns after playback')
    parser.add_argument('--expected-words', nargs='+', default=[], help='lowercase words expected in the separate ASR result')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.expected_words and not args.verify_asr:
        parser.error('--expected-words requires --verify-asr')
    if args.output.exists():
        parser.error('Refusing to overwrite existing output')
    with backend_session(args.url, test=True):
        main(args)
