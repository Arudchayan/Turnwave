"""Real backend + real VAD integration, with paced simulated mic/speaker devices.

No microphone is recorded and no sound is played. This complements, rather
than replaces, a live headset/loudspeaker listening check.
"""
import json
import pathlib
import queue
import sys
import threading
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from app import Assistant
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
    def __init__(self): self.playing = False
    def stop(self): self.playing = False
    def play_chunks(self, chunks, cancel, on_start, on_play):
        started = False
        try:
            for chunk in chunks:
                for i in range(0, len(chunk), 480):
                    if cancel.is_set(): return
                    if not started:
                        self.playing = True
                        started = True
                        on_start()
                    piece = chunk[i:i + 480]
                    on_play(piece)
                    cancel.wait(len(piece) / 24000)
        finally:
            self.playing = False


def wav(name):
    with wave.open(str(ROOT / 'data' / 'synth' / name), 'rb') as reader:
        assert reader.getframerate() == 16000
        return np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').astype(np.float32) / 32768


def main():
    from conversation_suite import ensure_idle
    ensure_idle()
    assistant = Assistant(mic=SyntheticMic(), player=PacedSpeaker(), front=DuplexFrontend(), client=ServerClient())
    assert assistant.client.health(), 'Start the local GPU server first'
    assistant.front.warmup()
    assistant.status = 'listening'
    assistant._spawn(assistant._loop)
    greet = wav('user_greet.wav')
    correction = wav('user_barge.wav')
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
        feed(np.zeros(4096, np.float32))
        feed(greet)
        input_end = time.perf_counter()
        await_condition(lambda: assistant.status == 'speaking')
        report['input_file_end_to_playback_s'] = round(time.perf_counter() - input_end, 3)
        first_job = assistant._active
        report['request_first_audio_s'] = round(first_job.turn.ttft, 3)
        feed(np.zeros(16000, np.float32))
        # Remove initial recorded silence, so onset timing means speech onset.
        nonquiet = np.flatnonzero(np.abs(correction) > .02)
        assert len(nonquiet)
        correction = correction[max(0, nonquiet[0] - 160):]
        onset = time.perf_counter()
        interrupted = None
        for i in range(0, len(correction), 512):
            feed(correction[i:i + 512])
            if interrupted is None and first_job.turn.cancel.is_set():
                interrupted = time.perf_counter() - onset
        assert interrupted is not None, 'Barge-in did not cancel the real backend turn'
        report['barge_cancel_s'] = round(interrupted, 3)
        await_condition(lambda: assistant.status == 'speaking', seconds=5)
        report['replacement_request_first_audio_s'] = round(assistant._active.turn.ttft, 3)
        assert assistant._active is not first_job
        assert not assistant.error, assistant.error
        report['passed'] = (report['input_file_end_to_playback_s'] < 1.2
                            and report['barge_cancel_s'] < .5)
        report['devices'] = 'paced simulated devices; real GPU inference and Silero/AEC'
        report['transcript'] = assistant.state()['transcript']
    finally:
        assistant.close()
    out = ROOT / 'outputs' / 'app_smoke.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    from backend_session import backend_session
    with backend_session(test=True):
        main()
