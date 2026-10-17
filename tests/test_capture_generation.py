"""Reset boundaries with the real microphone queue/callback and fake I/O."""
from contextlib import nullcontext
from pathlib import Path
import socket
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import app
import audio_io
from audio_io import Mic
from app import Assistant
from client import wav16k_bytes
from test_capture_reset_queue import RecordingClient
from test_realtime import FakeFront, FakePlayer, wait_for


@pytest.fixture(autouse=True)
def simulated_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Physical streams and network calls are forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(audio_io.sd, 'InputStream', forbidden)
    monkeypatch.setattr(audio_io.sd, 'OutputStream', forbidden)
    monkeypatch.setattr(audio_io.sd, 'query_devices', lambda *a: dict(
        name='Simulated microphone', index=0, hostapi=0,
        max_input_channels=1, default_samplerate=16000))
    monkeypatch.setattr(audio_io.sd, 'query_hostapis', lambda *a: dict(name='Simulated host'))
    monkeypatch.setattr(app, 'audio_thread', nullcontext)


def microphone():
    return Mic(device=0, gain=1, max_buffer_s=2)


def action(assistant, name):
    if name == 'reset':
        assistant.reset()
    elif name == 'mute-cycle':
        assistant.set_mic_enabled(False)
        assistant.set_mic_enabled(True)


def observed_front(barrier):
    class Front(FakeFront):
        def update(self, data):
            result = super().update(data)
            if np.all(data == -.01):
                barrier.set()
            return result
    return Front()


@pytest.mark.parametrize('name', ['reset', 'mute-cycle', 'none'])
def test_audio_dequeued_before_boundary_is_rejected_but_fresh_speech_works(name):
    mic = microphone()
    dequeued, release, barrier = (threading.Event() for _ in range(3))

    class DelayedQueue(type(mic.q)):
        def delay(self, result):
            if not dequeued.is_set():
                dequeued.set()
                assert release.wait(2)
            return result

        def get(self, *args, **kwargs):
            return self.delay(super().get(*args, **kwargs))

        def get_with_generation(self, *args, **kwargs):
            return self.delay(super().get_with_generation(*args, **kwargs))

    mic.q = DelayedQueue(maxsize=mic.q.maxsize)
    assistant = Assistant(mic=mic, player=FakePlayer(), front=observed_front(barrier),
                          client=RecordingClient())
    assistant.status = 'listening'
    assistant._spawn(assistant._loop)
    try:
        mic.q.put(np.full(3200, .1, np.float32))
        assert dequeued.wait(1)
        action(assistant, name)
        mic.q.put(np.zeros(3200, np.float32))
        mic.q.put(np.full(3200, -.01, np.float32))
        release.set()
        assert barrier.wait(1)
        wait_for(lambda: not assistant._pending_jobs)
        if name == 'none':
            assert len(assistant.client.requests) == 1
        else:
            assert assistant.client.requests == []
            assert assistant.state()['transcript'] == []
        # Post-boundary speech is accepted, with no old block in its WAV.
        speech, silence = np.full(3200, .2, np.float32), np.zeros(3200, np.float32)
        previous = len(assistant.client.requests)
        mic.q.put(speech)
        mic.q.put(silence)
        wait_for(lambda: len(assistant.client.requests) == previous + 1)
        wait_for(lambda: not assistant._pending_jobs)
        assert assistant.client.requests[-1][0] == wav16k_bytes(np.concatenate((speech, silence)))
        assert assistant.error is None and assistant.mic_on
    finally:
        release.set()
        assistant.close()


@pytest.mark.parametrize('name', ['reset', 'mute-cycle'])
def test_first_fresh_block_after_reset_while_waiting_is_not_discarded(name):
    mic = microphone()
    waiting, barrier = threading.Event(), threading.Event()

    class WaitingQueue(type(mic.q)):
        def get(self, *args, **kwargs):
            waiting.set()
            return super().get(*args, **kwargs)

        def get_with_generation(self, *args, **kwargs):
            waiting.set()
            return super().get_with_generation(*args, **kwargs)

    mic.q = WaitingQueue(maxsize=mic.q.maxsize)
    assistant = Assistant(mic=mic, player=FakePlayer(), front=observed_front(barrier),
                          client=RecordingClient())
    assistant.status = 'listening'
    assistant._spawn(assistant._loop)
    try:
        assert waiting.wait(1)
        action(assistant, name)
        speech, silence = np.full(3200, .2, np.float32), np.zeros(3200, np.float32)
        mic.q.put(speech)
        mic.q.put(silence)
        mic.q.put(np.full(3200, -.01, np.float32))
        assert barrier.wait(1)
        wait_for(lambda: not assistant._pending_jobs)
        assert assistant.client.requests == [(wav16k_bytes(np.concatenate((speech, silence))), True)]
        assert assistant.error is None
    finally:
        assistant.close()


def test_callback_started_before_drain_cannot_enqueue_into_new_generation(monkeypatch):
    mic = microphone()
    entered, release = threading.Event(), threading.Event()
    original = np.asarray
    old = np.full((320, 1), .1, np.float32)
    failures = []

    def delayed(value, *args, **kwargs):
        if value is old:
            entered.set()
            assert release.wait(2)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(audio_io.np, 'asarray', delayed)
    def callback():
        try:
            mic._cb(old, len(old), None, False)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=callback)
    thread.start()
    try:
        assert entered.wait(1)
        mic.drain()
        release.set()
        thread.join(1)
        assert not thread.is_alive() and not failures
        assert mic.q.empty(), 'Callback from before drain leaked old audio'
        new = np.full((320, 1), .2, np.float32)
        mic._cb(new, len(new), None, False)
        np.testing.assert_array_equal(mic.q.get_nowait(), new[:, 0])
        assert mic.capture_errors == mic.dropped_blocks == 0
    finally:
        release.set()
        thread.join(1)


def test_callback_overflow_preserves_bounded_oldest_drop_and_array_api():
    mic = microphone()
    mic.q.maxsize = 2
    for value in [.1, .2, .3]:
        data = np.full((320, 1), value, np.float32)
        mic._cb(data, len(data), None, False)
    assert mic.q.qsize() == 2 and mic.dropped_blocks == 1
    np.testing.assert_array_equal(mic.q.get_nowait(), np.full(320, .2, np.float32))
    np.testing.assert_array_equal(mic.q.get_nowait(), np.full(320, .3, np.float32))
    data = np.arange(320, dtype=np.float32) / 1000
    mic.q.put(data)
    np.testing.assert_array_equal(mic.q.get_nowait(), data)
