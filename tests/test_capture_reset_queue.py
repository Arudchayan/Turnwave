"""Explicit conversation reset drops queued input; all devices/transport are fake."""
from contextlib import nullcontext
from pathlib import Path
import socket
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import app
from app import Assistant
from client import wav16k_bytes
from test_realtime import FakeClient, FakeFront, FakeMic, FakePlayer, wait_for


@pytest.fixture(autouse=True)
def fake_io_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Only fake devices and in-memory clients are allowed')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('audio_io.sd.InputStream', forbidden)
    monkeypatch.setattr('audio_io.sd.OutputStream', forbidden)
    monkeypatch.setattr(app, 'audio_thread', nullcontext)


class RecordingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.requests = []

    def chat_turn(self, wav, turn, first, on_audio=None):
        self.requests.append((wav, first))
        self.calls.append(first)
        turn.text_parts.append('Neutral reply.')
        turn.audio_samples = 3
        turn.finish_reason = 'stop'
        on_audio(np.array([.1, -.1, .2], np.float32))
        turn.done.set()


def capture_fixture():
    entered, release, barrier = (threading.Event() for _ in range(3))

    class BlockedFront(FakeFront):
        def update(self, samples):
            if not entered.is_set():
                entered.set()
                assert release.wait(2), 'Fake frontend barrier was not released'
                return False, False, False, samples
            result = super().update(samples)
            if np.all(samples == -.01):
                barrier.set()
            return result

    assistant = Assistant(mic=FakeMic(), player=FakePlayer(), front=BlockedFront(),
                          client=RecordingClient())
    assistant.status = 'listening'
    assistant._spawn(assistant._loop)
    assistant.mic.q.put(np.zeros(3200, np.float32))
    assert entered.wait(1)
    # These are both queued before any reset, while the old block is in-flight.
    assistant.mic.q.put(np.full(3200, .1, np.float32))
    assistant.mic.q.put(np.zeros(3200, np.float32))
    return assistant, release, barrier


def release_to_barrier(assistant, release, barrier):
    # A neutral marker proves every preceding queued block has been consumed.
    assistant.mic.q.put(np.full(3200, -.01, np.float32))
    release.set()
    assert barrier.wait(1)
    wait_for(lambda: not assistant._pending_jobs)


def test_reset_discards_queued_pre_reset_speech_without_repopulating_transcript():
    assistant, release, barrier = capture_fixture()
    try:
        assistant._say('Neutral previous conversation.')
        before_epoch = assistant._epoch
        assistant.reset()
        release_to_barrier(assistant, release, barrier)
        assert assistant.client.requests == []
        state = assistant.state()
        assert state['transcript'] == [] and state['assistant_partial'] == ''
        assert not state['active_turn'] and state['pending_turns'] == 0
        assert state['status'] == 'listening' and state['mic_on']
        assert assistant._epoch == before_epoch + 1 and assistant.first
    finally:
        release.set()
        assistant.close()


def test_fresh_post_reset_speech_is_submitted_with_reset_context():
    assistant, release, barrier = capture_fixture()
    try:
        assistant.reset()
        release_to_barrier(assistant, release, barrier)
        assert not assistant.client.requests
        speech = np.full(3200, .2, np.float32)
        silence = np.zeros(3200, np.float32)
        assistant.mic.q.put(speech)
        assistant.mic.q.put(silence)
        wait_for(lambda: len(assistant.client.requests) == 1)
        wait_for(lambda: not assistant._pending_jobs)
        # Normal 250ms pre-roll drops the earlier 200ms marker once this
        # 200ms speech block arrives. Only new speech and its endpoint remain.
        expected = wav16k_bytes(np.concatenate((speech, silence)))
        assert assistant.client.requests == [(expected, True)]
        assert assistant.state()['transcript'][-1] == 'assistant: Neutral reply.'
        assert assistant.error is None and assistant.first is False
    finally:
        release.set()
        assistant.close()


def test_without_reset_the_same_queue_is_processed_normally():
    assistant, release, barrier = capture_fixture()
    try:
        release_to_barrier(assistant, release, barrier)
        expected = wav16k_bytes(np.concatenate((np.full(3200, .1, np.float32), np.zeros(3200, np.float32))))
        assert assistant.client.requests == [(expected, True)]
        assert len(assistant.state()['transcript']) == 2
        assert assistant.turns == 1 and assistant.error is None
    finally:
        release.set()
        assistant.close()


@pytest.mark.parametrize('mic_on', [True, False])
def test_reset_preserves_microphone_setting_and_cancels_active_epoch(mic_on):
    entered = threading.Event()

    class WaitingClient(RecordingClient):
        def chat_turn(self, wav, turn, first, on_audio=None):
            entered.set()
            assert turn.cancel.wait(2)
            turn.done.set()

    assistant = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                          client=WaitingClient(), start_muted=not mic_on)
    try:
        job = assistant.submit_audio(np.full(3200, .1, np.float32))
        assert entered.wait(1)
        assistant.mic.q.put(np.full(320, .1, np.float32))
        assistant.reset()
        assert job.turn.cancel.is_set()
        wait_for(lambda: not assistant._pending_jobs)
        assert assistant.mic.q.empty()
        assert assistant.mic_on is mic_on and assistant._epoch == job.epoch + 1
        assert assistant.state()['transcript'] == [] and assistant.first
        assert assistant.status == 'listening' and assistant.error is None
    finally:
        assistant.close()
