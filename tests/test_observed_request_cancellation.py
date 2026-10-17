"""Cancellation at the archival observer boundary, without network or devices."""
import base64
from contextlib import nullcontext
import io
import json
from pathlib import Path
import socket
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import app
from app import Assistant
from client import ServerClient, Turn, wav16k_bytes
from test_realtime import FakeFront, FakeMic, FakePlayer, wait_for


@pytest.fixture(autouse=True)
def no_live_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Only in-memory transport and fake devices are allowed')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('audio_io.sd.InputStream', forbidden)
    monkeypatch.setattr('audio_io.sd.OutputStream', forbidden)
    monkeypatch.setattr(app, 'audio_thread', nullcontext)


def response():
    samples = np.array([.125, -.25, .375], dtype='<f4')
    event = {'choices': [{'delta': {'content': 'Neutral response.',
             'audio_chunk': {'data': base64.b64encode(samples).decode()}},
             'finish_reason': 'stop'}]}
    return io.BytesIO(('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode())


def assert_released(client, turn):
    assert turn.done.is_set() and turn._response is None
    assert client.request_lock.acquire(blocking=False)
    client.request_lock.release()


def test_observer_cancellation_does_not_submit_and_releases_successor():
    client = ServerClient()
    turn = Turn()
    observed, posted = [], []
    def observer(payload):
        observed.append(payload)
        turn.abort()
    turn.request_observer = observer
    client._post = lambda payload: posted.append(payload) or response()
    client.chat_turn(b'neutral-old-wave', turn, True)
    assert len(observed) == 1
    assert posted == []
    assert turn.cancel.is_set() and turn.error is None and turn.t_req == 0
    assert_released(client, turn)

    recovery = Turn()
    received = []
    client.chat_turn(b'neutral-new-wave', recovery, True, on_audio=received.append)
    assert len(posted) == 1 and posted[0]['reset_context'] is True
    assert recovery.error is None and recovery.finish_reason == 'stop'
    np.testing.assert_array_equal(np.concatenate(received), [.125, -.25, .375])
    assert_released(client, recovery)


def test_already_cancelled_turn_skips_observer_and_transport():
    client = ServerClient()
    turn = Turn()
    turn.abort()
    def forbidden(*args):
        pytest.fail('Pre-cancelled turn crossed the request boundary')
    turn.request_observer = forbidden
    client._post = forbidden
    client.chat_turn(b'neutral-wave', turn, True)
    assert turn.error is None and turn.t_req == 0
    assert_released(client, turn)


def test_normal_observers_preserve_payload_audio_and_context_sequence():
    client = ServerClient()
    observed, posted = [], []
    client._post = lambda payload: posted.append(payload) or response()
    wav = wav16k_bytes(np.full(3200, .1, np.float32))
    for first in [True, False]:
        turn = Turn()
        turn.request_observer = observed.append
        received = []
        client.chat_turn(wav, turn, first, on_audio=received.append)
        assert turn.error is None and turn.audio_samples == 3
        np.testing.assert_array_equal(np.concatenate(received), [.125, -.25, .375])
        assert_released(client, turn)
    assert observed == posted
    assert [payload['reset_context'] for payload in posted] == [True, False]
    for payload in posted:
        encoded = payload['messages'][-1]['content'][0]['input_audio']
        assert encoded['format'] == 'wav'
        assert base64.b64decode(encoded['data'], validate=True) == wav


def test_app_interruption_during_archive_observer_does_not_block_replacement(tmp_path):
    entered, release = threading.Event(), threading.Event()
    obsolete_post, release_obsolete_post, successor_post = (threading.Event() for _ in range(3))
    requests = []

    class Archive:
        def __init__(self, turn_id):
            self.turn_id = turn_id
        def record_request(self, payload):
            if self.turn_id == 1:
                entered.set()
                assert release.wait(3)
        def write_audio(self, audio): pass
        def finish_generation(self, turn): pass
        def finish_playback(self, *args, **kwargs): pass

    class Session:
        folder = tmp_path
        sealed = False
        def begin_turn(self, turn_id, *args, **kwargs):
            return Archive(turn_id)

    client = ServerClient()
    old_wav = wav16k_bytes(np.full(8000, .1, np.float32))
    new_wav = wav16k_bytes(np.full(8000, .2, np.float32))
    def post(payload):
        data = payload['messages'][-1]['content'][0]['input_audio']['data']
        wav = base64.b64decode(data, validate=True)
        requests.append((wav, payload['reset_context']))
        if wav == old_wav:
            obsolete_post.set()
            assert release_obsolete_post.wait(3)
        else:
            assert wav == new_wav
            successor_post.set()
        return response()
    client._post = post
    assistant = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                          client=client, session_archive=Session())
    new = None
    try:
        old = assistant.submit_audio(np.full(8000, .1, np.float32))
        assert entered.wait(1)
        assistant.interrupt('neutral test interruption')
        new = assistant.submit_audio(np.full(8000, .2, np.float32))
        release.set()
        # The successor must post before the obsolete response is released.
        # This is a forced scheduling boundary, not a latency benchmark.
        assert successor_post.wait(1), 'Cancelled observer request delayed the replacement'
        assert not obsolete_post.is_set()
        assert new.playback_closed.wait(1)
        wait_for(lambda: assistant.state()['pending_turns'] == 0)
        assert requests == [(new_wav, True)]
        assert old.turn.cancel.is_set() and old.turn.t_req == 0
        assert old.generation_closed.is_set() and old.playback_closed.is_set()
        assert new.turn.error is None and new.played_samples == 3
        assert assistant.error is None and assistant.first is False
        assert_released(client, old.turn)
        assert_released(client, new.turn)
    finally:
        release.set()
        release_obsolete_post.set()
        if new is not None:
            assert new.playback_closed.wait(2)
        assistant.close()
