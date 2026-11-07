"""The optional realtime client without a live llama.cpp server."""
import base64
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import realtime_client
from client import Turn, wav16k_bytes
from realtime_client import RealtimeClient


class FakeSocket:
    def __init__(self, events=None):
        self.events = list(events or [])
        self.sent = []
        self.closed = False
        self.timeouts = 0

    def recv(self):
        if self.events:
            event = self.events.pop(0)
            return event if isinstance(event, str) else json.dumps(event)
        self.timeouts += 1
        raise realtime_client.websocket.WebSocketTimeoutException()

    def send(self, data):
        self.sent.append(json.loads(data))

    def settimeout(self, value):
        pass

    def close(self):
        self.closed = True


def pcm_event(samples=2400):
    data = base64.b64encode(np.ones(samples, dtype='<i2').tobytes()).decode()
    return {'type': 'response.output_audio.delta', 'delta': data}


def client_with(socket):
    client = RealtimeClient('ws://127.0.0.1:8090/v1/realtime')
    client._connect = lambda *args, **kwargs: socket
    return client


def test_realtime_turn_streams_text_and_audio():
    ws = FakeSocket([
        {'type': 'response.created', 'response': {'status': 'in_progress'}},
        {'type': 'response.output_text.delta', 'delta': 'hello'},
        pcm_event(2400),
        {'type': 'response.done', 'response': {'status': 'completed'}},
    ])
    client = client_with(ws)
    turn = Turn()
    client.chat_turn(wav16k_bytes(np.ones(1600, np.float32) * .1), turn, first=True)
    assert turn.text == 'hello'
    assert turn.audio_samples == 2400
    assert turn.ttft is not None and turn.ttft >= 0
    assert turn.finish_reason == 'stop'
    assert turn.error is None and turn.done.is_set()
    kinds = [event['type'] for event in ws.sent]
    assert kinds[0] == 'input_audio_buffer.append'
    assert 'input_audio_buffer.commit' in kinds
    assert 'response.create' in kinds


def test_realtime_typed_turn_sends_text_and_receives_spoken_response():
    ws = FakeSocket([{'type': 'response.output_text.delta', 'delta': 'Four.'},
                     pcm_event(2400),
                     {'type': 'response.done', 'response': {'status': 'completed'}}])
    turn = Turn()
    client_with(ws).chat_text('What is two plus two?', turn, first=True)
    assert ws.sent[0]['item']['content'][0]['text'] == 'What is two plus two?'
    assert not any(e['type'] == 'input_audio_buffer.commit' for e in ws.sent)
    assert turn.text == 'Four.' and turn.audio_samples == 2400 and turn.error is None


def test_cancelled_typed_turn_does_not_connect_or_send():
    client = RealtimeClient('ws://127.0.0.1:8090/v1/realtime')
    client._connect = lambda *args, **kwargs: pytest.fail('Unexpected connection')
    turn = Turn()
    turn.abort()
    client.chat_text('A cancelled question', turn, first=True)
    assert turn.done.is_set() and client._ws is None


def test_first_turn_closes_a_previous_socket():
    old, new = FakeSocket(), FakeSocket([
        {'type': 'response.done', 'response': {'status': 'completed'}}])
    client = RealtimeClient('ws://127.0.0.1:8090/v1/realtime')
    client._ws = old
    client._connect = lambda *args, **kwargs: new
    turn = Turn()
    client.chat_turn(wav16k_bytes(np.ones(160, np.float32) * .1), turn, first=True)
    assert old.closed
    assert client._ws is new


def test_cancellation_before_a_response_creates_nothing():
    ws = FakeSocket([])
    client = client_with(ws)
    turn = Turn()
    turn.cancel.set()
    client.chat_turn(wav16k_bytes(np.ones(160, np.float32) * .1), turn, first=True)
    assert not any(event['type'] == 'response.create' for event in ws.sent)
    assert client._ws is None
    assert turn.done.is_set()


def test_stream_cancellation_sends_response_cancel():
    ws = FakeSocket([])
    client = client_with(ws)
    turn = Turn()
    turn.cancel.set()
    client._stream(ws, turn)
    assert any(event['type'] == 'response.cancel' for event in ws.sent)


def test_audio_input_resamples_to_16k():
    import io, wave
    samples = np.ones(2400, dtype='<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(samples.tobytes())
    pcm = RealtimeClient._audio_input(RealtimeClient.__new__(RealtimeClient), buf.getvalue())
    assert len(pcm) == 1600 * 2


def test_health_rejects_a_closed_port():
    client = RealtimeClient('ws://127.0.0.1:1/v1/realtime')
    assert client.health() is False


def test_app_selects_realtime_only_when_asked(monkeypatch):
    from app import Assistant
    from test_realtime import FakeFront, FakeMic, FakePlayer

    monkeypatch.setenv('LOCALVOICE_TRANSPORT', 'realtime')
    assistant = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront())
    try:
        assert isinstance(assistant.client, RealtimeClient)
        with pytest.raises(ValueError, match='local conversation only'):
            assistant.set_agent_mode('opencode-go')
        assert assistant.agent is None
    finally:
        assistant.close()


def test_main_transport_factory_and_context_lock_follow_realtime_url(monkeypatch):
    from app import create_voice_clients, voice_context_endpoint

    monkeypatch.setenv('LOCALVOICE_TRANSPORT', 'realtime')
    monkeypatch.setenv('LOCALVOICE_REALTIME', 'ws://localhost:8090/v1/realtime')
    client, asr = create_voice_clients('lfm3', 'realtime')
    try:
        assert isinstance(client, RealtimeClient)
        assert client.url == 'ws://localhost:8090/v1/realtime'
        assert asr is None
        assert voice_context_endpoint() == 'http://localhost:8090'
        with pytest.raises(ValueError, match='local conversation only'):
            create_voice_clients('lfm3', 'realtime', agent_enabled=True)
        with pytest.raises(ValueError, match='local conversation only'):
            create_voice_clients('lfm3', 'realtime', asr_url='http://127.0.0.1:8081')
    finally:
        client.close()
