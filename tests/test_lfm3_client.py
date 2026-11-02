import base64
import io
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from client import Turn, wav16k_bytes
from lfm3_client import LFM3Client


class Socket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []
        self.closed = False
    def recv(self): return json.dumps(next(self.events))
    def send(self, message): self.sent.append(json.loads(message))
    def settimeout(self, timeout): pass
    def close(self): self.closed = True


def initial_events():
    return [{'type': 'session.created', 'session': {'audio': {'input': {'format': {'rate': 16000}},
             'output': {'format': {'rate': 24000}}}}}, {'type': 'session.updated'}]


def test_realtime_chat_streams_and_reuses_context(monkeypatch):
    pcm = np.array([500, -500], dtype='<i2').tobytes()
    output = [{'type': 'response.output_text.delta', 'delta': 'Hello'},
              {'type': 'response.output_audio.delta', 'delta': base64.b64encode(pcm).decode()},
              {'type': 'response.done', 'response': {'status': 'completed'}}]
    sock = Socket(initial_events() + output + output)
    connections = []
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: connections.append(sock) or sock)
    client = LFM3Client()
    wav = wav16k_bytes(np.ones(4000, np.float32) * .1)
    chunks = []
    for first in (True, False):
        turn = Turn()
        client.chat_turn(wav, turn, first, on_audio=chunks.append)
        assert turn.done.is_set() and turn.error is None and turn.text == 'Hello'
        assert turn.audio_samples == 2
    assert len(connections) == 1
    assert sum(message['type'] == 'response.create' for message in sock.sent) == 2
    assert np.concatenate(chunks).shape == (4,)
    client.close()
    assert sock.closed


def test_transcription_uses_separate_session(monkeypatch):
    sock = Socket(initial_events() + [{'type': 'conversation.item.input_audio_transcription.completed',
                                      'transcript': 'Hello there'}])
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: sock)
    turn = Turn()
    text = LFM3Client().asr_text(wav16k_bytes(np.ones(800, np.float32)), turn=turn)
    assert text == 'Hello there' and turn.done.is_set() and sock.closed
    assert sock.sent[0]['session']['type'] == 'transcription'
    assert sock.sent[-1] == {'type': 'input_audio_buffer.commit', 'final': True}


def test_tts_handles_split_pcm_frames(monkeypatch):
    data = np.array([.25, -.5, .75], dtype='<f4').tobytes()
    class Response:
        headers = {'Content-Type': 'audio/pcm;rate=24000;encoding=float;bits=32'}
        def __init__(self): self.parts = iter([data[:3], data[3:7], data[7:], b''])
        def read1(self, n): return next(self.parts)
        def close(self): pass
    monkeypatch.setattr('lfm3_client.urllib.request.urlopen', lambda *a, **k: Response())
    turn = Turn(); chunks = []
    LFM3Client().tts('Hello', turn, on_audio=chunks.append)
    assert turn.error is None and turn.done.is_set()
    np.testing.assert_array_equal(np.concatenate(chunks), [.25, -.5, .75])


def test_cancelled_turn_does_not_connect(monkeypatch):
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    turn = Turn(); turn.abort()
    LFM3Client().chat_turn(wav16k_bytes(np.ones(8)), turn, True)
    assert turn.done.is_set() and turn.error is None


def test_tts_rejects_wrong_format_and_releases_lock(monkeypatch):
    class Response:
        headers = {'Content-Type': 'audio/wav'}
        def close(self): pass
    monkeypatch.setattr('lfm3_client.urllib.request.urlopen', lambda *a, **k: Response())
    client = LFM3Client(); turn = Turn()
    client.tts('Hello', turn)
    assert turn.done.is_set() and 'format' in turn.error
    assert client.request_lock.acquire(blocking=False)
    client.request_lock.release()


def test_tts_close_failure_does_not_mask_result_or_leak_lock(monkeypatch):
    class Response:
        headers = {'Content-Type': 'audio/pcm;rate=24000;encoding=float;bits=32'}
        def __init__(self): self.data = iter([np.array([.1], '<f4').tobytes(), b''])
        def read1(self, n): return next(self.data)
        def close(self): raise OSError('close failed')
    monkeypatch.setattr('lfm3_client.urllib.request.urlopen', lambda *a, **k: Response())
    client = LFM3Client(); turn = Turn()
    client.tts('Hello', turn)
    assert turn.done.is_set() and turn.error is None and turn.audio_samples == 1
    assert client.request_lock.acquire(blocking=False)
    client.request_lock.release()


def test_set_voice_params_passes_sampling_and_voice_to_tts(monkeypatch):
    captured = {}
    class Response:
        headers = {'Content-Type': 'audio/pcm;rate=24000;encoding=float;bits=32'}
        def __init__(self): self.data = iter([np.array([.1], '<f4').tobytes(), b''])
        def read1(self, n): return next(self.data)
        def close(self): pass
    def fake_urlopen(req, timeout=None):
        captured['payload'] = json.loads(req.data.decode('utf-8'))
        return Response()
    monkeypatch.setattr('lfm3_client.urllib.request.urlopen', fake_urlopen)
    client = LFM3Client()
    client.set_voice_params(temperature=0.55, top_k=30, top_p=0.88, min_p=0.06, repeat_penalty=1.08, voice='jarvis')
    turn = Turn()
    client.tts('Hello', turn)
    assert turn.error is None
    assert captured['payload']['temperature'] == 0.55
    assert captured['payload']['top_k'] == 30
    assert captured['payload']['top_p'] == 0.88
    assert captured['payload']['min_p'] == 0.06
    assert captured['payload']['repeat_penalty'] == 1.08
    assert captured['payload']['voice'] == 'jarvis'

