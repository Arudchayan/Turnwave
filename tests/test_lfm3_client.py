import base64
import io
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from client import Turn, wav16k_bytes
from lfm3_client import LFM3Client


def test_assistant_closes_both_voice_clients():
    from app import Assistant
    from test_realtime import FakeMic, FakePlayer, FakeFront
    closed = []
    client = LFM3Client('http://127.0.0.1:8080')
    client.close = lambda: closed.append('conversation')
    transcriber = LFM3Client('http://127.0.0.1:8081')
    transcriber.close = lambda: closed.append('transcription')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                    client=client, asr_client=transcriber)
    app.close()
    assert closed == ['conversation', 'transcription']


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
        assert 'lock_wait' in turn.voice_setup_s
        assert ('session_handshake' in turn.voice_setup_s) is first
        assert ('close_previous' in turn.voice_setup_s) is first
    assert len(connections) == 1
    assert sum(message['type'] == 'response.create' for message in sock.sent) == 2
    assert np.concatenate(chunks).shape == (4,)
    client.close()
    assert sock.closed


def test_text_prompt_uses_conversation_not_speech_synthesis(monkeypatch):
    pcm = np.array([500, -500], dtype='<i2').tobytes()
    sock = Socket(initial_events() + [
        {'type': 'response.output_text.delta', 'delta': 'Four.'},
        {'type': 'response.output_audio.delta', 'delta': base64.b64encode(pcm).decode()},
        {'type': 'response.done', 'response': {'status': 'completed'}}])
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: sock)
    client = LFM3Client()
    turn, requests = Turn(), []
    turn.request_observer = requests.append
    client.chat_text('What is two plus two?', turn, first=True)
    sent = next(x for x in sock.sent if x['type'] == 'conversation.item.create')
    assert sent['item']['content'][0]['text'] == 'What is two plus two?'
    assert requests[0]['text'] == 'What is two plus two?'
    assert turn.text == 'Four.' and turn.audio_samples == 2 and turn.error is None


def test_typed_and_spoken_turns_share_the_lfm3_conversation(monkeypatch):
    pcm = base64.b64encode(np.array([500], dtype='<i2').tobytes()).decode()
    output = [{'type': 'response.output_text.delta', 'delta': 'Answer.'},
              {'type': 'response.output_audio.delta', 'delta': pcm},
              {'type': 'response.done', 'response': {'status': 'completed'}}]
    sock = Socket(initial_events() + output + output)
    connections = []
    monkeypatch.setattr('lfm3_client.websocket.create_connection',
                        lambda *a, **k: connections.append(sock) or sock)
    client = LFM3Client()
    typed, spoken = Turn(), Turn()
    client.chat_text('Remember my name.', typed, first=True)
    client.chat_turn(wav16k_bytes(np.ones(800, np.float32)), spoken, first=False)
    assert len(connections) == 1
    assert typed.error is None and spoken.error is None
    assert [event['type'] for event in sock.sent].count('response.create') == 2
    assert any(event['type'] == 'conversation.item.create' for event in sock.sent)
    assert any(event['type'] == 'input_audio_buffer.append' for event in sock.sent)


def test_transcription_uses_separate_session(monkeypatch):
    sock = Socket(initial_events() + [{'type': 'conversation.item.input_audio_transcription.completed',
                                      'transcript': 'Hello there'}])
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: sock)
    turn = Turn()
    text = LFM3Client().asr_text(wav16k_bytes(np.ones(800, np.float32)), turn=turn)
    assert text == 'Hello there' and turn.done.is_set() and sock.closed
    assert sock.sent[0]['session']['type'] == 'transcription'
    assert sock.sent[-1] == {'type': 'input_audio_buffer.commit', 'final': True}


def test_interrupted_question_is_transcribed_and_supplied_as_context(monkeypatch):
    pcm = np.array([500, -500], dtype='<i2').tobytes()
    sock = Socket(initial_events() + [
        {'type': 'response.output_text.delta', 'delta': 'Try a warm bath.'},
        {'type': 'response.output_audio.delta', 'delta': base64.b64encode(pcm).decode()},
        {'type': 'response.done', 'response': {'status': 'completed'}}])
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: sock)
    client = LFM3Client()
    def transcribe(wav, turn):
        assert wav == b'previous audio'
        turn.text_parts.append('Give me ways to relax.')
        turn.done.set()
        return turn.text
    monkeypatch.setattr(client, 'asr_text', transcribe)
    turn = Turn()
    client.chat_turn(wav16k_bytes(np.ones(800, np.float32)), turn, True,
                     recovery_wav=b'previous audio', recovery_reply='Try going for a walk.')
    context = next(x for x in sock.sent if x['type'] == 'conversation.item.create')
    assert 'Give me ways to relax.' in context['item']['content'][0]['text']
    assert 'may not have been heard' in context['item']['content'][0]['text']
    assert 'Try going for a walk.' in context['item']['content'][0]['text']
    assert turn.text == 'Try a warm bath.' and turn.error is None and turn.done.is_set()


def test_failed_context_transcription_falls_back_to_fresh_turn(monkeypatch):
    pcm = np.array([500], dtype='<i2').tobytes()
    sock = Socket(initial_events() + [
        {'type': 'response.output_audio.delta', 'delta': base64.b64encode(pcm).decode()},
        {'type': 'response.done', 'response': {'status': 'completed'}}])
    monkeypatch.setattr('lfm3_client.websocket.create_connection', lambda *a, **k: sock)
    client = LFM3Client()
    def fail(wav, turn):
        turn.error = 'transcription unavailable'
        raise RuntimeError(turn.error)
    monkeypatch.setattr(client, 'asr_text', fail)
    turn = Turn()
    client.chat_turn(wav16k_bytes(np.ones(800, np.float32)), turn, True, recovery_wav=b'old')
    assert turn.error is None and turn.audio_samples == 1 and turn.done.is_set()
    assert not any(x['type'] == 'conversation.item.create' for x in sock.sent)


def test_cancellation_during_recovery_never_starts_a_new_reply(monkeypatch):
    client = LFM3Client()
    def cancel(wav, turn):
        turn.abort()
        turn.done.set()
        return ''
    monkeypatch.setattr(client, 'asr_text', cancel)
    monkeypatch.setattr('lfm3_client.websocket.create_connection',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('unexpected connection')))
    turn = Turn()
    client.chat_turn(wav16k_bytes(np.ones(800, np.float32)), turn, True, recovery_wav=b'old')
    assert turn.cancel.is_set() and turn.done.is_set() and turn.error is None


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

