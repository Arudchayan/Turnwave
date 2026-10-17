"""Regression tests for streaming, interruption, context, and audio continuity."""
import base64
import json
import pathlib
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from app import Assistant, Handler, Job
from audio_io import Player
from client import ServerClient, Turn, extract_audio, sse_events, wav16k_bytes
from duplex import DuplexFrontend


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'Timed out waiting for condition'


def audio_event(text='hello'):
    return {'choices': [{'delta': {'content': text, 'audio': {
        'data': base64.b64encode(np.ones(1440, dtype='<i2').tobytes()).decode(),
        'sample_rate': 24000}}}]}


@pytest.fixture
def streaming_server():
    release = threading.Event()
    first = threading.Event()
    active = [0, 0]
    lock = threading.Lock()
    class StreamHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            with lock:
                active[0] += 1
                active[1] = max(active)
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                data = ('data: ' + json.dumps(audio_event()) + '\n\n').encode()
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.flush()
                first.set()
                release.wait(2)
                data = b'data: [DONE]\n\n'
                self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n0\r\n\r\n')
                self.wfile.flush()
            except (OSError, ValueError):
                pass
            finally:
                with lock: active[0] -= 1
    server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    yield ServerClient(f'http://127.0.0.1:{server.server_port}', timeout=2), release, first, active
    release.set(); server.shutdown(); server.server_close(); worker.join(1)


def test_first_audio_arrives_before_stream_end(streaming_server):
    client, release, first, _ = streaming_server
    turn = Turn(); heard = threading.Event()
    request = threading.Thread(target=client.chat_turn, args=(b'wav', turn, True),
                               kwargs={'on_audio': lambda a: heard.set()})
    request.start()
    assert first.wait(1)
    assert heard.wait(.3), 'First SSE frame was buffered until stream completion'
    assert request.is_alive()
    release.set(); request.join(1)
    assert turn.done.is_set() and turn.error is None


def test_abort_wakes_a_blocked_network_read(streaming_server):
    client, release, _, _ = streaming_server
    turn = Turn(); heard = threading.Event()
    request = threading.Thread(target=client.chat_turn, args=(b'wav', turn, True),
                               kwargs={'on_audio': lambda a: heard.set()})
    request.start(); assert heard.wait(1)
    start = time.monotonic(); turn.abort()
    request.join(.5)
    assert not request.is_alive()
    assert time.monotonic() - start < .5
    assert turn.error is None
    release.set()


def test_requests_are_serialized(streaming_server):
    client, release, first, active = streaming_server
    a, b = Turn(), Turn()
    threads = [threading.Thread(target=client.tts, args=('OK', t)) for t in (a, b)]
    for thread in threads: thread.start()
    assert first.wait(1)
    time.sleep(.05)
    assert active[0] == 1
    release.set()
    for thread in threads: thread.join(1)
    assert active[1] == 1
    assert a.done.is_set() and b.done.is_set()


def test_parser_handles_fragmented_crlf_and_utf8():
    payload = ('data: ' + json.dumps({'text': 'héllo'}, ensure_ascii=False) + '\r\n\r\ndata: [DONE]\r\n\r\n').encode()
    pieces = iter(bytes([b]) for b in payload)
    response = SimpleNamespace(read1=lambda n: next(pieces, b''))
    assert list(sse_events(response)) == [{'text': 'héllo'}]


def test_invalid_audio_is_an_error_not_a_hang():
    c = ServerClient()
    data = b'data: {"choices":[{"delta":{"audio":{"data":"bad"}}}]}\n\n'
    pieces = iter([data, b''])
    c._post = lambda p: SimpleNamespace(read1=lambda n: next(pieces, b''), close=lambda: None)
    turn = Turn(); c.chat_turn(b'wav', turn, True)
    assert turn.error and turn.done.is_set()


def test_audio_decoders_and_wav_roundtrip():
    import io, wave
    modern = extract_audio(audio_event()['choices'][0]['delta'])
    assert len(modern) == 1440
    raw = np.array([.1, -.2], dtype='<f4')
    legacy = extract_audio({'audio_chunk': {'data': base64.b64encode(raw).decode()}})
    np.testing.assert_allclose(legacy, raw)
    with wave.open(io.BytesIO(wav16k_bytes(raw))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (16000, 1, 2)


def test_playback_preserves_samples_and_releases_device(monkeypatch):
    written, closed = [], []
    class Output:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def write(self, a): written.append(a.copy())
        def stop(self): pass
        def close(self): closed.append(True)
    monkeypatch.setattr('audio_io.sd.OutputStream', Output)
    chunks = [np.arange(1440, dtype=np.float32) / 1440 for _ in range(10)]
    references = []
    player = Player()
    player.native_rate = 24000
    player.native_channels = 1
    player.play_chunks(iter(chunks), on_play=lambda a: references.append(a.copy()))
    np.testing.assert_array_equal(np.concatenate(written), np.concatenate(chunks))
    np.testing.assert_array_equal(np.concatenate(references), np.concatenate(chunks))
    assert closed == [True]


def test_playback_exception_releases_device(monkeypatch):
    closed = []
    class Output:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def write(self, a): raise RuntimeError('device disconnected')
        def close(self): closed.append(True)
    monkeypatch.setattr('audio_io.sd.OutputStream', Output)
    with pytest.raises(RuntimeError): Player().play_chunks([np.ones(1440)])
    assert closed == [True]


class FakeMic:
    device = 1
    def __init__(self): self.q = queue.Queue()
    def start(self): pass
    def stop(self): pass
    def drain(self):
        while not self.q.empty(): self.q.get_nowait()


class FakeFront:
    level = .1
    aec_error = None
    def __init__(self): self.calls = 0; self.speaking = False; self.references = []
    def warmup(self): pass
    def reset(self): self.speaking = False
    def clear_playback(self): pass
    def push_far(self, audio, rate=16000): self.references.append((len(audio), rate))
    def update(self, audio):
        self.calls += 1
        hot = bool(np.max(audio) > .05)
        started, ended = hot and not self.speaking, not hot and self.speaking
        self.speaking = hot
        return hot, started, ended, audio


class FakePlayer:
    def __init__(self): self.playing = False; self.stops = 0
    def stop(self): self.stops += 1; self.playing = False
    def play_chunks(self, chunks, cancel=None, on_start=None, on_play=None, on_written=None):
        started = False
        try:
            for audio in chunks:
                if cancel.is_set(): break
                if not started:
                    started = True; self.playing = True; on_start()
                on_play(audio)
                if on_written is not None:on_written(audio)
                time.sleep(.005)
        finally: self.playing = False


class FakeClient:
    base = 'http://chat'
    def __init__(self, fail=False): self.calls = []; self.fail = fail
    def health(self): return False
    def chat_turn(self, wav, turn, first, on_audio=None):
        self.calls.append(first)
        if self.fail: raise RuntimeError('simulated disconnect')
        for _ in range(40):
            if turn.cancel.wait(.01): break
            turn.text_parts.append('hello ')
            turn.audio_samples += 1440
            turn.ttft = .01
            on_audio(np.ones(1440, np.float32) * .1)
        turn.done.set()
    def tts(self, text, turn, max_tokens=128, on_audio=None):
        self.chat_turn(b'', turn, True, on_audio)


@pytest.fixture
def assistant():
    a = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=FakeClient())
    yield a
    a.close()


def test_mic_processes_barge_in_during_app_playback(assistant):
    a = assistant
    a._spawn(a._loop)
    job = a.submit_audio(np.ones(8000, np.float32) * .1)
    wait_for(lambda: a.status == 'speaking')
    a.mic.q.put(np.ones(3200, np.float32) * .1)
    wait_for(job.turn.cancel.is_set)
    assert a.status == 'recording'
    assert a.front.calls >= 1
    a.mic.q.put(np.zeros(3200, np.float32))
    wait_for(lambda: len(a.client.calls) == 2)
    assert a.client.calls == [True, True]


def test_request_failure_restores_listening(assistant):
    a = assistant; a.client.fail = True
    a.submit_audio(np.ones(8000, np.float32) * .1)
    wait_for(lambda: a._active is None)
    assert a.status == 'listening'
    assert 'simulated disconnect' in a.error
    assert a.first is True


def test_failed_partial_reply_remains_labelled_after_recovery(assistant):
    a = assistant
    def partial(wav, turn, first, on_audio=None):
        turn.text_parts.append('Here is the beginning')
        raise RuntimeError('simulated stream failure')
    a.client.chat_turn = partial
    a.submit_audio(np.ones(8000, np.float32)*.1)
    wait_for(lambda: a._active is None)
    assert a.state()['transcript'][-1].endswith(' [reply incomplete]')
    assert a.error == 'simulated stream failure'
    a.client = FakeClient()
    a.submit_audio(np.ones(8000, np.float32)*.1)
    wait_for(lambda: a._active is None)
    assert a.error is None
    assert any(line.endswith(' [reply incomplete]') for line in a.state()['transcript'])


def test_followup_preserves_context_without_background_asr(assistant):
    a = assistant
    for _ in range(2):
        a.submit_audio(np.ones(8000, np.float32) * .1)
        wait_for(lambda: a._active is None)
    assert a.client.calls == [True, False]


def test_reset_prevents_stale_reply_or_context_writes(assistant):
    a = assistant
    job = a.submit_audio(np.ones(8000, np.float32) * .1)
    wait_for(lambda: a.status == 'speaking')
    a.reset()
    wait_for(lambda: not a.player.playing)
    time.sleep(.03)
    assert job.turn.cancel.is_set()
    assert a.first is True and a.state()['transcript'] == []
    assert a.status == 'listening'


def test_transcript_ids_and_revisions_survive_retention(assistant):
    a = assistant
    for i in range(40): a._say(str(i))
    user_id = a._say('user: placeholder')
    a._say('assistant: reply')
    before = a.transcript_revision
    assert a._replace(user_id, 'user: actual words')
    state = a.state()
    assert state['transcript'][-2:] == ['user: actual words', 'assistant: reply']
    assert state['transcript_revision'] > before
    for i in range(40): a._say(str(i))
    assert not a._replace(user_id, 'user: stale')


def test_same_server_transcription_is_rejected():
    with pytest.raises(ValueError):
        Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=FakeClient(), asr_client=FakeClient())


def test_frontend_resamples_actual_playback():
    front = DuplexFrontend(aec=False)
    front.push_far(np.ones(24000, np.float32), rate=24000)
    # Bounded to 0.5 s at the CORRECT rate, then consumed consistently.
    assert len(front._far_buf) == 8000
    front._prob = lambda chunk: 0
    front.update(np.zeros(1600, np.float32))
    assert len(front._far_buf) == 6400


def test_holdoff_uses_audio_time_not_test_execution_speed():
    front = DuplexFrontend(aec=False, holdoff_s=.35)
    front._prob = lambda chunk: 1
    starts = 0
    for _ in range(32):
        front.push_far(np.ones(512, np.float32) * .1)
        starts += int(front.update(np.ones(512, np.float32) * .1)[1])
    assert starts == 1, 'Offline echo tests must not pass merely by executing quickly'


def test_mute_discards_partial_utterance(assistant):
    a = assistant; a._spawn(a._loop)
    a.mic.q.put(np.ones(3200, np.float32) * .1)
    wait_for(lambda: a.status == 'recording')
    a.toggle_mic(); a.mic.q.put(np.zeros(512, np.float32))
    wait_for(lambda: not a._capture_reset.is_set())
    a.toggle_mic(); a.mic.q.put(np.zeros(3200, np.float32))
    time.sleep(.05)
    assert not a.client.calls and a.status == 'listening'


def test_longer_preroll_retains_opening_speech_without_delaying_submission():
    class DelayedFront(FakeFront):
        def update(self,audio):
            self.calls+=1
            return 30<=self.calls<34,self.calls==30,self.calls==34,audio
    class Recorder(Assistant):
        def __init__(self,**kwargs):super().__init__(**kwargs);self.captured=[]
        def submit_audio(self,audio):self.captured.append((self.front.calls,audio.copy()))
    results=[]
    for milliseconds in [250,500]:
        app=Recorder(mic=FakeMic(),player=FakePlayer(),front=DelayedFront(),client=FakeClient(),pre_roll_ms=milliseconds)
        try:
            app._spawn(app._loop)
            for index in range(1,36):app.mic.q.put(np.full(320,.2 if index==6 else .1,np.float32))
            wait_for(lambda:len(app.captured)==1)
            results.append(app.captured[0])
        finally:app.close()
    assert results[0][0]==results[1][0]==34
    assert not np.any(results[0][1]==np.float32(.2))
    assert np.count_nonzero(results[1][1]==np.float32(.2))==320


def test_oversized_control_requests_receive_complete_error_responses(assistant):
    import urllib.request
    import urllib.error
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.assistant=assistant
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        for size in [4097,65537]*10:
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/mic',data=b'x'*size)
            with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=2)
            assert error.value.code==413
            assert json.load(error.value)==dict(error='Request too large')
        assert assistant.mic_on
    finally:server.shutdown();server.server_close();worker.join(1)

def test_interrupt_before_any_audio_returns_promptly(assistant):
    a = assistant
    def delayed(wav, turn, first, on_audio=None):
        turn.cancel.wait(2)
    a.client.chat_turn = delayed
    job = a.submit_audio(np.ones(8000, np.float32) * .1)
    wait_for(lambda: a.gen_start is not None)
    start = time.monotonic()
    a.interrupt()
    assert time.monotonic() - start < .1
    wait_for(job.produced.is_set, timeout=.3)
    assert a.status == 'listening'


def test_cancel_under_audio_backpressure(assistant):
    a = assistant
    def fast(wav, turn, first, on_audio=None):
        for _ in range(512):
            if turn.cancel.is_set(): return
            on_audio(np.ones(1440, np.float32))
    a.client.chat_turn = fast
    class StalledPlayer:
        def stop(self): pass
        def play_chunks(self, chunks, cancel, on_start, on_play, on_written=None): cancel.wait(2)
    a.player = StalledPlayer()
    job = a.submit_audio(np.ones(8000, np.float32) * .1)
    wait_for(lambda: job.chunks.full())
    a.interrupt()
    wait_for(job.produced.is_set, timeout=.3)
    assert a.first


@pytest.mark.parametrize('prefix_samples', [0, 1440])
def test_final_audio_arriving_at_empty_queue_timeout_is_played(assistant, prefix_samples):
    """Force producer completion between the consumer's Empty and done check."""
    a = assistant
    empty_observed = threading.Event()
    job = Job(id=1, epoch=a._epoch)

    class CompletionBoundaryQueue(queue.Queue):
        boundary_forced = False

        def get(self, block=True, timeout=None):
            if not self.boundary_forced and self.empty():
                self.boundary_forced = True
                # A timed get has just found no audio. The producer runs
                # before the consumer handles Empty, a legal scheduling order.
                empty_observed.set()
                assert job.produced.wait(1), 'Producer did not finish'
                raise queue.Empty
            return super().get(block, timeout)

    final_audio = np.linspace(-.2, .2, 2003, dtype=np.float32)
    prefix = np.full(prefix_samples, .1, dtype=np.float32)
    expected = np.concatenate([prefix, final_audio])

    def final_reply(wav, turn, first, on_audio=None):
        assert empty_observed.wait(1), 'Consumer did not reach empty queue'
        turn.audio_samples = len(expected)
        turn.text_parts.append('complete reply')
        on_audio(final_audio)
        turn.done.set()

    job.chunks = CompletionBoundaryQueue(maxsize=128)
    if prefix_samples:
        job.chunks.put(prefix)
    a.client.chat_turn = final_reply
    played = []
    a.front.push_far = lambda audio, rate=16000: played.append(audio.copy())
    a._active = job
    a._run_job(job)
    assert job.chunks.boundary_forced
    assert job.played_samples == len(expected)
    np.testing.assert_array_equal(np.concatenate(played), expected)
    assert job.chunks.empty()
    assert a._active is None and a.status == 'listening'


@pytest.mark.parametrize('fault', [
    'eof_after_audio', 'invalid_audio_after_audio',
    'server_error_after_audio', 'no_audio',
])
def test_application_recovers_from_real_client_stream_fault(fault, tmp_path):
    requests, responses, jobs, played = [], [], [], []

    def frame(event):
        return ('data: ' + json.dumps(event) + '\n\n').encode()

    healthy = (frame(audio_event('Recovered.')) +
               frame({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}) +
               b'data: [DONE]\n\n')
    broken = {
        'eof_after_audio': frame(audio_event('Partial.')),
        'invalid_audio_after_audio': frame(audio_event('Partial.')) + frame(
            {'choices': [{'delta': {'audio': {'data': 'bad'}}}]}),
        'server_error_after_audio': frame(audio_event('Partial.')) + frame(
            {'error': 'Injected generation failure'}),
        'no_audio': frame({'choices': [{'delta': {'content': 'Text only.'},
                                       'finish_reason': 'stop'}]}) + b'data: [DONE]\n\n',
    }[fault]

    class FaultHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = broken if len(requests) == 1 else healthy
            responses.append(body)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), FaultHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    a = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                  client=ServerClient(f'http://127.0.0.1:{server.server_port}', timeout=1))
    a.front.push_far = lambda audio, rate=16000: played.append(audio.copy())
    try:
        for index in range(3):
            job = a.submit_audio(np.full(8000, .1, np.float32))
            jobs.append(job)
            wait_for(lambda: a._active is None)
            assert a.status == 'listening'
            if index == 0:
                assert job.turn.error and a.error and a.first
                assert job.played_samples == (0 if fault == 'no_audio' else 1440)
            else:
                assert not job.turn.error and not a.error and not a.first
                assert job.turn.finish_reason == 'stop'
                assert job.played_samples == job.turn.audio_samples == 1440
        assert [body['reset_context'] for body in requests] == [True, True, False]
        count = 2880 if fault == 'no_audio' else 4320
        assert sum(map(len, played)) == count
        np.testing.assert_array_equal(np.concatenate(played),
                                      np.full(count, 1 / 32768, np.float32))
    finally:
        a.close()
        server.shutdown()
        server.server_close()
        worker.join(1)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path/'requests.json').write_text(json.dumps(requests, indent=2))
        for index, body in enumerate(responses):
            (tmp_path/f'response-{index}.sse').write_bytes(body)
        np.save(tmp_path/'played.npy', np.concatenate(played) if played else np.zeros(0))
        (tmp_path/'outcomes.json').write_text(json.dumps([
            dict(error=j.turn.error, generated_samples=j.turn.audio_samples,
                 played_samples=j.played_samples, first=j.first,
                 finish_reason=j.turn.finish_reason) for j in jobs], indent=2))
        (tmp_path/'diagnostics.json').write_text(json.dumps(a.diagnostics.recent(200), indent=2))


def test_tts_and_chat_use_distinct_modalities():
    from client import TTS_VOICE, CHAT_SYS, ASR_SYS
    client = ServerClient()
    assert client._payload(TTS_VOICE, 'hello', True, 64)['modalities'] == ['audio']
    assert client._payload(CHAT_SYS, [], True, 512)['modalities'] == ['text', 'audio']
    assert client._payload(ASR_SYS, [], True, 64, audio=False)['modalities'] == ['text']
    assert ServerClient('http://localhost:8080').base == 'http://127.0.0.1:8080'


def test_http_controls_and_origin_checks(assistant):
    import urllib.request, urllib.error
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.assistant = assistant
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base + '/api/state') as response:
            state = json.load(response)
        assert 'transcript_revision' in state
        req = urllib.request.Request(base + '/api/mic', data=b'', method='POST')
        with urllib.request.urlopen(req) as response: assert response.status == 200
        assert not assistant.mic_on
        req = urllib.request.Request(base + '/api/reset', data=b'', headers={'Origin': 'https://evil.invalid'})
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code == 403
        req = urllib.request.Request(base + '/api/mic', data=b'x' * 4097)
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code == 413
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(base + '/missing')
        assert error.value.code == 404
    finally:
        server.shutdown(); server.server_close(); worker.join(1)

def test_repetition_detector_does_not_reject_normal_long_answers():
    turn = Turn()
    turn.text_parts = ['The moon influences ocean tides. Its gravity attracts water, and the changing relative positions of Earth and the moon produce tidal cycles. Coastal geography changes their timing and height.']
    assert not turn.degenerate
    turn.text_parts = ['The following is a remote? ' * 8]
    assert turn.degenerate
    turn.text_parts = ['\u00a0' * 80]
    assert turn.degenerate

def test_native_rate_stereo_playback_keeps_duration_and_reference(monkeypatch):
    writes, refs = [], []
    class Output:
        latency = .04
        def __init__(self, **kwargs):
            assert kwargs['samplerate'] == 48000 and kwargs['channels'] == 2
        def start(self): pass
        def write(self, a): writes.append(a.copy()); return False
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr('audio_io.sd.OutputStream', Output)
    player = Player()
    player.native_rate = 48000; player.native_channels = 2
    source = np.linspace(-.5, .5, 1440, dtype=np.float32)
    player.play_chunks([source], on_play=lambda a:refs.append(a.copy()))
    device = np.concatenate(writes)
    assert device.shape == (2880, 2)
    np.testing.assert_array_equal(device[::2, 0], source)
    np.testing.assert_array_equal(device[:, 0], device[:, 1])
    np.testing.assert_array_equal(np.concatenate(refs), source)
    assert player.stats['latency_s'] == .04


def test_wasapi_selection_preserves_the_default_endpoint(monkeypatch):
    from audio_io import preferred_device
    devices = [dict(name='Monitor HDMI', hostapi=0, max_output_channels=2),
               dict(name='Laptop Speakers', hostapi=1, max_output_channels=2),
               dict(name='Monitor HDMI extended', hostapi=1, max_output_channels=2)]
    monkeypatch.setattr('audio_io.sd.default.device', [0, 0])
    monkeypatch.setattr('audio_io.sd.query_devices', lambda i=None: devices if i is None else devices[i])
    monkeypatch.setattr('audio_io.sd.query_hostapis', lambda i: {'name':['MME','Windows WASAPI'][i]})
    assert preferred_device('output') == 2


def test_diagnostics_are_bounded_and_do_not_store_audio(tmp_path):
    from diagnostics import Diagnostics
    path = tmp_path / 'events.jsonl'
    diag = Diagnostics(path)
    for i in range(250): diag.event('turn_submit', turn_id=i, input_s=.5)
    assert len(diag.recent(1000)) == 200
    assert diag.recent(1)[0]['turn_id'] == 249
    diag.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 250
    assert all(set(row) == {'ts','pid','event','turn_id','input_s'} for row in records)


def test_capture_gain_preserves_loud_speech_without_flattening(monkeypatch):
    from audio_io import Mic
    monkeypatch.setattr('audio_io.sd.query_devices', lambda *args: dict(default_samplerate=16000, max_input_channels=1, hostapi=0))
    monkeypatch.setattr('audio_io.sd.query_hostapis', lambda *args: {'name': 'Windows WASAPI'})
    mic = Mic(device=0)
    assert mic.gain == 1.0
    source = np.linspace(-.8, .8, 320, dtype=np.float32)
    mic._cb(source[:, None], 320, None, False)
    np.testing.assert_allclose(mic.q.get(), source, atol=1e-6)
    boosted = Mic(device=0, gain=6)
    boosted._cb(source[:, None], 320, None, False)
    result = boosted.q.get()
    assert np.max(abs(result)) <= .951
    np.testing.assert_allclose(result, source * (.95/.8), atol=1e-6)
    assert boosted.clipped_fraction == 0


def test_echo_cancellation_continues_through_cancelled_playback_tail():
    from duplex import DuplexFrontend
    class AEC:
        def __init__(self): self.frames = []
        def process(self, near, far):
            self.frames.append(far.copy())
            return near * 0
    front = DuplexFrontend(aec=False)
    front.aec = AEC()
    front._prob = lambda chunk: 0.0
    front.push_far(np.ones(320, np.float32) * .1)
    front.update(np.ones(320, np.float32) * .05)
    front.clear_playback()
    count = len(front.aec.frames)
    _, started, _, clean = front.update(np.ones(320, np.float32) * .05)
    assert len(front.aec.frames) > count
    assert all(not np.any(frame) for frame in front.aec.frames[count:])
    assert not started and not np.any(clean)
    for _ in range(16): front.update(np.zeros(320, np.float32))
    assert not front._far_active

def test_quiet_echo_residual_does_not_interrupt_but_quiet_user_speech_can():
    class ResidualAEC:
        def __init__(self,ratio):self.ratio=ratio
        def process(self,near,far):return near*self.ratio
    def run(raw_level,ratio):
        front=DuplexFrontend(aec=False,holdoff_s=0)
        front.aec=ResidualAEC(ratio);front._prob=lambda samples:.9
        started=False
        for _ in range(5):
            front.push_far(np.ones(512,np.float32)*.1)
            _,onset,_,_=front.update(np.ones(512,np.float32)*raw_level)
            started|=onset
        return started,front.residual_rejections
    assert run(.028,.04)[0] is False
    assert run(.028,.04)[1]>0
    assert run(.002,1)[0] is True
    assert run(.05,.5)[0] is True


def test_correlation_reference_uses_consumed_audio_and_keeps_render_gaps():
    front = DuplexFrontend(aec=False)
    front._prob = lambda samples: 0.0
    render = np.linspace(-.2, .2, 640, dtype=np.float32)
    front.push_far(render)
    assert len(front._far_history) == 0
    front.update(np.zeros(320, np.float32))
    assert len(front._far_history) == 0
    np.testing.assert_array_equal(front._far_vad_buf, render[:320])
    front.clear_playback()
    front.update(np.zeros(320, np.float32))
    np.testing.assert_array_equal(front._far_history, np.concatenate((render[:320], np.zeros(192))))
    # A subsequent chunk follows the silent interval, not the aborted queue.
    front.push_far(render[:320])
    front.update(np.zeros(320, np.float32))
    front.update(np.zeros(320, np.float32))
    np.testing.assert_array_equal(front._far_history, np.concatenate((render[:320], np.zeros(320), render[:320], np.zeros(64))))


def test_echo_correlation_is_aligned_and_invariant_to_input_block_sizes():
    rng=np.random.default_rng(20261020)
    raw=rng.normal(0,.05,16000).astype(np.float32)
    far=rng.normal(0,.05,16000).astype(np.float32)
    traces=[]
    for frame in [160,320,512,1024]:
        front=DuplexFrontend(aec=False);trace=[]
        def probability(chunk):
            end=front._sample_clock
            expected=DuplexFrontend._echo_similarity(SimpleNamespace(_near_history=raw[:end][-1600:],_far_history=far[:end][-12000:]))
            np.testing.assert_array_equal(front._near_history,raw[:end][-1600:])
            np.testing.assert_array_equal(front._far_history,far[:end][-12000:])
            np.testing.assert_array_equal(front._clean_history,raw[:end][-1600:])
            assert abs(front.echo_score-expected)<1e-7
            trace.append(front.echo_score)
            return 0.
        front._prob=probability
        for offset in range(0,len(raw),frame):
            front.push_far(far[offset:offset+frame]);front.update(raw[offset:offset+frame])
        traces.append(trace)
    for trace in traces[1:]:np.testing.assert_array_equal(trace,traces[0])


def test_frontend_reset_preserves_continuous_aec_and_flushes_partial_decision():
    front=DuplexFrontend(aec=False);front._prob=lambda chunk:0.
    raw=np.linspace(-.1,.1,1700,dtype=np.float32);far=raw[::-1].copy()
    front.push_far(far[:700]);_,_,_,first=front.update(raw[:700])
    assert len(first)==640 and front._sample_clock==512
    front._onset_bits=0b11000
    front.reset()
    assert front._onset_bits==0
    assert front._sample_clock==640 and len(front._mic_buf)==60
    np.testing.assert_array_equal(front._near_history,raw[:640])
    np.testing.assert_array_equal(front._far_history,far[:640])
    front.push_far(far[700:]);_,_,_,second=front.update(raw[700:])
    np.testing.assert_array_equal(np.concatenate((first,second)),raw[:1600])
    np.testing.assert_array_equal(front._far_history,far[:1152])


def onset_sequence(probabilities,**kwargs):
    front=DuplexFrontend(aec=False,**kwargs);values=iter(probabilities);front._prob=lambda chunk:next(values,0.)
    samples=((len(probabilities)*512+319)//320)*320;starts=[]
    for _ in range(samples//320):
        front.push_far(np.full(320,.1,np.float32))
        if front.update(np.full(320,.1,np.float32))[1]:starts.append(front._sample_clock)
    return starts


@pytest.mark.parametrize('window,expected',[(5,[2560]),(3,[])])
def test_confirmation_can_bridge_brief_masking_without_reducing_evidence(window,expected):
    assert onset_sequence([.9,.9,.1,.1,.9],holdoff_s=0,onset_window_chunks=window)==expected


def test_isolated_pairs_and_currently_vetoed_frames_cannot_confirm_speech():
    assert onset_sequence([.9,.9,.1,.1,.1,.9,.9],holdoff_s=0)==[]
    # Three earlier frames precede holdoff; a later cold frame cannot reuse
    # their evidence to trigger an interruption as holdoff expires.
    assert onset_sequence([.9,.9,.9,.1,.1],holdoff_s=.128)==[]


def test_future_queued_speech_cannot_veto_present_microphone_speech():
    rng = np.random.default_rng(20261016)
    near = rng.normal(0, .05, 1600).astype(np.float32)
    present_render = rng.normal(0, .05, 1600).astype(np.float32)
    front = DuplexFrontend(aec=False, holdoff_s=0)
    front._prob = lambda samples: .99
    # A queued future repetition of the near-end signal must not match now.
    front.push_far(np.concatenate((present_render, near)))
    _, started, _, _ = front.update(near)
    assert started
    assert front.echo_score < .6


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_stream_preserves_completion_reason_and_generated_audio(monkeypatch, reason):
    import io
    events = [audio_event('Hello'), {'choices': [{'delta': {}, 'finish_reason': reason}]}]
    encoded = ''.join('data: '+json.dumps(event)+'\n\n' for event in events) + 'data: [DONE]\n\n'
    client = ServerClient()
    monkeypatch.setattr(client, '_post', lambda payload: io.BytesIO(encoded.encode()))
    turn = Turn(); chunks = []
    client._stream({}, turn, on_audio=chunks.append)
    assert turn.error is None and turn.done.is_set()
    assert turn.finish_reason == reason and turn.truncated == (reason == 'length')
    assert turn.text == 'Hello' and len(np.concatenate(chunks)) == 1440


def test_length_limited_reply_is_labelled_and_followup_preserves_context(assistant):
    a = assistant
    original = a.client.chat_turn
    def limited(wav, turn, first, on_audio=None):
        original(wav, turn, first, on_audio)
        turn.finish_reason = 'length'
    a.client.chat_turn = limited
    for _ in range(2):
        a.submit_audio(np.ones(8000, np.float32) * .1)
        wait_for(lambda: a._active is None)
    assert a.client.calls == [True, False]
    assert a.state()['transcript'][-1].endswith(' [reply reached length limit]')
    assert a.error is None
    assert any(event.get('finish_reason') == 'length' for event in a.diagnostics.recent())
