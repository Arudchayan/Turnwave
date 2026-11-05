import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app import Assistant, Job, configured_voice_variants, voice_context_endpoints
from lfm3_variants import VoiceVariants, local_base
from test_realtime import FakeMic, FakePlayer, FakeFront, wait_for


class Opener:
    def __init__(self):
        self.models = {'http://127.0.0.1:8080/v1/models': '/models/standard.gguf',
                       'http://127.0.0.1:18087/v1/models': '/models/research.gguf'}
        self.calls = []
    def open(self, request, timeout):
        self.calls.append(request.full_url)
        return io.BytesIO(json.dumps({'data': [{'id': self.models[request.full_url]}]}).encode())


class Voice:
    def __init__(self, base, timeout=60):
        self.base, self.timeout, self.closed, self.calls = base, timeout, False, []
        self._custom_voice_params = False
    def health(self): return True
    def close(self): self.closed = True
    def set_voice_params(self, **values): self._custom_voice_params = True; self.params = values
    def asr_text(self, wav, turn):
        self.calls.append('asr')
        return 'Hello'
    def tts(self, text, turn, max_tokens=128, on_audio=None):
        self.calls.append('tts')
        turn.audio_samples += 480
        on_audio(np.ones(480, dtype=np.float32) * .1)
        turn.done.set()
    def chat_text(self, text, turn, first, on_audio=None):
        self.calls.append('chat')
        turn.text_parts.append('Local reply')
        turn.audio_samples += 480
        on_audio(np.ones(480, dtype=np.float32) * .1)
        turn.done.set()


class Agent:
    search_enabled = False
    def __init__(self): self.resets = 0
    def reset(self): self.resets += 1
    def respond(self, text, history, turn, on_search=None):
        return 'Cloud answer.', []


def make_app(monkeypatch):
    monkeypatch.setattr('lfm3_client.LFM3Client', Voice)
    opener = Opener()
    config = VoiceVariants('http://localhost:8080/', 'http://127.0.0.1:18087',
                           'standard.gguf', 'research.gguf', opener=opener)
    voice = Voice('http://127.0.0.1:8080')
    agent = Agent()
    assistant = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=voice,
                          agent=agent, variant_config=config)
    assistant.status = 'listening'
    return assistant, voice, agent, opener


@pytest.mark.parametrize('url', ['https://localhost:8080', 'http://192.168.0.1:8080',
                                 'http://127.0.0.1:8080/path', 'http://localhost:8080/?a=1',
                                 'http://localhost:8080/#x', 'http://user@localhost:8080',
                                 'http://localhost', 'http://127.1:8080'])
def test_variant_urls_reject_noncanonical_or_nonlocal_endpoints(url):
    with pytest.raises(ValueError): local_base(url)


def test_model_identity_requires_one_matching_reported_identifier():
    opener = Opener()
    config = VoiceVariants('http://localhost:8080', 'http://localhost:18087',
                           'standard.gguf', 'research.gguf', opener=opener)
    assert config.verify('research') == '/models/research.gguf'
    opener.models['http://127.0.0.1:18087/v1/models'] = '/models/standard.gguf'
    with pytest.raises(ValueError, match='unexpected'):
        config.verify('research')


def test_model_identity_does_not_follow_redirects():
    requests = []
    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args): pass
    target = ThreadingHTTPServer(('127.0.0.1', 0), Target)
    class Redirect(Target):
        def do_GET(self):
            self.send_response(302)
            self.send_header('Location', f'http://127.0.0.1:{target.server_port}/private')
            self.end_headers()
    source = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (source, target)]
    for thread in threads: thread.start()
    try:
        config = VoiceVariants(f'http://127.0.0.1:{source.server_port}',
                               f'http://127.0.0.1:{target.server_port}')
        with pytest.raises(ValueError, match='unavailable'):
            config.verify('standard')
        assert requests == []
    finally:
        for server in (source, target): server.shutdown(); server.server_close()
        for thread in threads: thread.join()


def test_configured_endpoints_are_both_reserved(monkeypatch):
    monkeypatch.setenv('LOCALVOICE_BACKEND', 'lfm3')
    monkeypatch.setenv('LOCALVOICE_SERVER', 'http://localhost:8080')
    monkeypatch.setenv('LOCALVOICE_LFM3_RESEARCH_SERVER', 'http://localhost:18087')
    monkeypatch.delenv('LOCALVOICE_ASR_SERVER', raising=False)
    config = configured_voice_variants()
    assert voice_context_endpoints(config) == {'http://127.0.0.1:8080', 'http://127.0.0.1:18087'}
    monkeypatch.setenv('LOCALVOICE_ASR_SERVER', 'http://localhost:8082')
    with pytest.raises(ValueError, match='separate ASR'):
        configured_voice_variants()


def test_switch_routes_cloud_asr_and_tts_and_resets_context(monkeypatch):
    app, old, agent, _ = make_app(monkeypatch)
    try:
        app._agent_history.append({'role': 'user', 'content': 'prior turn'})
        result = app.set_voice_variant('research')
        assert result['reported_model_id'] == '/models/research.gguf'
        assert old.closed and app.client.base == 'http://127.0.0.1:18087'
        assert app.agent is agent and app._agent_history == [] and agent.resets == 1
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and job.generation_closed.is_set())
        assert app.client.calls == ['asr', 'tts']
        assert app.state()['voice_variant'] == 'research'
        app.set_agent_mode('local')
        job = app.submit_text('hello')
        wait_for(lambda: job.playback_closed.is_set() and job.generation_closed.is_set())
        assert app.client.calls[-1] == 'chat'
    finally:
        app.close()


def test_busy_or_unverified_target_does_not_close_or_reset_current_session(monkeypatch):
    app, old, agent, opener = make_app(monkeypatch)
    try:
        app._agent_history.append({'role': 'user', 'content': 'keep me'})
        app._active = Job()
        with pytest.raises(RuntimeError, match='current conversation'):
            app.set_voice_variant('research')
        assert not old.closed and app._agent_history and agent.resets == 0
        app._active = None
        opener.models['http://127.0.0.1:18087/v1/models'] = '/models/standard.gguf'
        with pytest.raises(ValueError, match='unexpected'):
            app.set_voice_variant('research')
        assert app.client._client is old and not old.closed and app._agent_history and agent.resets == 0
    finally:
        app._active = None
        app.close()


def test_identity_change_before_turn_blocks_voice_processing(monkeypatch):
    app, old, _, opener = make_app(monkeypatch)
    try:
        opener.models['http://127.0.0.1:8080/v1/models'] = '/models/research.gguf'
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and job.generation_closed.is_set())
        assert not old.calls
        assert 'verification failed' in job.turn.error
        assert app.state()['voice_variant'] is None
    finally:
        app.close()


def test_reselecting_changed_server_clears_stale_model_identity(monkeypatch):
    app, old, _, opener = make_app(monkeypatch)
    try:
        opener.models['http://127.0.0.1:8080/v1/models'] = '/models/research.gguf'
        with pytest.raises(ValueError, match='unexpected'):
            app.set_voice_variant('standard')
        assert app.state()['voice_variant'] is None
        assert not old.closed
    finally:
        app.close()


def test_identity_change_between_asr_and_tts_prevents_synthesis(monkeypatch):
    app, old, agent, opener = make_app(monkeypatch)
    try:
        def respond(text, history, turn, on_search=None):
            opener.models['http://127.0.0.1:8080/v1/models'] = '/models/research.gguf'
            return 'Answer', []
        agent.respond = respond
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and job.generation_closed.is_set())
        assert old.calls == ['asr']
        assert 'verification failed' in job.turn.error
        assert app.state()['voice_variant'] is None
    finally:
        app.close()
