import io
import json
import pathlib
import sys
import time
import urllib.error

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from agent import AgentClient, AgentError
from app import Assistant
from client import Turn
from test_realtime import FakeMic, FakePlayer, FakeFront


def response(value):
    class Response(io.BytesIO):
        pass
    return Response(json.dumps(value).encode())


def test_agent_searches_and_passes_results_to_model(monkeypatch):
    requests = []
    def open_url(req, timeout):
        requests.append(req)
        if 'api.search.brave.com' in req.full_url:
            return response({'web': {'results': [
                {'url': 'https://example.org/article', 'title': 'Article', 'description': 'Evidence'},
                {'url': 'javascript:alert(1)', 'title': 'Bad'}]}})
        if sum('opencode.ai' in r.full_url for r in requests) == 1:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"latest weather"}'}}]}}]})
        payload = json.loads(req.data)
        assert payload['messages'][-1]['tool_call_id'] == 'call-1'
        assert 'Evidence' in payload['messages'][-1]['content']
        return response({'choices': [{'message': {'content': 'Here is the report.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    phases = []
    client = AgentClient('secret-model-key', search_key='secret-search-key')
    answer, sources = client.respond('What is the weather?', [], Turn(), lambda active: phases.append(active))
    assert answer == 'Here is the report.'
    assert sources == [{'title': 'Article', 'url': 'https://example.org/article', 'description': 'Evidence'}]
    assert phases == [True, False]
    assert requests[0].get_header('Authorization') == 'Bearer secret-model-key'
    assert requests[0].get_header('User-agent') == 'Turnwave/0.1'
    assert requests[0].get_header('X-opencode-session') == requests[2].get_header('X-opencode-session')
    assert requests[1].get_header('X-subscription-token') == 'secret-search-key'
    assert 'secret' not in requests[1].full_url


def test_agent_without_search_does_not_offer_tool(monkeypatch):
    def open_url(req, timeout):
        body = json.loads(req.data)
        assert 'tools' not in body
        assert 'search is unavailable' in body['messages'][0]['content']
        return response({'choices': [{'message': {'content': 'Hello.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    assert AgentClient('key').respond('Hi', [], Turn()) == ('Hello.', [])


def test_zen_uses_its_endpoint_and_resets_conversation_session(monkeypatch):
    requests = []
    def open_url(req, timeout):
        requests.append(req)
        return response({'choices': [{'message': {'content': 'Hello.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', provider='zen')
    client.respond('Hi', [], Turn())
    client.reset()
    client.respond('Hi again', [], Turn())
    assert requests[0].full_url == 'https://opencode.ai/inference/openai/v1/chat/completions'
    assert requests[0].get_header('X-opencode-session') != requests[1].get_header('X-opencode-session')


def test_agent_http_failure_does_not_expose_body_or_key(monkeypatch):
    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, 'secret', {}, io.BytesIO(b'secret'))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(AgentError, match='HTTP 401') as error:
        AgentClient('secret').respond('Hi', [], Turn())
    assert 'secret' not in str(error.value)


def test_response_close_error_does_not_hide_completed_reply(monkeypatch):
    class FailingClose(io.BytesIO):
        def close(self):
            super().close()
            raise OSError('close failed')
    payload = json.dumps({'choices': [{'message': {'content': 'Hello.'}}]}).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: FailingClose(payload))
    turn = Turn()
    assert AgentClient('key').respond('Hi', [], turn) == ('Hello.', [])
    assert turn._response is None


def test_usage_limit_reports_actionable_message(monkeypatch):
    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, 'limit', {}, io.BytesIO(b''))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(AgentError, match='usage limit'):
        AgentClient('key').respond('Hi', [], Turn())


def test_cancelled_agent_does_not_send(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: pytest.fail('Unexpected request'))
    turn = Turn(); turn.abort()
    assert AgentClient('key').respond('Hi', [], turn) is None


def test_missing_key_produces_actionable_error():
    client = AgentClient(None)
    assert 'OPENCODE_GO_API_KEY' in client.configuration_error
    with pytest.raises(AgentError, match='OPENCODE_GO_API_KEY'):
        client.respond('Hi', [], Turn())


def test_ddg_search_parses_results_and_filters_unsafe_urls(monkeypatch):
    page = (b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fnews">'
            b'<b>Latest</b> News</a><a class="result__snippet">First <b>snippet</b>.</a>'
            b'<a class="result__a" href="javascript:alert(1)">Bad</a>')
    class RawResponse(io.BytesIO):
        pass
    def open_url(req, timeout):
        if 'duckduckgo.com' in req.full_url:
            return RawResponse(page)
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            assert body['reasoning_effort'] == 'low'
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"space news"}'}}]}}]})
        return response({'choices': [{'message': {'content': 'Found Latest News.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg')
    assert client.search_enabled and client.search_provider == 'ddg'
    answer, sources = client.respond('What is new?', [], Turn())
    assert answer == 'Found Latest News.'
    assert sources == [{'title': 'Latest News', 'url': 'https://example.org/news',
                        'description': 'First snippet.'}]


def test_reasoning_effort_falls_back_when_model_rejects_parameter(monkeypatch):
    efforts = []
    def open_url(req, timeout):
        body = json.loads(req.data)
        efforts.append(body.get('reasoning_effort'))
        if len(efforts) == 1:
            raise urllib.error.HTTPError(req.full_url, 400, 'bad param', {}, io.BytesIO(b''))
        return response({'choices': [{'message': {'content': 'Fallback succeeded.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    assert AgentClient('key').respond('Hi', [], Turn()) == ('Fallback succeeded.', [])
    assert efforts == ['low', None]


class LocalVoice:
    base = 'http://local'
    def health(self): return False
    def asr_text(self, wav, turn):
        turn.text_parts.append('What is new?')
        turn.done.set()
        return turn.text
    def tts(self, text, turn, max_tokens=128, on_audio=None):
        assert text == 'A short answer.'
        turn.audio_samples += 480
        turn.ttft = .01
        on_audio(np.ones(480, np.float32) * .1)
        turn.done.set()


class FakeAgent:
    search_enabled = True
    def __init__(self): self.histories = []
    def respond(self, text, history, turn, on_search):
        self.histories.append(history)
        on_search(True)
        on_search(False)
        return 'A short answer.', [{'title': 'Source', 'url': 'https://example.org'}]


def wait_for(predicate):
    limit = time.monotonic() + 3
    while time.monotonic() < limit:
        if predicate(): return
        time.sleep(.01)
    pytest.fail('Timed out')


def test_voice_agent_transcribes_speaks_remembers_and_resets():
    agent = FakeAgent()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set())
        state = app.state()
        assert state['agent_mode'] and state['search_enabled']
        assert state['agent_sources'][0]['url'] == 'https://example.org'
        assert 'user: What is new?' in state['transcript']
        assert 'assistant: A short answer.' in state['transcript']
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: second.playback_closed.is_set())
        assert agent.histories[1] == [{'role': 'user', 'content': 'What is new?'},
                                      {'role': 'assistant', 'content': 'A short answer.'}]
        app.reset()
        assert app.state()['agent_sources'] == [] and app._agent_history == []
    finally:
        app.close()


def test_interrupted_agent_keeps_request_but_not_unspoken_reply():
    import threading
    entered = threading.Event()
    class WaitingAgent(FakeAgent):
        def respond(self, text, history, turn, on_search):
            entered.set()
            turn.cancel.wait(2)
            return None
    local = LocalVoice()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=local, agent=WaitingAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert entered.wait(1)
        app.interrupt()
        wait_for(lambda: job.generation_closed.is_set() and job.playback_closed.is_set())
        assert job.turn.cancel.is_set() and job.played_samples == 0
        assert app._agent_history == [{'role': 'user', 'content': 'What is new?'}]
        assert app.state()['agent_sources'] == []
        app.reset()
        assert app._agent_history == []
    finally:
        app.close()


def test_next_agent_turn_receives_interrupted_user_request_without_answer():
    import threading
    entered = threading.Event()
    class InterruptThenReply(FakeAgent):
        def respond(self, text, history, turn, on_search):
            self.histories.append(history)
            if len(self.histories) == 1:
                entered.set()
                turn.cancel.wait(2)
                return None
            return 'A short answer.', []
    agent = InterruptThenReply()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        first = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert entered.wait(1)
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: second.playback_closed.is_set() and app._active is None)
        assert first.turn.cancel.is_set()
        assert agent.histories[1] == [{'role': 'user', 'content': 'What is new?'}]
        assert app._agent_history == [{'role': 'user', 'content': 'What is new?'},
                                      {'role': 'user', 'content': 'What is new?'},
                                      {'role': 'assistant', 'content': 'A short answer.'}]
    finally:
        app.close()


def test_failed_agent_request_keeps_only_recognized_input():
    class FailingAgent(FakeAgent):
        def respond(self, text, history, turn, on_search):
            raise AgentError('Provider unavailable')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=FailingAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        assert job.turn.error == 'Provider unavailable'
        assert app._agent_history == [{'role': 'user', 'content': 'What is new?'}]
    finally:
        app.close()
