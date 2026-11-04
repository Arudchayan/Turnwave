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


@pytest.mark.parametrize('modality,expected', [
    ('voice', 'spoken and transcribed locally'),
    ('text', 'typed, not spoken'),
])
def test_agent_receives_accurate_input_modality(monkeypatch, modality, expected):
    def open_url(req, timeout):
        messages = json.loads(req.data)['messages']
        assert expected in messages[0]['content']
        if modality == 'voice':
            assert 'not the raw audio' in messages[0]['content']
        return response({'choices': [{'message': {'content': 'Hello.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    assert AgentClient('key').respond('Hello', [], Turn(), input_modality=modality) == ('Hello.', [])


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


def test_zen_error_identifies_selected_provider(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response({'choices': []}))
    with pytest.raises(AgentError, match='OpenCode Zen returned an invalid response'):
        AgentClient('key', provider='zen').respond('Hi', [], Turn())


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


@pytest.mark.parametrize('provider', ['ddg', 'brave'])
def test_search_results_respect_site_restriction_and_reject_malformed_urls(monkeypatch, provider):
    results = [
        ('https://docs.example.org/release', 'Correct'),
        ('https://evil-example.org/', 'Wrong domain'),
        ('https://example.org@evil.org/', 'Credentials'),
        ('https://[broken', 'Malformed'),
    ]
    if provider == 'ddg':
        body = ''.join(f'<a class="result__a" href="{url}">{title}</a>'
                       f'<a class="result__snippet">Details</a>' for url, title in results).encode()
        monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: io.BytesIO(body))
    else:
        payload = {'web': {'results': [{'url': url, 'title': title, 'description': 'Details'}
                                      for url, title in results]}}
        monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response(payload))
    client = AgentClient('key', search_provider=provider, search_key='search-key')
    found = client.search('release site:example.org', Turn())
    assert [item['url'] for item in found] == ['https://docs.example.org/release']


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


def test_sources_remain_attached_to_their_reply_across_followups_and_reset():
    class FirstSourceAgent(FakeAgent):
        def __init__(self): self.calls = 0
        def respond(self, text, history, turn, on_search):
            self.calls += 1
            sources = ([{'title': 'First source', 'url': 'https://example.org/first'}]
                       if self.calls == 1 else [])
            return 'A short answer.', sources
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                    client=LocalVoice(), agent=FirstSourceAgent())
    try:
        first = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: first.playback_closed.is_set() and app._active is None)
        state = app.state()
        first_reply = next(m for m in state['messages'] if m['id'] == first.reply_id)
        assert first_reply['sources'] == [{'title': 'First source', 'url': 'https://example.org/first'}]
        first_reply['sources'][0]['title'] = 'Changed in caller'
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: second.playback_closed.is_set() and app._active is None)
        messages = app.state()['messages']
        assert next(m for m in messages if m['id'] == first.reply_id)['sources'][0]['title'] == 'First source'
        assert next(m for m in messages if m['id'] == second.reply_id)['sources'] == []
        app.set_agent_mode('local')
        assert app.state()['messages'] == []  # Switching mode starts a new conversation.
        app.reset()
        assert app.state()['messages'] == []
    finally:
        app.close()


def test_cancelled_before_playback_keeps_sources_with_the_visible_draft():
    import threading
    entered = threading.Event()
    class WaitingVoice(LocalVoice):
        def tts(self, text, turn, max_tokens=128, on_audio=None):
            entered.set()
            turn.cancel.wait(2)
            turn.done.set()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                    client=WaitingVoice(), agent=FakeAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert entered.wait(2)
        app.interrupt()
        wait_for(lambda: job.playback_closed.is_set() and job.generation_closed.is_set())
        reply = next(m for m in app.state()['messages'] if m['id'] == job.reply_id)
        assert reply['text'].startswith('assistant: A short answer.') and job.played_samples == 0
        assert reply['sources'] == [{'title': 'Source', 'url': 'https://example.org'}]
    finally:
        app.close()


def test_app_distinguishes_transcribed_and_typed_turns():
    class ModalityAgent(FakeAgent):
        supports_input_modality = True
        def __init__(self): self.modalities = []
        def respond(self, text, history, turn, on_search, input_modality=None):
            self.modalities.append(input_modality)
            return 'A short answer.', []

    agent = ModalityAgent()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        voice = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: voice.playback_closed.is_set() and app._active is None)
        typed = app.submit_text('Hello')
        wait_for(lambda: typed.playback_closed.is_set() and app._active is None)
        assert agent.modalities == ['voice', 'text']
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


def test_local_tools_time_weather_memory_and_status(monkeypatch):
    calls_made = []
    def open_url(req, timeout):
        if 'geocoding-api.open-meteo.com' in req.full_url:
            return response({'results': [{'name': 'Basel', 'country': 'Switzerland', 'latitude': 47.55, 'longitude': 7.59}]})
        if 'api.open-meteo.com' in req.full_url:
            return response({'timezone': 'Europe/Zurich', 'current': {
                'time': '2026-11-02T17:15', 'temperature_2m': 8.0, 'relative_humidity_2m': 70,
                'apparent_temperature': 5.7, 'precipitation': 0.0, 'weather_code': 3, 'wind_speed_10m': 5.1},
                'daily': {'temperature_2m_max': [11.0], 'temperature_2m_min': [4.0]}})
        calls_made.append(json.loads(req.data))
        if len(calls_made) == 1:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 't1', 'type': 'function', 'function': {'name': 'get_time', 'arguments': '{"timezone":"UTC"}'}},
                {'id': 't2', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location":"Basel"}'}},
                {'id': 't3', 'type': 'function', 'function': {'name': 'remember', 'arguments': '{"fact":"User likes anime","category":"pref"}'}},
            ]}}]})
        if len(calls_made) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 't4', 'type': 'function', 'function': {'name': 'recall', 'arguments': '{"query":"anime"}'}},
                {'id': 't5', 'type': 'function', 'function': {'name': 'system_status', 'arguments': '{}'}},
            ]}}]})
        return response({'choices': [{'message': {'content': '**It is 8°C and overcast in Basel.**'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg', local_tools=True)
    answer, sources = client.respond('Time, weather in Basel, and status?', [], Turn())
    assert answer == 'It is 8°C and overcast in Basel.'
    assert any('Open-Meteo: Basel' in s['title'] for s in sources)
    assert client.last_trace['checks']['all_passed'] is False if 'all_passed' in client.last_trace['checks'] else client.last_trace['checks']['tool_calls_succeeded']
    assert [tc['tool'] for tc in client.last_trace['tool_calls']] == ['get_time', 'get_weather', 'remember', 'recall', 'system_status']


def test_ddg_empty_falls_back_to_wikipedia(monkeypatch):
    class RawResponse(io.BytesIO):
        pass
    def open_url(req, timeout):
        if 'duckduckgo.com' in req.full_url:
            return RawResponse(b'<html><body>No results</body></html>')
        if 'en.wikipedia.org' in req.full_url:
            return response({'query': {'search': [{'title': 'Jujutsu Kaisen', 'snippet': 'Japanese <b>manga</b> series'}]}} )
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'w1', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query":"Jujutsu Kaisen"}'}}
            ]}}]})
        return response({'choices': [{'message': {'content': 'Jujutsu Kaisen is a manga series.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg')
    answer, sources = client.respond('Tell me about Jujutsu Kaisen', [], Turn())
    assert answer == 'Web search was unavailable, so these results are limited to Wikipedia. Jujutsu Kaisen is a manga series.'
    assert sources == [{'title': 'Jujutsu Kaisen (Wikipedia)', 'url': 'https://en.wikipedia.org/wiki/Jujutsu_Kaisen', 'description': 'Japanese manga series'}]


def test_wikipedia_fallback_is_disclosed_to_the_model(monkeypatch):
    def open_url(req, timeout):
        if 'duckduckgo.com' in req.full_url:
            return io.BytesIO(b'<html>No usable search results</html>')
        if 'en.wikipedia.org' in req.full_url:
            return response({'query': {'search': [{'title': 'Python', 'snippet': 'A language'}]}})
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'w1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"latest Python release"}'}}]}}]})
        result = json.loads(body['messages'][-1]['content'])
        assert 'Wikipedia-only' in result['notice']
        assert 'not a current web search' in result['notice']
        assert result['results'][0]['url'].startswith('https://en.wikipedia.org/')
        return response({'choices': [{'message': {'content': 'I could not verify the latest release.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg')
    answer, sources = client.respond('Latest Python?', [], Turn())
    assert 'could not verify' in answer
    assert len(sources) == 1
    assert 'limited to Wikipedia' in client.last_trace['search_warning']


def test_site_restriction_is_not_replaced_with_unrelated_wikipedia(monkeypatch):
    def open_url(req, timeout):
        if 'duckduckgo.com' in req.full_url:
            return io.BytesIO(b'<html>No usable search results</html>')
        pytest.fail('A site-restricted query must not fall back to Wikipedia')
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg')
    result, sources = client._run_tool('web_search', '{"query":"latest Python site:python.org"}', Turn(), None)
    assert result['results'] == [] and sources == []
    assert 'site restriction' in result['notice']


def test_empty_search_and_fallback_report_no_results(monkeypatch):
    def open_url(req, timeout):
        if 'duckduckgo.com' in req.full_url:
            return io.BytesIO(b'<html>No usable search results</html>')
        return response({'query': {'search': []}})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='ddg')
    result, sources = client._run_tool('web_search', '{"query":"latest Python"}', Turn(), None)
    assert result['results'] == [] and sources == []
    assert 'no usable results' in result['notice']


def test_model_cannot_claim_an_answer_after_empty_search(monkeypatch):
    def open_url(req, timeout):
        if 'bing.com/search?' in req.full_url:
            return io.BytesIO(b'<rss><channel></channel></rss>')
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 's1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"today news"}'}}]}}]})
        return response({'choices': [{'message': {'content': 'The event definitely happened today.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='bing')
    answer, sources = client.respond('What happened today?', [], Turn())
    assert answer == "I couldn't find usable web results, so I can't verify an answer. Please try again."
    assert sources == []
    assert client.last_trace['checks']['search_evidence_missing']
    assert 'no verified results' in client.last_trace['search_warning']


def test_partial_search_warning_does_not_claim_all_results_are_fallback(monkeypatch):
    from agent import SearchResults
    client = AgentClient('key', search_provider='bing')
    searches = iter([SearchResults(notice='No results'),
                     SearchResults([{'title': 'Source', 'url': 'https://example.org/', 'description': 'Evidence'}])])
    monkeypatch.setattr(client, 'search', lambda query, turn: next(searches))
    responses = iter([
        {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query":"a"}'}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query":"b"}'}}]}}]},
        {'choices': [{'message': {'content': 'Here is what I found.'}}]},
    ])
    monkeypatch.setattr(client, '_json', lambda *args: next(responses))
    answer, sources = client.respond('Look up two things', [], Turn())
    assert answer == 'Some web searches returned incomplete results. Here is what I found.' and len(sources) == 1
    assert client.last_trace['search_warning'] == 'Some web searches returned incomplete results.'


def test_bing_rss_search_honors_site_restriction_and_returns_sources(monkeypatch):
    page = b'''<rss><channel>
      <item><title>Python Downloads</title><link>https://www.python.org/downloads/</link>
      <description>Official releases</description></item>
      <item><title>Wrong domain</title><link>https://evilpython.org/</link></item>
      <item><title>Bad link</title><link>https://[bad</link></item>
      <item><title>Unsafe</title><link>javascript:alert(1)</link></item>
    </channel></rss>'''
    def open_url(req, timeout):
        if 'bing.com/search?' in req.full_url:
            assert 'format=rss' in req.full_url
            assert 'site%3Apython.org' in req.full_url
            return io.BytesIO(page)
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 's1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"Python site:python.org"}'}}]}}]})
        assert 'Official releases' in body['messages'][-1]['content']
        return response({'choices': [{'message': {'content': 'See Python downloads.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_provider='bing')
    answer, sources = client.respond('Find the release.', [], Turn())
    assert answer == 'See Python downloads.'
    assert sources == [{'title': 'Python Downloads', 'url': 'https://www.python.org/downloads/',
                        'description': 'Official releases'}]


def test_bing_malformed_feed_reports_a_search_error(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: io.BytesIO(b'<broken'))
    with pytest.raises(AgentError, match='invalid response'):
        AgentClient('key', search_provider='bing').search('python', Turn())


@pytest.mark.parametrize('claimed,expected_blocked', [('3.13.7', True), ('3.15.2', False)])
def test_version_claim_requires_support_from_search_snippets(monkeypatch, claimed, expected_blocked):
    def open_url(req, timeout):
        if 'api.search.brave.com' in req.full_url:
            return response({'web': {'results': [
                {'url': 'https://www.python.org/', 'title': 'Python 3.15.2', 'description': 'Official docs'}]}})
        body = json.loads(req.data)
        if len(body['messages']) == 2:
            return response({'choices': [{'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 's1', 'type': 'function', 'function': {
                    'name': 'web_search', 'arguments': '{"query":"latest Python"}'}}]}}]})
        return response({'choices': [{'message': {'content': f'The latest version is {claimed}.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', search_key='search-key')
    answer, sources = client.respond('Latest Python?', [], Turn())
    assert client.last_trace['checks']['unsupported_version_blocked'] is expected_blocked
    assert len(sources) == 1
    if expected_blocked:
        assert claimed not in answer and 'could not verify' in answer
        assert 'not supported' in client.last_trace['search_warning']
    else:
        assert claimed in answer


def test_version_claim_without_web_search_is_not_rewritten(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response({
        'choices': [{'message': {'content': 'The configured version is 3.13.7.'}}]}))
    answer, _ = AgentClient('key').respond('Which version is configured?', [], Turn())
    assert answer == 'The configured version is 3.13.7.'


@pytest.mark.parametrize('body', [
    'Searching now. <TOOL_CALL>{"name":"web_search"}',
    '<tool_call>{"name":"remember","arguments":{"fact":"unwanted"}}</tool_call>',
    '<tool_call>{"name":"web_search","arguments":{"query":"news","extra":1}}</tool_call>',
    '<tool_call>{"name":"web_search","arguments":{"query":"news"}}</tool_call><tool_call>{}</tool_call>',
])
def test_textual_tool_request_is_not_spoken_as_a_reply(monkeypatch, body):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response({
        'choices': [{'message': {'content': body}}]}))
    with pytest.raises(AgentError, match='tool request as text'):
        AgentClient('key', search_provider='bing').respond('Find news', [], Turn())


def test_well_formed_textual_search_is_executed_and_results_reach_model(monkeypatch):
    client = AgentClient('key', search_provider='bing', local_tools=False)
    requests, phases, queries = [], [], []
    def model(req, turn, service):
        payload = json.loads(req.data)
        requests.append(payload)
        if len(requests) == 1:
            return {'choices': [{'message': {'content':
                '<tool_call>{"name":"web_search","arguments":{"query":"latest news"}}</tool_call>'}}]}
        assistant, tool = payload['messages'][-2:]
        assert assistant['role'] == 'assistant' and assistant['content'] is None
        assert assistant['tool_calls'][0]['id'] == tool['tool_call_id']
        assert 'A source' in tool['content']
        return {'choices': [{'message': {'content': 'Here is the news.'}}]}
    monkeypatch.setattr(client, '_json', model)
    def search(query, turn):
        queries.append(query)
        return [{'title': 'A source', 'url': 'https://example.org/news', 'description': 'A report'}]
    monkeypatch.setattr(client, 'search', search)
    answer, sources = client.respond('Search for news', [], Turn(), on_search=phases.append)
    assert answer == 'Here is the news.' and sources[0]['url'] == 'https://example.org/news'
    assert queries == ['latest news'] and phases == [True, False]
    assert client.last_trace['iterations'][0]['recovered_text_search'] is True


def test_textual_search_is_not_executed_when_search_is_disabled(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response({
        'choices': [{'message': {'content':
            '<tool_call>{"name":"web_search","arguments":{"query":"news"}}</tool_call>'}}]}))
    with pytest.raises(AgentError, match='tool request as text'):
        AgentClient('key', search_provider='none').respond('Find news', [], Turn())


def test_cancellation_during_recovered_search_stops_followup(monkeypatch):
    client = AgentClient('key', search_provider='bing', local_tools=False)
    calls = []
    def model(req, turn, service):
        calls.append(1)
        return {'choices': [{'message': {'content':
            '<tool_call>{"name":"web_search","arguments":{"query":"news"}}</tool_call>'}}]}
    def search(query, turn):
        turn.abort()
        return []
    monkeypatch.setattr(client, '_json', model)
    monkeypatch.setattr(client, 'search', search)
    turn = Turn()
    assert client.respond('Find news', [], turn) is None
    assert len(calls) == 1 and turn.cancel.is_set()


def test_search_warning_is_visible_and_cleared_on_reset():
    class WarningAgent(FakeAgent):
        def respond(self, text, history, turn, on_search):
            self.last_trace = {'search_warning': 'Web search returned no verified results.'}
            return 'A short answer.', []

    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                    client=LocalVoice(), agent=WarningAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set())
        assert 'no verified results' in app.state()['warning']
        app.reset()
        assert 'no verified results' not in (app.state()['warning'] or '')
    finally:
        app.close()


def test_build_messages_dedupes_repeated_turns_and_closes_interrupted_ones():
    history = [
        {'role': 'user', 'content': 'What is the weather in Basel?'},
        {'role': 'user', 'content': 'Can you tell me a story?'},
        {'role': 'user', 'content': 'can you tell me a story'},
    ]
    messages, stats = AgentClient._build_messages('SYS', history, 'Can you tell me a story?')
    assert stats['deduped_user_turns'] >= 2
    assert stats['closed_unanswered_turns'] == 1
    roles = [m['role'] for m in messages[1:]]
    assert roles == ['user', 'assistant', 'user']
    assert messages[-1]['content'] == 'Can you tell me a story?'


def test_agent_is_told_previous_draft_may_not_have_been_heard(monkeypatch):
    def open_url(req, timeout):
        messages = json.loads(req.data)['messages']
        assert messages[1]['content'] == 'Suggest ways to relax.'
        assert messages[2]['role'] == 'assistant'
        assert 'may not have heard all' in messages[2]['content']
        assert 'Walk, read, or listen to music.' in messages[2]['content']
        assert messages[3]['content'] == 'What about something else?'
        return response({'choices': [{'message': {'content': 'Try a warm bath.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key')
    history = [{'role': 'user', 'content': 'Suggest ways to relax.'},
               {'role': 'assistant', 'content': 'Walk, read, or listen to music. [interrupted]'}]
    assert client.respond('What about something else?', history, Turn()) == ('Try a warm bath.', [])


def test_unanswered_prior_turn_is_marked_as_unheard():
    messages, _ = AgentClient._build_messages('SYS', [
        {'role': 'user', 'content': 'Tell me about the weather.'}], 'Actually, make it tomorrow.')
    assert 'without a confirmed audible answer' in messages[2]['content']
    assert messages[-1]['content'] == 'Actually, make it tomorrow.'


def test_interrupted_spoken_reply_is_recorded_in_history_to_prevent_repeat_hell():
    import threading
    playing_started = threading.Event()
    class SlowPlayer(FakePlayer):
        def play_chunks(self, chunks, cancel=None, on_start=None, on_play=None, on_written=None):
            for audio in chunks:
                if cancel.is_set():
                    break
                self.playing = True
                on_start()
                on_play(audio)
                if on_written:
                    on_written(audio)
                playing_started.set()
                cancel.wait(2)
                break
            self.playing = False
    agent = FakeAgent()
    app = Assistant(mic=FakeMic(), player=SlowPlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert playing_started.wait(1)
        app.interrupt('barge')
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        assert app._agent_history == [
            {'role': 'user', 'content': 'What is new?'},
            {'role': 'assistant', 'content': 'A short answer. [interrupted]'},
        ]
    finally:
        app.close()


def test_empty_reasoning_reply_retries_without_effort(monkeypatch):
    attempts = []
    def open_url(req, timeout):
        body = json.loads(req.data)
        attempts.append((body.get('reasoning_effort'), body.get('max_tokens')))
        if len(attempts) == 1:
            return response({'choices': [{'finish_reason': 'length', 'message': {'content': '', 'reasoning_content': 'thinking...'}}]})
        return response({'choices': [{'finish_reason': 'stop', 'message': {'content': 'Recovered reply.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    answer, _ = AgentClient('key').respond('Hello', [], Turn())
    assert answer == 'Recovered reply.'
    assert attempts == [('low', 450), (None, 1200)]


def test_traces_and_checks_are_saved_and_runtime_mode_switches(tmp_path):
    trace_file = tmp_path / 'traces.jsonl'
    agent = FakeAgent()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent, trace_path=trace_file)
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        state = app.state()
        assert state['trace_count'] == 1
        assert state['trace_file'] == 'traces.jsonl'
        assert state['last_trace']['checks']['all_passed'] is True
        traces = app.traces()
        assert len(traces) == 1 and traces[0]['turn_id'] == job.id
        saved = [json.loads(line) for line in trace_file.read_text(encoding='utf-8').splitlines()]
        assert len(saved) == 1 and saved[0]['checks']['all_passed'] is True
        res = app.set_agent_mode('local')
        assert res == {'ok': True, 'agent_mode': False} and not app.state()['agent_mode']
    finally:
        app.close()


def test_search_selection_and_zen_switch_preserve_provider(monkeypatch):
    monkeypatch.delenv('BRAVE_SEARCH_API_KEY', raising=False)
    monkeypatch.setenv('OPENCODE_ZEN_API_KEY', 'test-key')
    agent = AgentClient('test-key', search_provider='ddg')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        assert app.set_search_provider('bing')['search_enabled']
        assert app.state()['search_provider'] == 'bing'
        with pytest.raises(ValueError, match='BRAVE_SEARCH_API_KEY'):
            app.set_search_provider('brave')
        assert app.state()['search_provider'] == 'bing'
        app.set_agent_mode('opencode-zen', 'kimi-k3')
        assert app.state()['agent_provider'] == 'OpenCode Zen'
        assert app.state()['search_provider'] == 'bing'
        assert not app.set_search_provider('none')['search_enabled']
        app.set_agent_mode('local')
        with pytest.raises(ValueError, match='cloud agent'):
            app.set_search_provider('bing')
    finally:
        app.close()


def test_agent_switch_uses_consistent_defaults_and_does_not_reuse_another_providers_model(monkeypatch):
    monkeypatch.delenv('LOCALVOICE_AGENT_MODEL', raising=False)
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice())
    try:
        assert app.set_agent_mode('go')['model'] == 'kimi-k3'
        assert app.set_agent_mode('go', 'beam-free')['model'] == 'beam-free'
        app.set_agent_mode('local')
        assert app.set_agent_mode('go')['model'] == 'beam-free'
        assert app.set_agent_mode('zen')['model'] == 'kimi-k3'
        assert app.set_agent_mode('custom')['model'] == 'local-model'
    finally:
        app.close()


def test_agent_switch_honors_configured_default_model(monkeypatch):
    monkeypatch.setenv('LOCALVOICE_AGENT_MODEL', 'custom-go-model')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice())
    try:
        assert app.set_agent_mode('go')['model'] == 'custom-go-model'
    finally:
        app.close()


@pytest.mark.parametrize('expert,expected', [(None, 'kimi-k3'), ('expert-model', 'expert-model')])
def test_model_switch_updates_default_expert_but_preserves_explicit_expert(expert, expected):
    agent = AgentClient('key', model='beam-free', expert_model=expert)
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        app.set_agent_mode('go', 'kimi-k3')
        assert app.agent.model == 'kimi-k3'
        assert app.agent.expert_model == expected
    finally:
        app.close()


def test_search_provider_http_rejects_missing_brave_key_and_invalid_body():
    import threading
    from http.server import ThreadingHTTPServer
    from app import Handler

    agent = AgentClient('test-key', search_provider='ddg')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.assistant = app
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}/api/search-provider'
    def post(body):
        request = urllib.request.Request(base, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as result:
            return json.load(result)
    try:
        assert post({'provider': 'bing'})['search_provider'] == 'bing'
        for body in ({'provider': 'brave'}, {'provider': 'bing', 'extra': True}, {'provider': 'other'}):
            with pytest.raises(urllib.error.HTTPError) as error:
                post(body)
            assert error.value.code == 400
        assert app.state()['search_provider'] == 'bing'
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        app.close()


def test_player_recovers_when_cached_output_device_fails(monkeypatch):
    import sounddevice as sd
    from audio_io import Player
    opened_devices = []
    class Output:
        latency = .02
        def __init__(self, **kwargs):
            opened_devices.append(kwargs.get('device'))
            if kwargs.get('device') == 99:
                raise sd.PortAudioError('Error opening OutputStream: Insufficient memory [PaErrorCode -9992]')
        def start(self): pass
        def write(self, a): return False
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr('audio_io.sd.OutputStream', Output)
    monkeypatch.setattr('audio_io.preferred_device', lambda kind: 10)
    monkeypatch.setattr('audio_io.sd.query_devices', lambda dev=None, kind=None: (
        [{'default_samplerate': 48000, 'max_output_channels': 2}] if dev is None
        else {'default_samplerate': 48000, 'max_output_channels': 2}))
    player = Player(device=99)
    player.play_chunks([np.zeros(480, np.float32)])
    assert opened_devices == [99, 10]
    assert player.device == 10


def test_consult_expert_delegates_to_expert_model_and_saves_notes(monkeypatch):
    calls = []
    def open_url(req, timeout):
        body = json.loads(req.data)
        calls.append(body)
        if len(calls) == 1:
            return response({'choices': [{'message': {'content': '', 'tool_calls': [{
                'id': 'exp-1',
                'function': {'name': 'consult_expert', 'arguments': json.dumps({'question': 'Explain FlashAttention-2 memory complexity'})},
            }]}}]})
        if len(calls) == 2:
            assert body['model'] == 'kimi-k3'
            return response({'choices': [{'message': {'content': 'FlashAttention-2 tiles Q, K, V in SRAM to reduce HBM accesses from O(N^2) to O(N).' * 4}}]})
        return response({'choices': [{'message': {'content': 'I consulted the reasoning engine: FlashAttention-2 keeps attention tiles in fast on-chip SRAM so memory traffic scales linearly instead of quadratically.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client = AgentClient('key', model='beam-free', expert_model='kimi-k3', local_tools=True)
    answer, _ = client.respond('How does FlashAttention-2 save memory?', [], Turn())
    assert 'FlashAttention-2' in answer
    assert len(client.last_notes) == 1
    assert client.last_notes[0]['model'] == 'kimi-k3'
    assert 'SRAM' in client.last_notes[0]['analysis']


def test_persistent_memory_and_persona_switching(monkeypatch, tmp_path):
    mem_file = tmp_path / 'agent_memory.json'
    client1 = AgentClient('key', local_tools=True, memory_path=mem_file, persona='jarvis')
    client1.remember('User runs on RTX 3050 4GB Laptop GPU', 'hardware')
    assert client1.set_persona('concise')['persona'] == 'concise'
    client2 = AgentClient('key', local_tools=True, memory_path=mem_file)
    assert client2.recall('RTX')['matches'][0]['fact'] == 'User runs on RTX 3050 4GB Laptop GPU'
    sent_system = []
    def open_url(req, timeout):
        body = json.loads(req.data)
        sent_system.append(body['messages'][0]['content'])
        return response({'choices': [{'message': {'content': 'All systems nominal.'}}]})
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    client2.respond('Check status', [], Turn())
    assert 'RTX 3050 4GB' in sent_system[0]


def test_submit_text_and_sentence_pipelined_tts():
    class TrackingVoice(LocalVoice):
        def __init__(self):
            self.asr_calls = []
            self.tts_calls = []
        def asr_text(self, wav, turn):
            self.asr_calls.append(wav)
            return super().asr_text(wav, turn)
        def tts(self, text, turn, max_tokens=128, on_audio=None):
            self.tts_calls.append(text)
            turn.audio_samples += 480
            turn.ttft = .01
            on_audio(np.ones(480, np.float32) * .1)
            turn.done.set()

    class LongReplyAgent(FakeAgent):
        def respond(self, text, history, turn, on_search):
            return (
                'First sentence explains the core architecture clearly and concisely for immediate playback. '
                'Second sentence adds details about the symmetric eight-bit KV cache quantization on laptop GPUs. '
                'Third sentence summarizes how the voice intermediary coordinates with deep reasoning models.'
            ), []

    voice = TrackingVoice()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=voice, agent=LongReplyAgent())
    try:
        job = app.submit_text('Explain the architecture in detail.')
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        assert voice.asr_calls == []
        assert len(voice.tts_calls) >= 2
        assert 'user: Explain the architecture in detail.' in app.state()['transcript']
    finally:
        app.close()


def test_assistant_persona_switching_text_prompt_and_notes_lifecycle():
    class NotesAgent(FakeAgent):
        persona = 'jarvis'
        last_trace = {'expert_notes': [{'model': 'kimi-k3', 'task': 'Deep Analysis', 'analysis': 'Kernel tiling in on-chip SRAM'}]}
        def set_persona(self, persona):
            self.persona = persona
            return persona

    local = LocalVoice()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=local, agent=None)
    try:
        with pytest.raises(ValueError, match='requires an active cloud agent'):
            app.set_persona('concise')

        app.agent = NotesAgent()
        assert app.set_persona('concise')['persona'] == 'concise'
        assert app.state()['agent_persona'] == 'concise'

        job = app.submit_text('Analyze attention kernels')
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        state = app.state()
        assert 'user: Analyze attention kernels' in state['transcript']
        assert len(state['agent_notes']) == 1
        assert state['agent_notes'][0]['model'] == 'kimi-k3'
        assert 'SRAM' in state['agent_notes'][0]['analysis']

        app.reset()
        assert app.state()['agent_notes'] == []
    finally:
        app.close()





def test_switching_provider_clears_context_and_busy_switch_is_rejected():
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(),
                    agent=AgentClient('key', provider='go', local_tools=False))
    try:
        app._agent_history.append({'role': 'user', 'content': 'private prior text'})
        app.status = 'recording'
        with pytest.raises(RuntimeError, match='current conversation turn'):
            app.set_agent_mode('opencode-zen', 'kimi-k3')
        assert app._agent_history and app.agent.provider == 'go'
        app.status = 'listening'
        app.set_agent_mode('opencode-zen', 'kimi-k3')
        assert app._agent_history == [] and app.agent.provider == 'zen'
    finally:
        app.close()


def test_followup_receives_interrupted_reply_as_possibly_unheard():
    import threading
    first_written = threading.Event()
    release_playback = threading.Event()

    class InterruptiblePlayer(FakePlayer):
        def play_chunks(self, chunks, cancel=None, on_start=None, on_play=None, on_written=None):
            self.playing = True
            try:
                for audio in chunks:
                    if cancel.is_set(): break
                    on_start()
                    on_play(audio)
                    on_written(audio)
                    first_written.set()
                    release_playback.wait(2)
                    break
            finally:
                self.playing = False

    agent = FakeAgent()
    app = Assistant(mic=FakeMic(), player=InterruptiblePlayer(), front=FakeFront(),
                    client=LocalVoice(), agent=agent)
    try:
        first = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert first_written.wait(2)
        app.interrupt()
        release_playback.set()
        wait_for(lambda: first.playback_closed.is_set())
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: second.playback_closed.is_set() and app._active is None)
        assert agent.histories[1] == [
            {'role': 'user', 'content': 'What is new?'},
            {'role': 'assistant', 'content': 'A short answer. [interrupted]'}]
        messages, _ = AgentClient._build_messages('system', agent.histories[1], 'What about something else?')
        assert 'may not have heard' in messages[2]['content']
        assert 'A short answer.' in messages[2]['content']
    finally:
        release_playback.set()
        app.close()


def test_interrupted_draft_is_saved_once_and_reset_discards_it():
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=FakeAgent())
    try:
        from app import Job
        job = Job(epoch=app._epoch, agent_input='question', played_samples=20)
        job.turn.text_parts.append('Generated answer.')
        app._active = job
        app.interrupt()
        with app._lock:
            app._remember_agent_reply(job, interrupted=True)
        assert app._agent_history == [{'role': 'user', 'content': 'question'},
                                      {'role': 'assistant', 'content': 'Generated answer. [interrupted]'}]
        app.reset()
        assert app._agent_history == []
    finally:
        app.close()


def test_sources_stay_with_their_reply_across_followup_and_reset():
    class FirstSourceAgent(FakeAgent):
        def __init__(self): self.calls = 0
        def respond(self, text, history, turn, on_search):
            self.calls += 1
            sources = [{'title': 'Official', 'url': 'https://example.org/article'},
                       {'title': 'bad', 'url': 'javascript:alert(1)'},
                       {'title': 'credentials', 'url': 'https://user:pass@example.org/'}] if self.calls == 1 else []
            return 'A short answer.', sources
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=LocalVoice(), agent=FirstSourceAgent())
    try:
        first = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: first.playback_closed.is_set() and app._active is None)
        first_reply = next(m for m in app.state()['messages'] if m['id'] == first.reply_id)
        assert first_reply['sources'] == [{'title': 'Official', 'url': 'https://example.org/article'}]
        assert app.state()['agent_sources'] == first_reply['sources']
        first_reply['sources'][0]['title'] = 'changed externally'
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: second.playback_closed.is_set() and app._active is None)
        state = app.state()
        assert next(m for m in state['messages'] if m['id'] == first.reply_id)['sources'][0]['title'] == 'Official'
        assert next(m for m in state['messages'] if m['id'] == second.reply_id)['sources'] == []
        app.reset()
        assert app.state()['messages'] == []
    finally:
        app.close()


def test_late_device_write_preserves_interrupted_reply_before_followup():
    import threading
    writing = threading.Event()
    release_write = threading.Event()
    class LateWritePlayer(FakePlayer):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def play_chunks(self, chunks, cancel=None, on_start=None, on_play=None, on_written=None):
            self.calls += 1
            if self.calls != 1:
                return super().play_chunks(chunks, cancel, on_start, on_play, on_written)
            self.playing = True
            try:
                for audio in chunks:
                    on_start()
                    on_play(audio)
                    writing.set()
                    assert release_write.wait(3)
                    # A device write returned successfully after cancellation.
                    on_written(audio)
                    break
            finally:
                self.playing = False
    agent = FakeAgent()
    app = Assistant(mic=FakeMic(), player=LateWritePlayer(), front=FakeFront(), client=LocalVoice(), agent=agent)
    try:
        first = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert writing.wait(2)
        second = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert first.turn.cancel.is_set() and first.played_samples == 0
        release_write.set()
        wait_for(lambda: second.playback_closed.is_set() and app._active is None)
        assert agent.histories[1] == [
            {'role': 'user', 'content': 'What is new?'},
            {'role': 'assistant', 'content': 'A short answer. [interrupted]'}]
    finally:
        release_write.set()
        app.close()


def test_late_write_after_reset_does_not_restore_old_context():
    import threading
    writing = threading.Event()
    release_write = threading.Event()
    class LateWritePlayer(FakePlayer):
        def play_chunks(self, chunks, cancel=None, on_start=None, on_play=None, on_written=None):
            self.playing = True
            try:
                for audio in chunks:
                    on_start()
                    writing.set()
                    assert release_write.wait(3)
                    on_written(audio)
                    break
            finally:
                self.playing = False
    app = Assistant(mic=FakeMic(), player=LateWritePlayer(), front=FakeFront(), client=LocalVoice(), agent=FakeAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert writing.wait(2)
        app.reset()
        release_write.set()
        wait_for(lambda: job.playback_closed.is_set())
        assert app._agent_history == []
    finally:
        release_write.set()
        app.close()


@pytest.mark.parametrize('transcript', ['', '   ', None])
def test_empty_transcription_does_not_call_cloud_or_speak(transcript):
    class EmptyVoice(LocalVoice):
        def asr_text(self, wav, turn):
            return transcript
        def tts(self, *args, **kwargs):
            pytest.fail('No speech should be synthesized for an empty transcription')
    class NeverAgent(FakeAgent):
        def respond(self, *args, **kwargs):
            pytest.fail('An empty transcription must not be sent to the provider')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=EmptyVoice(), agent=NeverAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        state = app.state()
        assert state['error'] == 'No speech was transcribed. Please try again.'
        assert job.played_samples == 0 and app._agent_history == []
        assert state['agent_sources'] == []
    finally:
        app.close()


def test_failed_transcription_does_not_call_cloud_or_speak():
    class FailingVoice(LocalVoice):
        def asr_text(self, wav, turn):
            raise RuntimeError('Local transcription unavailable')
        def tts(self, *args, **kwargs):
            pytest.fail('No reply should be synthesized after transcription failure')
    class NeverAgent(FakeAgent):
        def respond(self, *args, **kwargs):
            pytest.fail('The cloud provider must not receive a failed transcription')
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(), client=FailingVoice(), agent=NeverAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        assert app.state()['error'] == 'Local transcription unavailable'
        assert job.played_samples == 0 and app._agent_history == []
    finally:
        app.close()


def test_interrupt_clears_reference_after_inflight_callback():
    import threading
    entered = threading.Event()
    release = threading.Event()
    interrupt_done = threading.Event()
    class BlockingFront(FakeFront):
        def __init__(self):
            super().__init__()
            self.order = []
        def push_far(self, audio, rate=16000):
            entered.set()
            assert release.wait(2)
            self.order.append('push')
        def clear_playback(self):
            self.order.append('clear')
    front = BlockingFront()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=front, client=LocalVoice(), agent=FakeAgent())
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        assert entered.wait(2)
        def stop():
            app.interrupt()
            interrupt_done.set()
        thread = threading.Thread(target=stop)
        thread.start()
        assert not interrupt_done.wait(.05)
        release.set()
        assert interrupt_done.wait(2)
        thread.join(2)
        wait_for(lambda: job.playback_closed.is_set())
        assert front.order[-2:] == ['push', 'clear']
    finally:
        release.set()
        app.close()


def test_truncated_reply_speaks_retry_message_and_shows_warning(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response({
        'choices': [{'finish_reason': 'length', 'message': {'content': 'An incomplete claim'}}]}))
    class CheckingVoice(LocalVoice):
        def tts(self, text, turn, max_tokens=128, on_audio=None):
            assert text == 'My reply was cut short. Please ask me to try again.'
            turn.audio_samples += 480
            on_audio(np.ones(480, np.float32) * .1)
            turn.done.set()
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                    client=CheckingVoice(), agent=AgentClient('test-key'))
    try:
        job = app.submit_audio(np.ones(3200, np.float32) * .1)
        wait_for(lambda: job.playback_closed.is_set() and app._active is None)
        assert 'length limit' in app.state()['warning']
        assert 'assistant: My reply was cut short. Please ask me to try again.' in app.state()['transcript']
        assert app._agent_history[-1]['content'] == 'My reply was cut short. Please ask me to try again.'
        app.reset()
        assert app.state()['warning'] is None
    finally:
        app.close()
