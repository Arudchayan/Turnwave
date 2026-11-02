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
    assert answer == 'Jujutsu Kaisen is a manga series.'
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



