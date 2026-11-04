import io
import json
import pathlib
import sys
import urllib.error
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from agent import AgentClient, AgentError
from client import Turn

def response(value):
    return io.BytesIO(json.dumps(value).encode())

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
