"""Contracts for the deterministic model used by the spoken-search probe."""
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.request

from src.synth.spoken_search_probe import ANSWER, QUERY, StubModel


def test_probe_model_requests_search_and_depends_on_official_source():
    server = ThreadingHTTPServer(('127.0.0.1', 0), StubModel)
    server.synthetic_input = server.tool_result = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    def ask(messages):
        req = urllib.request.Request(endpoint, data=json.dumps({'messages': messages}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)['choices'][0]['message']
    try:
        initial = ask([{'role': 'user', 'content': 'Search Python downloads.'}])
        assert json.loads(initial['tool_calls'][0]['function']['arguments']) == {'query': QUERY}
        empty = ask([{'role': 'tool', 'content': '[]'}])
        assert empty['content'] != ANSWER
        found = ask([{'role': 'tool', 'content': json.dumps([
            {'title': 'Downloads', 'url': 'https://www.python.org/downloads/'}])}])
        assert found['content'] == ANSWER
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
