import http.server
import pathlib
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from agent import AgentClient
from client import Turn


def test_abort_unblocks_pooled_agent_response_read():
    pytest.importorskip('urllib3')
    headers_sent = threading.Event()
    release_server = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '100')
            self.end_headers()
            self.wfile.flush()
            headers_sent.set()
            release_server.wait(5)
            try:
                self.wfile.write(b'x' * 100)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = AgentClient('key', timeout=5)
    turn = Turn()
    result = []
    request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/')
    reader = threading.Thread(target=lambda: result.append(client._read(request, turn, 'Test')), daemon=True)
    reader.start()
    try:
        assert headers_sent.wait(2)
        deadline = time.monotonic() + 2
        while turn._response is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert turn._response is not None
        turn.abort()
        reader.join(1)
        assert not reader.is_alive(), 'Cancellation did not unblock the pooled response read'
        assert result == [None]
    finally:
        release_server.set()
        reader.join(6)
        server.shutdown()
        server.server_close()
        client._pool.clear()
