"""The pause control changes turn boundaries without interrupting live speech."""
import pathlib
import sys
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from duplex import DuplexFrontend
from app import Assistant, Handler
from test_realtime import FakeMic, FakePlayer, FakeClient


def frontend():
    front = DuplexFrontend(aec=False, holdoff_s=0)
    # Deterministic voice probabilities isolate endpoint timing from the VAD model.
    front._prob = lambda chunk: float(np.mean(np.abs(chunk)) > .02)
    return front


def boundaries(pause_ms):
    front = frontend()
    front.set_end_silence(pause_ms)
    audio = np.concatenate([np.full(16000, .1, np.float32),
                            np.zeros(10400, np.float32),
                            np.full(11200, .1, np.float32),
                            np.zeros(24000, np.float32)])
    events = []
    for offset in range(0, len(audio), 160):
        _, started, ended, _ = front.update(audio[offset:offset + 160])
        if started or ended:
            events.append(('start' if started else 'end', offset / 16000))
    return events


def test_longer_pause_preserves_a_650ms_gap_but_delays_endpoint():
    quick = boundaries(480)
    patient = boundaries(800)
    assert [kind for kind, _ in quick] == ['start', 'end', 'start', 'end']
    assert [kind for kind, _ in patient] == ['start', 'end']
    assert patient[-1][1] > quick[-1][1] + .25


@pytest.mark.parametrize('value', [True, None, 0, 319, 1601, 800.0, '800'])
def test_invalid_pause_is_rejected(value):
    with pytest.raises(ValueError):
        frontend().set_end_silence(value)


def test_pause_cannot_change_in_middle_of_speech():
    front = frontend()
    for _ in range(20):
        front.update(np.full(160, .1, np.float32))
    assert front.speaking
    with pytest.raises(RuntimeError, match='Finish speaking'):
        front.set_end_silence(800)
    assert front.end_chunks * 32 == 480


def test_pause_api_updates_state_and_rejects_changes_during_capture():
    app = Assistant(mic=FakeMic(), player=FakePlayer(), front=frontend(), client=FakeClient())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.assistant = app
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(value):
        request = urllib.request.Request(base + '/api/pause', data=json.dumps(value).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    try:
        assert post({'milliseconds': 800}) == {'ok': True, 'milliseconds': 800}
        assert app.state()['end_silence_ms'] == 800
        app.status = 'recording'
        with pytest.raises(urllib.error.HTTPError) as error:
            post({'milliseconds': 1120})
        assert error.value.code == 409
        assert app.state()['end_silence_ms'] == 800
        with pytest.raises(urllib.error.HTTPError) as error:
            post({'milliseconds': True})
        # Capture state is checked first; the setting remains unchanged.
        assert error.value.code == 409
        app.status = 'listening'
        with pytest.raises(urllib.error.HTTPError) as error:
            post({'milliseconds': True})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(1)
        app.close()
