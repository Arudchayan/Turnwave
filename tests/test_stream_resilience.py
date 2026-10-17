"""Generation result boundaries with deterministic in-memory SSE responses."""
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from client import ServerClient, Turn


SAMPLES = np.array([.125, -.25, .375], dtype='<f4')


def frame(event):
    return ('data: ' + json.dumps(event) + '\n\n').encode()


def reply(error=None, *, include_error=False):
    audio = {'choices': [{'delta': {
        'content': 'Prefix reply.',
        'audio_chunk': {'data': base64.b64encode(SAMPLES).decode()},
    }}]}
    result = frame(audio)
    if include_error:
        result += frame({'error': error})
    return result + b'data: [DONE]\n\n'


class Response(io.BytesIO):
    def __init__(self, body, close_error=None):
        super().__init__(body)
        self.close_error = close_error
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()
        if self.close_error is not None:
            raise self.close_error


def assert_released_and_recover(client, turn):
    assert turn.done.is_set()
    assert turn._response is None
    assert client.request_lock.acquire(blocking=False)
    client.request_lock.release()
    observed = []
    recovered = Turn()
    recovered.request_observer = observed.append
    client._post = lambda payload: Response(reply())
    client.chat_turn(b'recovery-wav', recovered, True)
    assert recovered.error is None
    assert recovered.done.is_set() and recovered.audio_samples == len(SAMPLES)
    assert len(observed) == 1 and observed[0]['reset_context'] is True


@pytest.mark.parametrize('close_error', [ValueError('close failed'), RuntimeError('close failed')])
@pytest.mark.parametrize('outcome', ['complete', 'server_error', 'cancelled'])
def test_response_cleanup_preserves_generation_result(close_error, outcome):
    client = ServerClient()
    turn = Turn()
    received = []
    response = Response(reply('primary generation failed', include_error=outcome == 'server_error'), close_error)
    client._post = lambda payload: response

    def receive(audio):
        received.append(audio.copy())
        if outcome == 'cancelled':
            turn.cancel.set()

    client.chat_turn(b'synthetic-wav', turn, True, on_audio=receive)
    assert turn.error == ('primary generation failed' if outcome == 'server_error' else None)
    assert turn.cancel.is_set() == (outcome == 'cancelled')
    assert turn.text == 'Prefix reply.' and turn.audio_samples == len(SAMPLES)
    np.testing.assert_array_equal(np.concatenate(received), SAMPLES)
    assert response.close_calls == 1 and response.closed
    assert_released_and_recover(client, turn)


@pytest.mark.parametrize('error', ['', False, 0, {}, []])
def test_falsey_server_error_after_audio_cannot_report_success(error):
    client = ServerClient()
    turn = Turn()
    received = []
    response = Response(reply(error, include_error=True))
    client._post = lambda payload: response
    client.chat_turn(b'synthetic-wav', turn, True, on_audio=received.append)
    assert turn.error == 'The server reported an error.'
    assert turn.text == 'Prefix reply.' and turn.audio_samples == len(SAMPLES)
    np.testing.assert_array_equal(np.concatenate(received), SAMPLES)
    assert response.close_calls == 1
    assert_released_and_recover(client, turn)


def test_null_error_metadata_keeps_successful_reply():
    client = ServerClient()
    turn = Turn()
    client._post = lambda payload: Response(reply(None, include_error=True))
    client.chat_turn(b'synthetic-wav', turn, True)
    assert turn.error is None and turn.audio_samples == len(SAMPLES)
    assert_released_and_recover(client, turn)
