"""Capture recovery boundaries, without devices or a backend."""
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app import Assistant
from test_realtime import FakeClient, FakeFront, FakeMic, FakePlayer


@pytest.mark.parametrize('action', ['reset', 'mute', 'none'])
def test_inflight_detection_failure_respects_capture_recovery(action):
    entered = threading.Event()
    release = threading.Event()

    class DelayedFailure(FakeFront):
        def update(self, audio):
            entered.set()
            assert release.wait(2)
            raise RuntimeError('simulated detector failure')

    assistant = Assistant(mic=FakeMic(), player=FakePlayer(),
                          front=DelayedFailure(), client=FakeClient())
    assistant.status = 'listening'
    worker = assistant._spawn(assistant._loop)
    try:
        assistant.mic.q.put(np.full(3200, .1, np.float32))
        assert entered.wait(1)
        if action == 'reset':
            assistant.reset()
        elif action == 'mute':
            assistant.set_mic_enabled(False)
        # End the loop after this in-flight block so no later update can mask
        # its result. This does not bypass the exception handler under test.
        assistant._stop.set()
        release.set()
        worker.join(1)
        assert not worker.is_alive()
        state = assistant.state()
        if action == 'none':
            assert state['status'] == 'error'
            assert state['error'] == 'Speech detection failed: simulated detector failure'
        else:
            assert state['status'] == 'listening'
            assert state['error'] is None
        assert not assistant.client.calls
        assert not state['active_turn'] and not state['pending_turns']
    finally:
        release.set()
        assistant.close()
