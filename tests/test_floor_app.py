"""Application wiring for the reversible floor policy."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app import Assistant
from test_realtime import FakeMic, FakePlayer, wait_for


class ScriptedFront:
    """Return a fixed sequence of VAD outcomes, one per audio block."""

    level = .1
    aec_error = None
    probability = .9
    echo_score = 0.0

    def __init__(self, events):
        self.events = list(events)
        self.calls = 0
        self.speaking = False
        self.references = []

    def warmup(self):
        pass

    def reset(self):
        self.speaking = False

    def clear_playback(self):
        pass

    def push_far(self, audio, rate=16000):
        self.references.append((len(audio), rate))

    def update(self, audio):
        speaking, started, ended = self.events[min(self.calls, len(self.events) - 1)]
        self.calls += 1
        self.speaking = speaking
        return speaking, started, ended, np.asarray(audio, dtype=np.float32)


class GainPlayer(FakePlayer):
    def __init__(self):
        super().__init__()
        self.gains = []

    def set_gain(self, gain, ramp_ms=30):
        self.gains.append((round(float(gain), 3), ramp_ms))


class LongClient:
    base = 'http://chat'

    def __init__(self):
        self.calls = []

    def health(self):
        return False

    def chat_turn(self, wav, turn, first, on_audio=None):
        self.calls.append((first, len(wav)))
        for _ in range(200):
            if turn.cancel.wait(.01):
                break
            turn.audio_samples += 1440
            turn.ttft = .01
            on_audio(np.ones(1440, np.float32) * .1)
        turn.done.set()


def make_assistant(front, player, client):
    assistant = Assistant(mic=FakeMic(), player=player, front=front,
                          client=client, floor_mode='reversible')
    assistant._spawn(assistant._loop)
    return assistant


def test_reversible_backchannel_resumes_without_cancelling():
    # 200 ms onset, then silence. A short vocalisation must not take the floor.
    front = ScriptedFront([(True, True, False), (False, False, True)])
    player, client = GainPlayer(), LongClient()
    assistant = make_assistant(front, player, client)
    try:
        job = assistant.submit_audio(np.ones(8000, np.float32) * .1)
        wait_for(lambda: player.playing)
        assistant.mic.q.put(np.ones(3200, np.float32) * .1)
        wait_for(lambda: front.calls >= 1)
        assistant.mic.q.put(np.zeros(3200, np.float32))
        wait_for(lambda: front.calls >= 2)
        wait_for(lambda: player.gains and player.gains[-1][0] == 1.0)
        assert not job.turn.cancel.is_set()
        assert len(client.calls) == 1
        assert (.25, 30) in player.gains
        assert assistant.floor.state == 'idle'
    finally:
        assistant.close()


def test_reversible_sustained_barge_in_commits_and_submits():
    # 200 ms onset, 200 ms continuation (400 ms total), then silence.
    front = ScriptedFront([(True, True, False), (True, False, False),
                           (False, False, True)])
    player, client = GainPlayer(), LongClient()
    assistant = make_assistant(front, player, client)
    try:
        job = assistant.submit_audio(np.ones(8000, np.float32) * .1)
        wait_for(lambda: player.playing)
        assistant.mic.q.put(np.ones(3200, np.float32) * .1)
        wait_for(lambda: front.calls >= 1)
        assistant.mic.q.put(np.ones(3200, np.float32) * .1)
        wait_for(job.turn.cancel.is_set)
        assistant.mic.q.put(np.zeros(3200, np.float32))
        wait_for(lambda: len(client.calls) == 2)
        assert client.calls[1][0] is True
        assert assistant.floor.state == 'idle'
    finally:
        assistant.close()


def test_hard_mode_preserves_immediate_cancellation():
    front = ScriptedFront([(True, True, False), (False, False, True)])
    player, client = GainPlayer(), LongClient()
    assistant = Assistant(mic=FakeMic(), player=player, front=front,
                          client=client, floor_mode='hard')
    assistant._spawn(assistant._loop)
    try:
        job = assistant.submit_audio(np.ones(8000, np.float32) * .1)
        wait_for(lambda: player.playing)
        assistant.mic.q.put(np.ones(3200, np.float32) * .1)
        wait_for(job.turn.cancel.is_set)
        assert assistant.floor.state == 'idle'
        assert (.25, 30) not in player.gains
    finally:
        assistant.close()
