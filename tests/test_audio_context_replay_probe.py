import json

from src.synth.audio_context_replay_probe import ReplayProbeClient
from lfm3_client import LFM3Client
from client import Turn


def test_replay_sends_prior_audio_before_followup_and_marks_draft_unheard(monkeypatch):
    events = []
    class Socket:
        def send(self, raw):
            events.append(json.loads(raw))
    monkeypatch.setattr(LFM3Client, '_append', staticmethod(
        lambda sock, pcm, turn: events.append({'audio': pcm})))
    monkeypatch.setattr(LFM3Client, '_pcm', staticmethod(lambda wav: wav))
    client = ReplayProbeClient('http://127.0.0.1:8080')
    client.replay = (b'old audio', 'Earlier draft')
    client._append(Socket(), b'new audio', Turn())
    assert events[0]['type'] == 'conversation.item.create'
    assert events[1] == {'audio': b'old audio'}
    assert events[2]['type'] == 'input_audio_buffer.commit'
    assert 'may not have been heard' in events[3]['item']['content'][0]['text']
    assert events[4] == {'audio': b'new audio'}
    assert client.replay is None
