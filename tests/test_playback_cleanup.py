"""Output cleanup failure must not leave a stale playback owner."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from app import Assistant
from audio_io import Player
from test_realtime import FakeMic,FakeFront,FakeClient,wait_for


def test_close_failure_releases_player_and_allows_next_reply(monkeypatch):
    closes=[]
    class Output:
        latency=.04
        def __init__(self,**kwargs):pass
        def start(self):pass
        def write(self,a):return False
        def stop(self):pass
        def abort(self):pass
        def close(self):
            closes.append(True)
            if len(closes)==1:raise OSError('simulated close failure')
    monkeypatch.setattr('audio_io.sd.query_devices',lambda *args:{'default_samplerate':24000,'max_output_channels':1})
    monkeypatch.setattr('audio_io.sd.OutputStream',Output)
    player=Player(device=1);client=FakeClient()
    def generate(wav,turn,first,on_audio=None):
        client.calls.append(first)
        turn.text_parts.append('A reply')
        turn.audio_samples=1440
        on_audio(np.full(1440,.1,np.float32))
    client.chat_turn=generate
    a=Assistant(mic=FakeMic(),player=player,front=FakeFront(),client=client,start_muted=True)
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        assert a.error=='Playback failed: simulated close failure'
        assert job.played_samples==1440
        assert not player.playing and player._stream is None
        assert a.state()['recording_safe']
        assert a.state()['transcript'][-1].endswith('[reply incomplete]')
        a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        assert a.error is None and a.state()['recording_safe']
        assert client.calls==[True,True] and len(closes)==2
    finally:a.close()
