"""Exercise real Player plus app/archive with controlled device write outcomes."""
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from app import Assistant
from audio_io import Player
from session_archive import SessionArchive
from test_realtime import FakeMic, FakeFront, FakeClient, wait_for


@pytest.mark.parametrize('accepted_writes',[0,1])
def test_archive_excludes_failed_output_write(monkeypatch,tmp_path,accepted_writes):
    class Output:
        latency=.04
        def __init__(self,**kwargs):self.writes=0
        def start(self):pass
        def write(self,a):
            if self.writes==accepted_writes:raise OSError('simulated device disconnect')
            self.writes+=1
            return False
        def stop(self):pass
        def abort(self):pass
        def close(self):pass
    monkeypatch.setattr('audio_io.sd.query_devices',lambda *args:{'default_samplerate':24000,'max_output_channels':1})
    monkeypatch.setattr('audio_io.sd.OutputStream',Output)
    client=FakeClient()
    def generate(wav,turn,first,on_audio=None):
        turn.text_parts.append('A partial reply')
        turn.audio_samples=1440
        on_audio(np.full(1440,.1,np.float32))
    client.chat_turn=generate
    player=Player(device=1)
    a=Assistant(mic=FakeMic(),front=FakeFront(),player=player,client=client,start_muted=True,session_archive=SessionArchive(tmp_path/'turns'))
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        status=json.loads((tmp_path/'turns/turn-1/status.json').read_text())
        assert status['checkpoint_safe']
        assert status['generated_samples']==1440
        assert status['played_samples']==job.played_samples==player.stats['source_samples']==accepted_writes*480
        assert status['playback']['error']=='Playback failed: simulated device disconnect'
        assert a.state()['transcript'][-1].endswith('[reply incomplete]')
    finally:a.close()


@pytest.mark.parametrize('cancel_during_write',[False,True])
def test_successful_write_accounting_preserves_reference_order(monkeypatch,cancel_during_write):
    events=[];cancel=threading.Event()
    class Output:
        latency=.04
        def __init__(self,**kwargs):pass
        def start(self):pass
        def write(self,a):
            events.append(('write',len(a)))
            if cancel_during_write:cancel.set()
            return False
        def stop(self):pass
        def close(self):pass
    monkeypatch.setattr('audio_io.sd.query_devices',lambda *args:{'default_samplerate':48000,'max_output_channels':2})
    monkeypatch.setattr('audio_io.sd.OutputStream',Output)
    player=Player(device=1)
    source=np.linspace(-.2,.2,1001,dtype=np.float32)
    accepted=[]
    player.play_chunks([source],cancel=cancel,
        on_play=lambda a:events.append(('reference',len(a))),
        on_written=lambda a:(events.append(('accepted',len(a))),accepted.append(a.copy())))
    expected=480 if cancel_during_write else len(source)
    assert player.stats['source_samples']==expected
    assert player.stats['device_samples']==expected*2
    np.testing.assert_array_equal(np.concatenate(accepted),source[:expected])
    assert [e[0] for e in events]==['reference','write','accepted']*(1 if cancel_during_write else 3)
