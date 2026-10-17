"""Mic lifecycle failures: exercise actual Mic with a controlled fake stream."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audio_io import Mic


@pytest.fixture
def mic(monkeypatch):
    monkeypatch.setattr('audio_io.sd.query_devices',lambda *args:{'default_samplerate':48000,'max_input_channels':1,'hostapi':0})
    monkeypatch.setattr('audio_io.sd.query_hostapis',lambda *args:{'name':'Windows WASAPI'})
    return Mic(device=1)


@pytest.mark.parametrize('failure',['stop','close','both'])
def test_stop_attempts_close_and_clears_reference_after_error(mic,failure):
    calls=[]
    class Broken:
        def stop(self):
            calls.append('stop')
            if failure in ['stop','both']:raise OSError('input stop failed')
        def close(self):
            calls.append('close')
            if failure in ['close','both']:raise OSError('input close failed')
    mic.stream=Broken()
    expected='input close failed' if failure=='close' else 'input stop failed'
    with pytest.raises(OSError,match=expected) as caught:mic.stop()
    if failure=='both':assert any('input close failed' in note for note in caught.value.__notes__)
    assert calls==['stop','close']
    assert mic.stream is None
    mic.stop()  # Already stopped: no repeated access to the broken stream.
    assert calls==['stop','close']


def test_start_preserves_primary_error_and_retries_with_fresh_stream(mic,monkeypatch):
    calls=[]
    class Broken:
        def start(self):calls.append('start');raise OSError('input start failed')
        def close(self):calls.append('close');raise OSError('input cleanup failed')
    monkeypatch.setattr('audio_io.sd.InputStream',lambda **kwargs:Broken())
    with pytest.raises(OSError,match='input start failed') as caught:mic.start()
    assert any('input cleanup failed' in note for note in caught.value.__notes__)
    assert mic.stream is None and calls==['start','close']
    class Healthy:
        active=True
        latency=.04
        def start(self):calls.append('new start')
        def stop(self):calls.append('new stop')
        def close(self):calls.append('new close')
    monkeypatch.setattr('audio_io.sd.InputStream',lambda **kwargs:Healthy())
    mic.start();assert mic.stream.active and mic.latency_s==.04
    mic.start();assert calls.count('new start')==1
    mic.stop();assert mic.stream is None
    assert calls[-2:]==['new stop','new close']


def test_inactive_stream_close_failure_does_not_keep_stale_reference(mic):
    class Inactive:
        active=False
        def close(self):raise OSError('inactive close failed')
    mic.stream=Inactive()
    with pytest.raises(OSError,match='inactive close failed'):mic.start()
    assert mic.stream is None
