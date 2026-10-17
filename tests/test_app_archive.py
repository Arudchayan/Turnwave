"""Application/archive boundaries, using generated fixtures and fake devices."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave
import pytest
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from app import Assistant, Handler
from client import ServerClient
from session_archive import SessionArchive
from test_realtime import FakeMic, FakePlayer, FakeFront, FakeClient, wait_for, audio_event


def make_app(tmp_path, client=None, archive=None):
    return Assistant(mic=FakeMic(), player=FakePlayer(), front=FakeFront(),
                     client=client or FakeClient(), start_muted=True,
                     session_archive=archive or SessionArchive(tmp_path/'turns'))


def test_actual_client_archives_input_request_generated_and_played(tmp_path):
    client=ServerClient(); requests=[]
    data=('data: '+json.dumps(audio_event('A complete reply.'))+
          '\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n').encode()
    def post(payload):
        requests.append(payload)
        pieces=iter([data,b''])
        return SimpleNamespace(read1=lambda n:next(pieces,b''),close=lambda:None)
    client._post=post
    a=make_app(tmp_path,client)
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        folder=tmp_path/'turns/turn-1'
        assert (folder/'input.wav').read_bytes()==job.wav
        assert json.loads((folder/'request.json').read_text())==requests[0]
        result=json.loads((folder/'status.json').read_text())
        assert result['checkpoint_safe'] and result['generation']['text']=='A complete reply.'
        assert result['generated_samples']==result['played_samples']==1440
        with wave.open(str(folder/'generated.wav'),'rb') as reader:
            assert reader.getnframes()==1440 and reader.getframerate()==24000
        assert a.state()['recording_safe'] and not a.state()['archive_error']
    finally:a.close()


def test_cancel_before_producer_still_finalizes_input(tmp_path):
    a=make_app(tmp_path)
    a._play_lock.acquire()
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:(tmp_path/'turns/turn-1/turn.json').exists())
        a.interrupt()
    finally:a._play_lock.release()
    try:
        wait_for(lambda:a.state()['pending_turns']==0)
        result=json.loads((tmp_path/'turns/turn-1/status.json').read_text())
        assert result['checkpoint_safe'] and result['generation']['cancelled']
        assert result['played_samples']==result['generated_samples']==0
        assert not (tmp_path/'turns/turn-1/request.json').exists()
        assert not a.client.calls
    finally:a.close()


def test_cancelled_late_producer_prevents_premature_recording_stop(tmp_path):
    release=threading.Event(); client=FakeClient()
    def delayed(wav,turn,first,on_audio=None):
        on_audio(np.ones(1440,np.float32)*.1)
        assert release.wait(3)
    client.chat_turn=delayed
    a=make_app(tmp_path,client)
    try:
        job=a.submit_audio(np.ones(8000,np.float32)*.1)
        wait_for(lambda:a.status=='speaking')
        a.interrupt()
        assert job.playback_closed.wait(1)
        state=a.state()
        assert not state['active_turn'] and state['pending_turns']==1
        assert state['archive_pending']==1 and not state['recording_safe']
        result=json.loads((tmp_path/'turns/turn-1/status.json').read_text())
        assert not result['checkpoint_safe']
        release.set()
        wait_for(lambda:a.state()['recording_safe'])
        assert json.loads((tmp_path/'turns/turn-1/status.json').read_text())['checkpoint_safe']
    finally:release.set();a.close()


@pytest.mark.parametrize('stage', ['begin', 'write'])
def test_storage_failure_stays_visible_without_breaking_voice(tmp_path, stage):
    class BrokenArchive:
        folder=tmp_path/'unavailable'
        def begin_turn(self,*args,**kwargs):
            if stage=='begin':raise OSError('simulated storage failure')
            turn=real.begin_turn(*args,**kwargs)
            def fail(samples):raise OSError('simulated storage failure')
            turn.write_audio=fail
            return turn
    real=SessionArchive(tmp_path/'real')
    a=make_app(tmp_path,archive=BrokenArchive())
    try:
        job=a.submit_audio(np.ones(8000,np.float32)*.1)
        wait_for(lambda:a._active is None)
        a.warning='simulated frontend warning'
        state=a.state()
        assert not job.turn.error and job.played_samples>0
        assert 'storage failure' in state['archive_error'] and 'storage failure' in state['warning']
        assert 'frontend warning' in state['warning']
        assert state['archive_pending']==1 and not state['recording_safe']
    finally:a.close()


def test_mute_during_frontend_processing_cannot_submit_stale_capture(tmp_path):
    entered=threading.Event(); release=threading.Event(); finished=threading.Event()
    class SlowFront(FakeFront):
        def update(self,audio):
            self.calls+=1
            if self.calls==1:return True,True,False,audio
            entered.set()
            assert release.wait(3)
            finished.set()
            return False,False,True,audio
    a=Assistant(mic=FakeMic(),player=FakePlayer(),front=SlowFront(),client=FakeClient())
    try:
        a._spawn(a._loop)
        a.mic.q.put(np.ones(3200,np.float32)*.1)
        wait_for(lambda:a.status=='recording')
        a.mic.q.put(np.zeros(3200,np.float32))
        assert entered.wait(1)
        a.set_mic_enabled(False);a.set_mic_enabled(False)
        assert not a.mic_on and a.state()['recording_safe']
        release.set();assert finished.wait(1)
        a._stop.set()
        for thread in a._threads:thread.join(1)
        assert not a.client.calls and not a._pending_jobs
    finally:release.set();a.close()


def test_generated_audio_is_retained_even_when_not_played(tmp_path):
    a=make_app(tmp_path)
    class Unplayed:
        playing=False
        def stop(self):self.playing=False
        def play_chunks(self,chunks,cancel,on_start,on_play,on_written=None):
            self.playing=True
            cancel.wait(3)
            self.playing=False
    def generate(wav,turn,first,on_audio=None):
        turn.audio_samples=5760
        on_audio(np.full(5760,.2,np.float32))
    a.client.chat_turn=generate;a.player=Unplayed()
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        assert job.generation_closed.wait(1)
        assert not a.state()['recording_safe']
        a.interrupt()
        wait_for(lambda:a.state()['recording_safe'])
        status=json.loads((tmp_path/'turns/turn-1/status.json').read_text())
        assert status['generated_samples']==5760 and status['played_samples']==0
        with wave.open(str(tmp_path/'turns/turn-1/generated.wav'),'rb') as reader:
            assert reader.getnframes()==5760
    finally:a.close()


def test_desired_mute_api_is_idempotent_and_validates_boolean(tmp_path):
    a=make_app(tmp_path)
    a.set_mic_enabled(True)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.assistant=a
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        for _ in range(2):
            req=urllib.request.Request(base+'/api/mic',data=b'{"enabled":false}')
            with urllib.request.urlopen(req,timeout=2) as response:
                assert json.load(response)=={'ok':True}
            assert not a.mic_on
        for body in [b'[]',b'{"enabled":"false"}']:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base+'/api/mic',data=body),timeout=2)
            assert error.value.code==400 and not a.mic_on
    finally:
        server.shutdown();server.server_close();worker.join(1);a.close()


def test_muting_a_processing_onset_does_not_interrupt_current_reply(tmp_path):
    entered=threading.Event();release=threading.Event();processed=threading.Event()
    class OnsetFront(FakeFront):
        def update(self,audio):
            entered.set()
            assert release.wait(3)
            processed.set()
            return True,True,False,audio
    a=Assistant(mic=FakeMic(),player=FakePlayer(),front=OnsetFront(),client=FakeClient())
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a.status=='speaking')
        a._spawn(a._loop)
        a.mic.q.put(np.full(3200,.1,np.float32))
        assert entered.wait(1)
        a.set_mic_enabled(False)
        release.set();assert processed.wait(1)
        wait_for(lambda:a._active is None)
        assert not job.turn.cancel.is_set()
        assert not any(e.get('reason')=='barge-in' for e in a.diagnostics.recent())
    finally:release.set();a.close()
