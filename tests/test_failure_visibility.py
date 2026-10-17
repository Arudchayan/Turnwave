"""Failure visibility through real application and browser-script control flow."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from app import Assistant
from ui import PAGE
from test_realtime import FakeMic, FakePlayer, FakeFront, FakeClient, wait_for


def test_playback_failure_is_incomplete_and_survives_recovery():
    class DisconnectedPlayer(FakePlayer):
        def play_chunks(self,chunks,cancel=None,on_start=None,on_play=None,on_written=None):
            next(iter(chunks))
            raise OSError('speaker disconnected')
    a=Assistant(mic=FakeMic(),player=DisconnectedPlayer(),front=FakeFront(),client=FakeClient())
    try:
        job=a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        assert job.turn.cancel.is_set(), 'Playback error must still abort generation'
        assert a.error=='Playback failed: speaker disconnected'
        assert a.first is True
        assert a.state()['transcript'][-1].endswith(' [reply incomplete]')
        a.player=FakePlayer()
        a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a._active is None and a.state()['pending_turns']==0)
        assert a.error is None
        assert any(line.endswith(' [reply incomplete]') for line in a.state()['transcript'])
        assert a.client.calls==[True,True]
    finally:a.close()


@pytest.mark.skipif(not shutil.which('node'),reason='Node required to execute browser script')
@pytest.mark.parametrize('connected,error,warning',[(True,'Playback failed','Recording could not be saved'),(False,None,'Recording could not be saved'),(False,'Playback failed','Recording could not be saved'),(True,None,None)])
def test_browser_does_not_hide_recording_warning(connected,error,warning):
    state=dict(server_ok=connected,error=error,warning=warning,status='listening',mic_on=False,mic='test mic',mics=[],transcript=[],transcript_revision=0,level=0)
    script=PAGE.split('<script>',1)[1].split('</script>',1)[0]
    harness=r'''
const vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const nodes={};
function element(){return {style:{},options:[],classList:{},appendChild(){},getContext(){return {fillRect(){}}}};}
const context={document:{getElementById(id){return nodes[id]??=element();},createElement:element},fetch:async()=>({json:async()=>input.state}),setTimeout(){}};
vm.runInNewContext(input.script,context);
setImmediate(()=>process.stdout.write(JSON.stringify({text:nodes.banner.textContent,display:nodes.banner.style.display})));
'''
    result=subprocess.run(['node','-e',harness],input=json.dumps(dict(script=script,state=state)),text=True,capture_output=True,check=True,timeout=10)
    banner=json.loads(result.stdout)
    assert banner['display']==('block' if not connected or error or warning else 'none')
    if warning:assert warning in banner['text']
    if error:assert error in banner['text']
    if not connected:assert 'offline' in banner['text'].lower()


def test_deliberate_interruption_keeps_its_label():
    a=Assistant(mic=FakeMic(),player=FakePlayer(),front=FakeFront(),client=FakeClient())
    try:
        a.submit_audio(np.full(8000,.1,np.float32))
        wait_for(lambda:a.status=='speaking')
        a.interrupt('button')
        wait_for(lambda:a.state()['pending_turns']==0)
        assert a.error is None
        assert a.state()['transcript'][-1].endswith(' [interrupted]')
    finally:a.close()
