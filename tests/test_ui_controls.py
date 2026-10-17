"""Actual browser-script controls with delayed and failed server responses."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ui import PAGE


HARNESS=r'''
const vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const nodes={},requests=[],timers=[];
const state={status:'listening',phase:'listening',mic_on:false,mic:'test',server_ok:true,mics:[],mic_index:1,transcript:[],transcript_revision:0,level:0,warning:'Recording warning'};
function element(){return {style:{},options:[],disabled:false,appendChild(){},getContext(){return {fillRect(){}}}};}
let release,releaseState,stateRequests=0;
const context={AbortController,clearTimeout(id){if(timers[id])timers[id].cleared=true;},setTimeout(fn,ms){timers.push({fn,ms});return timers.length-1;},document:{getElementById(id){return nodes[id]??=element();},createElement:element},fetch:async(path,options={})=>{
 if(path==='/api/state'){
  stateRequests++;
  if(input.scenario==='stale_poll'&&stateRequests>1)return new Promise(resolve=>{releaseState=()=>resolve({ok:true,json:async()=>state});});
  return {ok:true,json:async()=>state};
 }
 requests.push({path,body:options.body?JSON.parse(options.body):null});
 if(input.scenario==='retry'&&requests.length===1)return {ok:false,status:409,json:async()=>({error:'First attempt failed'})};
 if(input.scenario==='retry'||input.scenario==='successive')return {ok:true,status:200,json:async()=>({ok:true})};
 if(path==='/api/interrupt')return {ok:true,status:200,json:async()=>({ok:true})};
 if(input.scenario==='http_error')return {ok:false,status:409,json:async()=>({error:'Microphone unavailable'})};
 if(input.scenario==='network_error')throw Error('connection lost');
 if(input.scenario==='timeout')return new Promise((resolve,reject)=>{options.signal?.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'})));});
 return new Promise(resolve=>{release=()=>resolve({ok:true,status:200,json:async()=>({ok:true})});});
}};
(async()=>{
 vm.runInNewContext(input.script,context);
 await new Promise(setImmediate);
 if(input.scenario==='stale_poll'){
  const first=nodes.micbtn.onclick();await new Promise(setImmediate);
  const poll=context.tick();await new Promise(setImmediate);
  release();await first;releaseState();await poll;
  const next=nodes.micbtn.onclick();await new Promise(setImmediate);release();await next;
  process.stdout.write(JSON.stringify({requests}));
 }else if(input.scenario==='retry'||input.scenario==='successive'){
  await nodes.micbtn.onclick();
  const firstBanner=nodes.banner.textContent;
  await nodes.micbtn.onclick();
  process.stdout.write(JSON.stringify({requests,firstBanner,banner:nodes.banner.textContent,disabled:nodes.micbtn.disabled}));
 }else if(input.scenario==='duplicates'){
  const first=nodes.micbtn.onclick();const second=nodes.micbtn.onclick();
  await new Promise(setImmediate);
  const disabled=nodes.micbtn.disabled;
  await nodes.intbtn.onclick();
  if(release)release();
  await Promise.all([first,second]);
  process.stdout.write(JSON.stringify({requests,disabled,enabledAfter:!nodes.micbtn.disabled}));
 }else if(input.scenario==='timeout'){
  const pending=nodes.micbtn.onclick();await new Promise(setImmediate);
  const timer=timers.find(x=>x.ms>=5000&&!x.cleared);
  if(timer)timer.fn();
  await new Promise(setImmediate);
  if(timer)await pending;
  process.stdout.write(JSON.stringify({timed:!!timer,disabled:nodes.micbtn.disabled,banner:nodes.banner.textContent}));
 }else{
  let escaped=null;try{await nodes.micbtn.onclick();}catch(e){escaped=e.message;}
  await context.tick();
  process.stdout.write(JSON.stringify({requests,escaped,banner:nodes.banner.textContent,disabled:nodes.micbtn.disabled}));
 }
})().catch(e=>{process.stderr.write(e.stack);process.exitCode=1});
'''


def run_browser(scenario):
    script=PAGE.split('<script>',1)[1].split('</script>',1)[0]
    result=subprocess.run(['node','-e',HARNESS],input=json.dumps({'script':script,'scenario':scenario}),text=True,capture_output=True,timeout=10,check=True)
    return json.loads(result.stdout)


pytestmark=pytest.mark.skipif(not shutil.which('node'),reason='Node required for actual UI script')


@pytest.mark.parametrize('scenario,message',[('http_error','Microphone unavailable'),('network_error','connection lost')])
def test_action_failure_stays_visible_alongside_recording_warning(scenario,message):
    result=run_browser(scenario)
    assert result['escaped'] is None
    assert message in result['banner']
    assert 'Recording warning' in result['banner']
    assert not result['disabled']


def test_duplicate_mic_click_is_idempotent_and_does_not_block_interrupt():
    result=run_browser('duplicates')
    requests=[r for r in result['requests'] if r['path']=='/api/mic']
    assert len(requests)==1
    assert requests[0]['body']=={'enabled':True}
    assert result['disabled'] and result['enabledAfter']
    assert any(r['path']=='/api/interrupt' for r in result['requests'])


def test_unresponsive_action_times_out_and_releases_control():
    result=run_browser('timeout')
    assert result['timed'] and not result['disabled']
    assert 'timed out' in result['banner'].lower()


def test_successive_mic_intents_use_desired_state_before_next_poll():
    result=run_browser('successive')
    assert [r['body'] for r in result['requests']]==[{'enabled':True},{'enabled':False}]
    assert not result['disabled']


def test_successful_retry_clears_action_error_but_retains_recording_warning():
    result=run_browser('retry')
    assert 'First attempt failed' in result['firstBanner']
    assert 'First attempt failed' not in result['banner']
    assert 'Recording warning' in result['banner']
    assert [r['body'] for r in result['requests']]==[{'enabled':True},{'enabled':True}]


def test_delayed_poll_started_during_action_cannot_reverse_next_mic_intent():
    result=run_browser('stale_poll')
    assert [r['body'] for r in result['requests']]==[{'enabled':True},{'enabled':False}]
