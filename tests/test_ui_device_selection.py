"""Actual PAGE device selection, with realistic select semantics in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ui import PAGE


HARNESS = r'''
const vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const nodes={},requests=[],timers=[];
let nextState={status:'listening',phase:'listening',mic_on:false,mic:'Neutral microphone',server_ok:true,mics:[[2,'Neutral A'],[7,'Neutral B']],mic_index:2,transcript:[],transcript_revision:0,level:0,warning:'Neutral recording warning'};
class Element{
 constructor(tag='div'){this.tag=tag;this.style={};this.disabled=false;this.children=[];this._value='';this._text='';this._html='';}
 get options(){return this.children;}
 set value(value){value=String(value);this._value=this.tag==='select'&&!this.options.some(o=>o.value===value)?'':value;}
 get value(){return this._value;}
 set textContent(value){this._text=String(value);}
 get textContent(){return this._text;}
 set innerHTML(value){this._html=String(value);this.children=[];if(this.tag==='select')this._value='';}
 get innerHTML(){return this._html;}
 appendChild(child){this.children.push(child);if(this.tag==='select'&&this.children.length===1)this._value=child.value;return child;}
 getContext(){return {fillRect(){}};}
}
const node=id=>nodes[id]??=new Element(id==='micsel'?'select':'div');
const flush=()=>new Promise(setImmediate);
let deferGet=false,releaseGet,releasePost,postMode='success',devicePosts=0;
function response(ok=true){return {ok,status:ok?200:409,json:async()=>ok?{ok:true}:{error:'Neutral device unavailable'}};}
const context=vm.createContext({AbortController,document:{getElementById:node,createElement:tag=>new Element(tag)},
 clearTimeout(id){if(timers[id])timers[id].cleared=true;},setTimeout(fn,ms){timers.push({fn,ms});return timers.length-1;},
 fetch:async(path,options={})=>{
  if(path==='/api/state'){
   const snapshot=JSON.parse(JSON.stringify(nextState));
   const value={ok:true,json:async()=>snapshot};
   if(deferGet){deferGet=false;return new Promise(resolve=>{releaseGet=()=>resolve(value);});}
   return value;
  }
  requests.push({path,body:options.body?JSON.parse(options.body):null});
  if(path!=='/api/micdev')return response();
  devicePosts++;
  if(postMode==='fail')return response(false);
  if(postMode==='pending'||postMode==='timeout')return new Promise((resolve,reject)=>{
   releasePost=()=>resolve(response());
   options.signal?.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'})));
  });
  return response();
 }});
async function change(index){node('micsel').value=index;return node('micsel').onchange({target:node('micsel')});}
const view=()=>({options:node('micsel').options.map(o=>[o.value,o.textContent]),value:node('micsel').value,disabled:node('micsel').disabled,banner:node('banner').textContent});
(async()=>{
 if(input.scenario==='empty')nextState.mics=[];
 if(input.scenario==='unknown')nextState.mic_index=99;
 if(input.scenario==='unicode')nextState.mics=[[2,input.label],[7,input.label]];
 vm.runInContext(input.script,context);await flush();
 const initial=view();let extra={};
 if(input.scenario==='roster'){
  nextState.mics=input.roster;nextState.mic_index=input.selected;await context.tick();
 }else if(input.scenario==='pending'){
  postMode='pending';const task=change(7);await flush();
  const disabled=node('micsel').disabled;await context.tick();extra.during=view();
  await node('micsel').onchange({target:node('micsel')});await node('intbtn').onclick();
  releasePost();await task;extra.disabledDuring=disabled;
 }else if(input.scenario==='stale_before'||input.scenario==='stale_during'){
  let task,poll;
  if(input.scenario==='stale_before'){deferGet=true;poll=context.tick();await flush();task=change(7);await task;}
  else{postMode='pending';task=change(7);await flush();deferGet=true;poll=context.tick();await flush();releasePost();await task;}
  extra.beforeStale=view();releaseGet();await poll;extra.afterStale=view();
  nextState.mic_index=7;await context.tick();
 }else if(input.scenario==='failure'){
  postMode='fail';await change(7);extra.failed=view();postMode='success';await change(7);
 }else if(input.scenario==='failed_stale'){
  nextState.mic_index=7;deferGet=true;const poll=context.tick();await flush();
  postMode='fail';await change(7);extra.failed=view();releaseGet();await poll;extra.afterStale=view();
  await context.tick();extra.afterFresh=view();postMode='success';await change(2);
 }else if(input.scenario==='timeout'){
  postMode='timeout';const task=change(7);await flush();
  const timer=timers.find(t=>t.ms===7000&&!t.cleared);if(timer)timer.fn();await task;
  extra.timed=!!timer;extra.failed=view();postMode='success';await change(7);
 }else if(input.scenario==='empty'||input.scenario==='unknown'){
  await node('micsel').onchange({target:node('micsel')});
 }else if(input.scenario==='unicode'){
  await change(7);await node('micbtn').onclick();
 }
 process.stdout.write(JSON.stringify({initial,...extra,final:view(),requests,devicePosts}));
})().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
'''


def browser(scenario, **fields):
    script = PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
    result = subprocess.run(['node', '-e', HARNESS],
                            input=json.dumps(dict(script=script, scenario=scenario, **fields)),
                            text=True, encoding='utf-8', capture_output=True, timeout=10, check=True)
    return json.loads(result.stdout)


pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='Node required for actual PAGE script')


@pytest.mark.parametrize('roster,selected', [
    ([[3, 'Neutral C'], [8, 'Neutral D']], 8),
    ([[2, 'Renamed A'], [7, 'Renamed B']], 7),
    ([[7, 'Neutral B'], [2, 'Neutral A']], 2),
    ([], -1),
])
def test_full_ordered_roster_updates_even_when_option_count_is_unchanged(roster, selected):
    result = browser('roster', roster=roster, selected=selected)
    assert result['final']['options'] == [[str(i), name] for i, name in roster]
    assert result['final']['value'] == (str(selected) if roster else '')


def test_pending_device_choice_survives_poll_and_does_not_block_interrupt():
    result = browser('pending')
    assert result['disabledDuring'] and result['during']['value'] == '7'
    assert result['final']['value'] == '7' and not result['final']['disabled']
    assert result['devicePosts'] == 1
    assert any(r['path'] == '/api/interrupt' for r in result['requests'])


@pytest.mark.parametrize('scenario', ['stale_before', 'stale_during'])
def test_late_poll_cannot_revert_completed_selection(scenario):
    result = browser(scenario)
    assert result['beforeStale']['value'] == '7'
    assert result['afterStale']['value'] == result['final']['value'] == '7'
    assert result['devicePosts'] == 1


def test_failed_change_reverts_confirmed_choice_and_can_retry():
    result = browser('failure')
    assert result['failed']['value'] == '2' and not result['failed']['disabled']
    assert 'Neutral device unavailable' in result['failed']['banner']
    assert result['final']['value'] == '7' and 'unavailable' not in result['final']['banner']
    assert 'Neutral recording warning' in result['final']['banner']
    assert [r['body'] for r in result['requests']] == [{'index': 7}, {'index': 7}]


def test_failed_intent_rejects_stale_poll_but_accepts_fresh_state_and_retry():
    result = browser('failed_stale')
    assert result['failed']['value'] == result['afterStale']['value'] == '2'
    assert result['afterFresh']['value'] == '7'
    assert 'Neutral device unavailable' in result['afterFresh']['banner']
    assert result['final']['value'] == '2'
    assert [r['body'] for r in result['requests']] == [{'index': 7}, {'index': 2}]


def test_device_timeout_releases_select_and_allows_retry():
    result = browser('timeout')
    assert result['timed'] and not result['failed']['disabled']
    assert result['failed']['value'] == '2' and 'timed out' in result['failed']['banner']
    assert result['final']['value'] == '7' and result['devicePosts'] == 2


@pytest.mark.parametrize('scenario', ['empty', 'unknown'])
def test_missing_choice_never_submits_a_fabricated_zero_index(scenario):
    result = browser(scenario)
    assert result['initial']['value'] == ''
    assert result['devicePosts'] == 0


def test_duplicate_unicode_labels_remain_distinct_and_mic_control_is_independent():
    label = 'Neutral 🎙️ العربية 日本語 <label> ' + 'x' * 90
    result = browser('unicode', label=label)
    assert result['final']['options'] == [['2', label], ['7', label]]
    assert result['final']['value'] == '7'
    assert [r for r in result['requests'] if r['path'] == '/api/micdev'] == [{'path': '/api/micdev', 'body': {'index': 7}}]
    assert any(r['path'] == '/api/mic' and r['body'] == {'enabled': True} for r in result['requests'])
