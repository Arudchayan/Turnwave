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
const state={status:'listening',phase:'listening',mic_on:false,mic:'test',server_ok:true,mics:[],mic_index:1,transcript:[],transcript_revision:0,level:0,warning:'Recording warning',end_silence_ms:480};
function element(){return {style:{},options:[],children:[],disabled:false,set innerHTML(value){this.children=[];},appendChild(c){this.children.push(c);},getContext(){return {fillRect(){}}}};}
let release,releaseState,stateRequests=0;
const context={AbortController,URL,clearTimeout(id){if(timers[id])timers[id].cleared=true;},setTimeout(fn,ms){timers.push({fn,ms});return timers.length-1;},document:{getElementById(id){return nodes[id]??=element();},createElement:element},fetch:async(path,options={})=>{
 if(path==='/api/state'){
  stateRequests++;
  if(input.scenario==='stale_poll'&&stateRequests>1)return new Promise(resolve=>{releaseState=()=>resolve({ok:true,json:async()=>state});});
  return {ok:true,json:async()=>state};
 }
 requests.push({path,body:options.body?JSON.parse(options.body):null});
 if(input.scenario==='search_failure'&&path==='/api/search-provider')return {ok:false,status:400,json:async()=>({error:'Brave search requires an API key'})};
 if(input.scenario==='retry'&&requests.length===1)return {ok:false,status:409,json:async()=>({error:'First attempt failed'})};
 if(input.scenario==='retry'||input.scenario==='successive'||input.scenario==='pause'||input.scenario==='prompt'||input.scenario==='agent_persona_notes'||input.scenario==='voice_tuning'||input.scenario==='search_selection')return {ok:true,status:200,json:async()=>({ok:true})};
 if(path==='/api/interrupt')return {ok:true,status:200,json:async()=>({ok:true})};
 if(input.scenario==='http_error')return {ok:false,status:409,json:async()=>({error:'Microphone unavailable'})};
 if(input.scenario==='network_error')throw Error('connection lost');
 if(input.scenario==='timeout')return new Promise((resolve,reject)=>{options.signal?.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'})));});
 return new Promise(resolve=>{release=()=>resolve({ok:true,status:200,json:async()=>({ok:true})});});
}};
(async()=>{
 vm.runInNewContext(input.script,context);
 await new Promise(setImmediate);
 if(input.scenario==='pause'){
  await nodes.pausesel.onchange({target:{value:'800'}});
  process.stdout.write(JSON.stringify({requests,value:nodes.pausesel.value}));
 }else if(input.scenario==='prompt'){
  const inp=context.document.getElementById('promptin');
  inp.value='   Explain quantum computing   ';
  await nodes.promptform.onsubmit({preventDefault(){}});
  process.stdout.write(JSON.stringify({requests,inputAfter:inp.value}));
 }else if(input.scenario==='agent_persona_notes'){
  await context.tick();
  const personaDisabledInitially=nodes.personasel.disabled;
  await nodes.agentsel.onchange({target:{value:'opencode-go:kimi-k3'}});
  Object.assign(state,{
   agent_mode:true,agent_provider:'OpenCode Go',agent_model:'kimi-k3',agent_persona:'jarvis',
   agent_notes:[{model:'kimi-k3',task:'Architecture',analysis:'FlashAttention-2 SRAM tiling'}]
  });
  await context.tick();
  const personaDisabledAfter=nodes.personasel.disabled;
  const personaVal=nodes.personasel.value;
  await nodes.personasel.onchange({target:{value:'concise'}});
  const notesHdr=nodes.notes.children[0]?.textContent;
  const cardHdr=nodes.notes.children[1]?.children[0]?.textContent;
  const cardBody=nodes.notes.children[1]?.children[1]?.textContent;
  process.stdout.write(JSON.stringify({
   requests,personaDisabledInitially,personaDisabledAfter,personaVal,
   finalPersona:nodes.personasel.value,notes:{notesHdr,cardHdr,cardBody}
  }));
 }else if(input.scenario==='search_failure'){
  Object.assign(state,{agent_mode:true,agent_provider:'OpenCode Zen',agent_model:'kimi-k3',search_provider:'ddg'});
  await context.tick();
  await nodes.searchsel.onchange({target:{value:'brave'}});
  process.stdout.write(JSON.stringify({value:nodes.searchsel.value,banner:nodes.banner.textContent}));
 }else if(input.scenario==='search_selection'){
  Object.assign(state,{agent_mode:true,agent_provider:'OpenCode Zen',agent_model:'kimi-k3',search_provider:'ddg'});
  await context.tick();
  await nodes.searchsel.onchange({target:{value:'bing'}});
  await nodes.agentsel.onchange({target:{value:'opencode-zen:kimi-k3'}});
  process.stdout.write(JSON.stringify({requests,searchValue:nodes.searchsel.value,disabled:nodes.searchsel.disabled}));
 }else if(input.scenario==='voice_tuning'){
  await nodes.vpreset.onchange({target:{value:'crisp'}});
  nodes.v_temp.value='0.65';
  nodes.v_voice.value='alloy';
  await nodes.vapplybtn.onclick();
  process.stdout.write(JSON.stringify({requests,status:nodes.vstatus.textContent}));
 }else if(input.scenario==='agent_privacy'){
  Object.assign(state,{agent_mode:true,agent_provider:'OpenCode Go',agent_model:'beam-free',search_enabled:true,search_provider:'ddg',agent_phase:'searching',agent_sources:[{title:'Valid Source',url:'https://example.org/item'},{title:'Bad',url:'javascript:alert(1)'}],audio_diagnostics:{input_gain:1,input_drops:0,echo_score:0.12,residual_rejections:2,playback:{underflows:0,max_gap_ms:4.5}}});
  await context.tick();
  const links=nodes.sources.children.slice(1).map(row=>({href:row.children[0].href,text:row.children[0].textContent}));
  const beamMode=nodes.mode.textContent;
  Object.assign(state,{agent_model:'iceman-contributor-free'});
  await context.tick();
  const icemanMode=nodes.mode.textContent;
  Object.assign(state,{agent_mode:false});
  await context.tick();
  process.stdout.write(JSON.stringify({mode:beamMode,icemanMode,localMode:nodes.mode.textContent,links,diag:nodes.diag.textContent}));
 }else if(input.scenario==='activity'){
  Object.assign(state,{agent_mode:true,active_turn:true,agent_phase:'searching'});
  await context.tick();
  const search={label:nodes['activity-text'].textContent,visible:nodes.activity.className};
  state.agent_phase='thinking';await context.tick();
  const thinking={label:nodes['activity-text'].textContent,visible:nodes.activity.className};
  state.active_turn=false;await context.tick();
  process.stdout.write(JSON.stringify({search,thinking,after:nodes.activity.className}));
 }else if(input.scenario==='historical_sources'){
  Object.assign(state,{agent_mode:true,agent_sources:[{title:'Duplicate',url:'https://duplicate.example/'}],
   transcript_revision:1,messages:[
    {text:'user: First question',sources:[]},
    {text:'assistant: First answer',sources:[{title:'<b>First</b>',url:'https://example.org/first'},
     {title:'Unsafe',url:'javascript:alert(1)'},{title:'Credential',url:'https://user@example.org/'}]}]});
  await context.tick();
  const first=nodes.log.children[1].children[0].children.slice(1).map(a=>({href:a.href,text:a.textContent}));
  state.messages.push({text:'user: Second question',sources:[]},{text:'assistant: Second answer',
   sources:[{title:'Second',url:'https://example.org/second'}]});
  state.transcript_revision=2;
  await context.tick();
  const replies=nodes.log.children.filter(c=>c.className==='msg a').map(c=>({text:c.textContent,
   links:c.children[0].children.slice(1).map(a=>({href:a.href,text:a.textContent}))}));
  process.stdout.write(JSON.stringify({first,replies,globalSources:nodes.sources.children.length}));
 }else if(input.scenario==='stale_poll'){
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
    result=subprocess.run(['node','-e',HARNESS],input=json.dumps({'script':script,'scenario':scenario}),text=True,encoding='utf-8',capture_output=True,timeout=10,check=True)
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


def test_pause_selection_sends_requested_duration():
    result=run_browser('pause')
    assert result['requests']==[{'path':'/api/pause','body':{'milliseconds':800}}]
    assert str(result['value'])=='800'


def test_agent_mode_sources_and_diagnostics_render_safely():
    result=run_browser('agent_privacy')
    assert 'OpenCode Go (beam-free)' in result['mode']
    assert '30-day data retention' in result['mode']
    assert 'uses prompts and responses to improve this model' in result['icemanMode']
    assert 'improve this model' not in result['localMode']
    assert 'enabled (ddg)' in result['mode'] and 'searching' in result['mode']
    assert result['links']==[{'href':'https://example.org/item','text':'Valid Source'}]
    assert 'underflows 0' in result['diag'] and 'residual rejections 2' in result['diag']


def test_model_choices_disclose_provider_data_terms_before_selection():
    assert 'Go Beam Free (30-day retention)' in PAGE
    assert 'Go Iceman Free (data used to improve model)' in PAGE
    assert 'https://opencode.ai/v2/docs/console/go/#privacy' in PAGE


def test_each_answer_retains_its_own_safe_source_links():
    result = run_browser('historical_sources')
    assert result['first'] == [{'href': 'https://example.org/first', 'text': '<b>First</b>'}]
    assert result['replies'] == [
        {'text': 'First answer', 'links': [{'href': 'https://example.org/first', 'text': '<b>First</b>'}]},
        {'text': 'Second answer', 'links': [{'href': 'https://example.org/second', 'text': 'Second'}]},
    ]
    assert result['globalSources'] == 0


def test_prompt_submission_trims_text_dispatches_api_and_clears_input():
    result=run_browser('prompt')
    assert result['requests']==[{'path':'/api/prompt','body':{'text':'Explain quantum computing'}}]
    assert result['inputAfter']==''


def test_agent_mode_controls_persona_dropdown_and_renders_notes():
    result=run_browser('agent_persona_notes')
    assert result['personaDisabledInitially'] is True
    assert result['personaDisabledAfter'] is False
    assert result['personaVal']=='jarvis'
    assert result['finalPersona']=='concise'
    assert result['requests']==[
        {'path':'/api/agent','body':{'mode':'opencode-go','model':'kimi-k3'}},
        {'path':'/api/persona','body':{'persona':'concise'}}
    ]
    assert result['notes']['notesHdr']=='Expert Deep-Dive Brief'
    assert result['notes']['cardHdr']=='kimi-k3 · Architecture'
    assert 'FlashAttention-2 SRAM tiling' in result['notes']['cardBody']


def test_search_and_zen_controls_dispatch_the_selected_values():
    result = run_browser('search_selection')
    assert result['requests'] == [
        {'path': '/api/search-provider', 'body': {'provider': 'bing'}},
        {'path': '/api/agent', 'body': {'mode': 'opencode-zen', 'model': 'kimi-k3'}},
    ]
    assert result['searchValue'] == 'bing'
    assert result['disabled'] is False


def test_search_selection_error_preserves_previous_provider():
    result = run_browser('search_failure')
    assert result['value'] == 'ddg'
    assert 'requires an API key' in result['banner']


def test_voice_tuning_preset_and_manual_apply_dispatch_api():
    result=run_browser('voice_tuning')
    assert len(result['requests'])==2
    assert result['requests'][0]=={
        'path':'/api/voice',
        'body':{'temperature':0.45,'top_k':25,'top_p':0.85,'min_p':0.08,'repeat_penalty':1.08,'voice':''}
    }
    assert result['requests'][1]=={
        'path':'/api/voice',
        'body':{'temperature':0.65,'top_k':25,'top_p':0.85,'min_p':0.08,'repeat_penalty':1.08,'voice':'alloy'}
    }
    assert 'temp 0.65' in result['status'] and 'voice=alloy' in result['status']




def test_reply_sources_render_with_safe_links():
    code = PAGE.split('function bubble(t,sourceItems=[])', 1)[1].split('async function tick()', 1)[0]
    script = 'function bubble(t,sourceItems=[])' + code
    harness = r'''
const vm=require('node:vm'), fs=require('node:fs');
const source=fs.readFileSync(0,'utf8');
function element(){return {children:[],appendChild(x){this.children.push(x);}};}
const ctx={URL,document:{createElement:element}};
vm.runInNewContext(source,ctx);
const bubble=ctx.bubble('assistant: Answer',[
 {title:'Official',url:'https://example.org/article'},
 {title:'Bad',url:'javascript:alert(1)'},
 {title:'Credentials',url:'https://user:pass@example.org/'}]);
const area=bubble.children[0];
process.stdout.write(JSON.stringify({text:bubble.textContent,heading:area.children[0].textContent,
 links:area.children.slice(1)}));
'''
    result=subprocess.run(['node','-e',harness],input=script,text=True,capture_output=True,timeout=10,check=True)
    data=json.loads(result.stdout)
    assert data['text']=='Answer' and data['heading']=='Sources'
    assert len(data['links'])==1 and data['links'][0]['href']=='https://example.org/article'
    assert data['links'][0]['rel']=='noopener noreferrer'


def test_agent_activity_tracks_search_thinking_and_completion():
    result = run_browser('activity')
    assert result['search']['visible'] == 'visible'
    assert 'Searching' in result['search']['label']
    assert result['thinking']['visible'] == 'visible'
    assert 'Thinking' in result['thinking']['label']
    assert result['after'] == ''
