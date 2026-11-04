"""Local browser interface, served by app.py."""
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Turnwave</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--txt:#e6edf3;--dim:#8b949e;
--green:#3fb950;--red:#f85149;--amber:#d29922;--blue:#58a6ff}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;
max-width:760px;margin:0 auto;padding:16px;color:var(--txt);background:var(--bg)}
header{display:flex;align-items:center;gap:10px;margin-bottom:12px}
h1{font-size:1.3em;margin:0}
.dot{width:12px;height:12px;border-radius:50%;background:var(--red);flex:none}
.dot.ok{background:var(--green)}#srvlbl{font-size:.8em;color:var(--dim)}
#st{font-size:1.15em;font-weight:600;padding:.35em .9em;border-radius:999px;
background:var(--card);border:1px solid var(--line)}
#st.listening{border-color:var(--green);color:var(--green)}
#st.recording{border-color:var(--red);color:var(--red);animation:pulse 1s infinite}
#st.thinking{border-color:var(--amber);color:var(--amber)}
#st.speaking{border-color:var(--blue);color:var(--blue)}
@keyframes pulse{50%{opacity:.55}}
#banner{display:none;white-space:pre-line;background:#3a1d1d;border:1px solid var(--red);border-radius:8px;
padding:.6em 1em;margin:10px 0;font-size:.9em}
#wavewrap{background:#000;border-radius:10px;padding:6px;margin:10px 0}
canvas{width:100%;height:64px;display:block}
#lvl{font-size:.8em;color:var(--dim);margin-top:2px}
#controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;align-items:center}
button{font-size:.95em;padding:.5em 1em;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--txt);cursor:pointer}
button:hover{border-color:var(--blue)}
button:disabled,select:disabled{opacity:.55;cursor:wait}
button.big{background:var(--blue);border-color:var(--blue);color:#04121f;font-weight:700}
select{font-size:.9em;padding:.5em;border-radius:8px;background:var(--card);
color:var(--txt);border:1px solid var(--line);max-width:220px}
#cards{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.5em;text-align:center}
.card b{display:block;font-size:1.1em}.card span{font-size:.72em;color:var(--dim)}
#log{display:flex;flex-direction:column;gap:6px;max-height:300px;overflow-y:auto;
background:#000;border-radius:10px;padding:12px}
.msg{max-width:85%;padding:.45em .8em;border-radius:12px;font-size:.93em;line-height:1.35}
.msg.u{align-self:flex-end;background:#1f6feb;color:#fff}
.msg.a{align-self:flex-start;background:var(--card);border:1px solid var(--line)}
.msg.s{align-self:center;color:var(--dim);font-style:italic;font-size:.82em}
.msg.a.draft{border-style:dashed;opacity:.8}
#stages{display:flex;gap:6px;list-style:none;margin:8px 0 0;padding:0;font-size:.82em}
#stages li{padding:.25em .8em;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
#stages li.done{border-color:var(--green);color:var(--green)}
#stages li.active{border-color:var(--amber);color:var(--amber)}
#etawrap{display:none;background:#000;border:1px solid var(--line);border-radius:8px;
height:22px;margin:10px 0;position:relative;overflow:hidden}
#etafill{height:100%;width:0;background:var(--amber);transition:width .15s linear}
#etawrap.hit #etafill{background:var(--green)}
#etalbl{position:absolute;inset:0;text-align:center;font-size:.75em;line-height:22px;color:var(--txt)}
details#metrics{margin:10px 0}details#metrics summary{cursor:pointer;color:var(--dim);font-size:.85em}
footer{margin-top:10px;font-size:.75em;color:var(--dim)}
/* A quiet listening room: keep the controls close, diagnostics in the background. */
:root{--bg:#f4f0e8;--card:#fffdf7;--line:#d7d0c2;--txt:#252b26;--dim:#626e60;--green:#356d51;--red:#ac4c3d;--amber:#94621d;--blue:#355d4b}
body{max-width:940px;padding:clamp(20px,5vw,56px);font-family:Georgia,serif;line-height:1.5}
header{margin-bottom:clamp(28px,6vw,68px);gap:12px}
h1{font-size:1.65rem;letter-spacing:-.06em;font-weight:normal}
#srvlbl,#lvl,button,select,#stages,#banner,#cards,footer,summary{font-family:Segoe UI,sans-serif}
#srvlbl{font-size:.75rem;letter-spacing:.02em}
.dot{width:8px;height:8px}
#st{display:inline-block;font-size:clamp(1.6rem,4vw,2.8rem);font-weight:normal;letter-spacing:-.04em;padding:0;background:none;border:0;border-radius:0;color:var(--txt)!important}
#stages{margin:10px 0 24px;gap:12px;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
#stages li{border:0;border-bottom:2px solid var(--line);border-radius:0;padding:.3em 0}
#banner{background:#f9e9e2;border-color:#c47b6a;color:#702b20}
#wavewrap{background:#e7e8dc;border-radius:4px;padding:14px;margin:20px 0}
#lvl{font-size:.72rem;margin-top:4px}
#controls{gap:8px;margin:24px 0 34px}
#pause-control{display:flex;align-items:center;gap:8px;font:.8rem Segoe UI,sans-serif;color:var(--dim);white-space:nowrap}
button,select{border-radius:4px;background:transparent;padding:.7em 1em;font-size:.85rem}
button:hover{border-color:var(--green);background:#e8ebe0}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #b47439;outline-offset:3px}
button.big{background:var(--green);border-color:var(--green);color:#fffdf7}
button.big:hover{background:#254f3b}
#cards{gap:0;border-top:1px solid var(--line)}
.card{background:none;border:0;border-bottom:1px solid var(--line);border-radius:0;text-align:left;padding:12px 8px}
#log{background:transparent;border-top:1px solid var(--line);border-radius:0;padding:20px 0;gap:14px;min-height:60px}
.msg{border-radius:4px;line-height:1.5;padding:.65em .9em}
.msg.u{background:#dce6d8;color:var(--txt)}
.msg.a{background:var(--card);border-color:var(--line)}
#etawrap{background:#e7e8dc}#etalbl{color:var(--txt)}
#mode{font: .8rem Segoe UI,sans-serif;color:var(--dim);margin:8px 0 0}
#conversation-settings{margin:18px 0;font:.85rem Segoe UI,sans-serif}
#conversation-settings summary{cursor:pointer;color:var(--green)}
#conversation-settings p{color:var(--dim);line-height:1.5}
.settings-grid{display:grid;grid-template-columns:minmax(120px, 1fr) minmax(0, 2fr);align-items:center;gap:10px}
.settings-grid select,.settings-grid input{width:100%;max-width:none}
.settings-grid input{font:inherit;padding:.7em 1em;color:var(--txt);background:var(--card);border:1px solid var(--line);border-radius:4px;min-width:0}
.settings-grid input:focus-visible{outline:3px solid #b47439;outline-offset:3px}
#agentapply{grid-column:2;justify-self:start}
#sources{font: .82rem Segoe UI,sans-serif;line-height:1.6}
#sources a{color:var(--green);overflow-wrap:anywhere}
footer{border-top:1px solid var(--line);padding-top:16px;margin-top:28px}
@media(max-width:560px){#cards{grid-template-columns:repeat(2,1fr)}#controls select{max-width:100%;width:100%}.msg{max-width:95%}}
@media(prefers-reduced-motion:reduce){#st.recording{animation:none}#etafill{transition:none}}
</style></head><body>
<header><div class=dot id=srv></div><div><h1>Turnwave</h1>
<div id=srvlbl>checking server…</div></div></header>
<div id=mode>Checking processing mode…</div>
<div><span id=st>starting…</span></div>
<ol id=stages><li id=st_heard>heard</li><li id=st_under>understanding</li><li id=st_voice>voicing</li></ol>
<div id=banner></div>
<div id=etawrap><div id=etafill></div><span id=etalbl></span></div>
<div id=wavewrap><canvas id=wave width=700 height=64></canvas>
<div id=lvl>mic level —</div></div>
<div id=controls>
<button id=micbtn disabled>checking mic…</button>
<button id=intbtn class=big>interrupt</button>
<button id=newbtn>new conversation</button>
<button id=tstbtn>test speaker</button>
<select id=micsel aria-label="Input device"></select>
<label id=pause-control for=pausesel>Pause before reply
<select id=pausesel><option value=480>Quick · 0.48 s</option><option value=800>Balanced · 0.8 s</option><option value=1120>Patient · 1.12 s</option></select></label>
</div>
<details id=conversation-settings><summary>Conversation and search settings</summary>
<p>Changing the conversation mode or model starts a new conversation. Cloud mode sends recognized speech and conversation history to the selected provider.</p>
<div class=settings-grid><label for=agentsel>Conversation mode</label>
<select id=agentsel disabled><option value=local>Local voice</option><option value=opencode-go>OpenCode Go</option><option value=opencode-zen>OpenCode Zen</option></select>
<label for=agentmodel>Cloud model</label><input id=agentmodel maxlength=100 placeholder="kimi-k3" autocomplete=off>
<button id=agentapply disabled>Apply conversation settings</button>
<label for=searchsel>Web search</label><select id=searchsel disabled><option value=none>Off</option><option value=ddg>DuckDuckGo</option><option value=bing>Bing</option><option value=brave>Brave</option></select>
</div><p>Search sends queries to the selected search provider. Provider API keys must be configured in the app environment.</p>
</details>
<details id=metrics><summary>metrics</summary>
<div id=cards>
<div class=card><b id=m_ttft>—</b><span>time to first audio</span></div>
<div class=card><b id=m_aud>—</b><span>reply audio</span></div>
<div class=card><b id=m_turn>—</b><span>turn time</span></div>
<div class=card><b id=m_n>0</b><span>replies</span></div>
</div></details>
<h3 style="margin:.4em 0">conversation</h3><div id=log role=log aria-live=polite></div>
<div id=draft class="msg a draft" style=display:none></div>
<div id=sources></div>
<footer>Voice uses your configured server · headphones improve interruption handling</footer>
<script>
let lv=[],lastRevision=-1,pollDelay=300;
const STLBL={st_heard:'heard',st_under:'understanding',st_voice:'voicing'};
function setStage(id,ts,activePh,ph){let e=$(id);
e.className=ts!=null?'done':(ph===activePh?'active':'');
e.textContent=(ts!=null?'✓ ':'')+STLBL[id];}
const $=id=>document.getElementById(id);
let actionError=null,lastState=null,lastMicOn=null,micVersion=0;
let lastDeviceIndex=null,deviceVersion=0;
let lastPause=null,pauseVersion=0;
let settingsVersion=0,settingsBusy=false,settingsState=null,settingsDirty=false;
const pendingActions=new Set();
function renderAlerts(){
let s=lastState;
let alerts=[s&&!s.server_ok?'The local voice server is offline. Start it to continue.':null,
s?.error,s?.warning,actionError].filter(Boolean);
$('banner').style.display=alerts.length?'block':'none';
$('banner').textContent=alerts.join('\\n');}
async function action(id,path,body){
if(pendingActions.has(id))return false;
pendingActions.add(id);$(id).disabled=true;actionError=null;renderAlerts();
const controller=new AbortController();
const timer=setTimeout(()=>controller.abort(),7000);
try{
let response=await fetch(path,{method:'POST',signal:controller.signal,
headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
if(!response.ok){
let message='Request failed (HTTP '+response.status+').';
try{let result=await response.json();if(result.error)message=String(result.error);}catch{}
throw new Error(message);}
return true;
}catch(e){actionError=e.name==='AbortError'?'Request timed out. Check the current state before retrying.':
'Action failed: '+(e.message||String(e));return false;
}finally{clearTimeout(timer);pendingActions.delete(id);
$(id).disabled=id==='micbtn'&&lastMicOn===null;renderAlerts();}}
$('micbtn').onclick=async()=>{
if(lastMicOn===null||pendingActions.has('micbtn'))return;
let desired=!lastMicOn;micVersion++;
if(await action('micbtn','/api/mic',{enabled:desired}))lastMicOn=desired;
$('micbtn').textContent=lastMicOn?'mute mic':'unmute mic';};
$('intbtn').onclick=()=>action('intbtn','/api/interrupt');
$('newbtn').onclick=()=>action('newbtn','/api/reset');
$('tstbtn').onclick=()=>action('tstbtn','/api/test');
$('micsel').onchange=async e=>{
let sel=$('micsel'),value=String(e.target.value);
if(pendingActions.has('micsel')||value===''||!Number.isSafeInteger(+value)||+value<0||
!Array.from(sel.options).some(o=>o.value===value))return;
deviceVersion++;
if(await action('micsel','/api/micdev',{index:+value}))lastDeviceIndex=value;
sel.value=lastDeviceIndex??'';};
$('pausesel').onchange=async e=>{
let sel=$('pausesel'),value=Number(e.target.value);
if(pendingActions.has('pausesel')||![480,800,1120].includes(value))return;
pauseVersion++;
if(await action('pausesel','/api/pause',{milliseconds:value}))lastPause=value;
sel.value=lastPause??'';};
$('agentsel').onchange=()=>{settingsDirty=true;};
$('agentmodel').oninput=()=>{settingsDirty=true;};
$('agentapply').onclick=async()=>{
if(settingsBusy||!settingsState)return;
let mode=$('agentsel').value,model=$('agentmodel').value.trim();
if(!['local','opencode-go','opencode-zen'].includes(mode))return;
settingsBusy=true;settingsVersion++;
let body=mode==='local'?{mode}:{mode,model:model||'kimi-k3'};
if(await action('agentapply','/api/agent',body))settingsDirty=false;
settingsBusy=false;};
$('searchsel').onchange=async e=>{
if(settingsBusy||!settingsState||!['none','ddg','bing','brave'].includes(e.target.value))return;
settingsBusy=true;settingsVersion++;
await action('searchsel','/api/search-provider',{provider:e.target.value});
settingsBusy=false;};
function bubble(t){let d=document.createElement('div');
if(t.startsWith('user:')){d.className='msg u';d.textContent=t.slice(5).trim();}
else if(t.startsWith('assistant:')){d.className='msg a';d.textContent=t.slice(10).trim();}
else{d.className='msg s';d.textContent=t;}return d;}
async function tick(){
try{
let observedSettingsVersion=settingsVersion,observedSettingsBusy=settingsBusy;
let observedMicVersion=micVersion;
let observedMicPending=pendingActions.has('micbtn');
let observedDeviceVersion=deviceVersion,observedDevicePending=pendingActions.has('micsel');
let observedPauseVersion=pauseVersion,observedPausePending=pendingActions.has('pausesel');
let s=await(await fetch('/api/state')).json();
lastState=s;
let pause=$('pausesel');
if(!observedPausePending&&observedPauseVersion===pauseVersion&&!pendingActions.has('pausesel')&&[480,800,1120].includes(s.end_silence_ms)){
lastPause=s.end_silence_ms;pause.value=String(lastPause);}
pause.disabled=lastPause===null||pendingActions.has('pausesel');
if(!observedSettingsBusy&&observedSettingsVersion===settingsVersion&&!settingsBusy){
settingsState=s;
let modeSelect=$('agentsel'),modelInput=$('agentmodel'),searchSelect=$('searchsel');
if(!settingsDirty){
if(document.activeElement!==modeSelect)modeSelect.value=s.agent_mode_id||'local';
if(document.activeElement!==modelInput&&s.agent_model)modelInput.value=s.agent_model;}
if(document.activeElement!==searchSelect)searchSelect.value=s.search_provider||'none';
let busy=s.active_turn||s.pending_turns||s.status==='recording';
modeSelect.disabled=!!busy;modelInput.disabled=!!busy;
$('agentapply').disabled=!!busy;searchSelect.disabled=!!busy||!s.agent_mode;
}
let mode=$('mode');mode.textContent=s.agent_mode?
('Cloud conversation: recognized speech is sent to '+(s.agent_provider||'OpenCode')+' · web search '+(s.search_enabled?(s.search_provider||'enabled'):'not configured')+(s.agent_phase?' · '+s.agent_phase:'')+(s.agent_provider==='OpenCode Go'?' · Go is intended for coding-agent traffic':'')):
'Voice conversation · processed by your configured voice server';
let sources=$('sources');sources.innerHTML='';
if(Array.isArray(s.agent_sources)&&s.agent_sources.length){
let label=document.createElement('strong');label.textContent='Sources';sources.appendChild(label);
s.agent_sources.forEach(source=>{try{let url=new URL(source.url);if(!['http:','https:'].includes(url.protocol))return;
let line=document.createElement('div'),a=document.createElement('a');a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';a.textContent=source.title||url.hostname;line.appendChild(a);sources.appendChild(line);}catch{}});}
let st=$('st');st.textContent=s.status+(s.mic_on?'':' (muted)')+' · '+s.mic;st.className=s.status;
if(!observedMicPending&&observedMicVersion===micVersion&&!pendingActions.has('micbtn'))lastMicOn=s.mic_on;
$('micbtn').textContent=lastMicOn?'mute mic':'unmute mic';
$('micbtn').disabled=lastMicOn===null||pendingActions.has('micbtn');
let ok=s.server_ok;$('srv').className='dot'+(ok?' ok':'');
$('srvlbl').textContent=ok?'voice server connected':'voice server offline — check its connection';
renderAlerts();
$('m_ttft').textContent=s.last_ttft??'—';$('m_aud').textContent=s.last_audio_s??'—';
$('m_turn').textContent=s.last_turn_s??'—';$('m_n').textContent=s.turns;
if(Array.isArray(s.mics)&&!observedDevicePending&&observedDeviceVersion===deviceVersion&&!pendingActions.has('micsel')){
let sel=$('micsel'),roster=s.mics.map(([i,n])=>[String(i),String(n)]);
if(sel.options.length!==roster.length||roster.some(([i,n],j)=>sel.options[j].value!==i||sel.options[j].textContent!==n)){
sel.innerHTML='';roster.forEach(([i,n])=>{let o=document.createElement('option');o.value=i;o.textContent=n;sel.appendChild(o);});}
lastDeviceIndex=String(s.mic_index);sel.value=lastDeviceIndex;}
let log=$('log');
if(s.transcript_revision!==lastRevision){log.innerHTML='';
s.transcript.forEach(t=>log.appendChild(bubble(t)));lastRevision=s.transcript_revision;}
let dr=$('draft');
if(s.assistant_partial){dr.style.display='block';dr.textContent='… '+s.assistant_partial;}
else{dr.style.display='none';dr.textContent='';}
log.scrollTop=log.scrollHeight;
let ph=s.phase||s.status;
setStage('st_heard',s.heard_at,'hearing',ph);
setStage('st_under',s.gen_start,'thinking',ph);
setStage('st_voice',s.first_audio_at,'speaking',ph);
let eta=$('etawrap');
if(s.eta_first_audio_ms!=null){
eta.style.display='block';
let done=s.first_audio_at!=null;
eta.className=done?'hit':'';
let pct=done?100:Math.min(100,Math.round(((s.eta_budget_ms||3000)-s.eta_first_audio_ms)/(s.eta_budget_ms||3000)*100));
$('etafill').style.width=pct+'%';
$('etalbl').textContent=done?'first audio ✓':(s.eta_first_audio_ms>0?('voice usually starts in ~'+(s.eta_first_audio_ms/1000).toFixed(1)+' s'):'still preparing your reply…');}
else eta.style.display='none';
if(s.interrupt_candidate)st.textContent+=' · maybe you?';
pollDelay=(ph==='hearing'||ph==='thinking'||ph==='speaking'||ph==='interrupted')?150:300;
let db=s.level>0?(20*Math.log10(s.level)).toFixed(0):'-inf';
$('lvl').textContent='mic level '+db+' dBFS · '+s.mic;
lv.push(Math.min(1,s.level*8));if(lv.length>140)lv.shift();
let c=$('wave'),x=c.getContext('2d');x.fillStyle='#e7e8dc';x.fillRect(0,0,700,64);
lv.forEach((v,i)=>{let h=Math.max(2,v*60);
x.fillStyle=s.status==='recording'?'#ac4c3d':(s.status==='speaking'?'#4c7560':'#8e9f86');
x.fillRect(i*5,32-h/2,4,h);});
}catch(e){$('srvlbl').textContent='app unreachable — is app.py running?';pollDelay=300;}
setTimeout(tick,pollDelay);}
tick();</script></body></html>"""
