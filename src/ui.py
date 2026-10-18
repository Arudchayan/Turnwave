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
</style></head><body>
<header><div class=dot id=srv></div><div><h1>Turnwave</h1>
<div id=srvlbl>checking server…</div></div></header>
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
</div>
<details id=metrics><summary>metrics</summary>
<div id=cards>
<div class=card><b id=m_ttft>—</b><span>time to first audio</span></div>
<div class=card><b id=m_aud>—</b><span>reply audio</span></div>
<div class=card><b id=m_turn>—</b><span>turn time</span></div>
<div class=card><b id=m_n>0</b><span>replies</span></div>
</div></details>
<h3 style="margin:.4em 0">conversation</h3><div id=log role=log aria-live=polite></div>
<div id=draft class="msg a draft" style=display:none></div>
<footer>100% local · LFM2.5-Audio-1.5B · wear headphones for interruption to work</footer>
<script>
let lv=[],lastRevision=-1,pollDelay=300;
const STLBL={st_heard:'heard',st_under:'understanding',st_voice:'voicing'};
function setStage(id,ts,activePh,ph){let e=$(id);
e.className=ts!=null?'done':(ph===activePh?'active':'');
e.textContent=(ts!=null?'✓ ':'')+STLBL[id];}
const $=id=>document.getElementById(id);
let actionError=null,lastState=null,lastMicOn=null,micVersion=0;
let lastDeviceIndex=null,deviceVersion=0;
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
function bubble(t){let d=document.createElement('div');
if(t.startsWith('user:')){d.className='msg u';d.textContent=t.slice(5).trim();}
else if(t.startsWith('assistant:')){d.className='msg a';d.textContent=t.slice(10).trim();}
else{d.className='msg s';d.textContent=t;}return d;}
async function tick(){
try{
let observedMicVersion=micVersion;
let observedMicPending=pendingActions.has('micbtn');
let observedDeviceVersion=deviceVersion,observedDevicePending=pendingActions.has('micsel');
let s=await(await fetch('/api/state')).json();
lastState=s;
let st=$('st');st.textContent=s.status+(s.mic_on?'':' (muted)')+' · '+s.mic;st.className=s.status;
if(!observedMicPending&&observedMicVersion===micVersion&&!pendingActions.has('micbtn'))lastMicOn=s.mic_on;
$('micbtn').textContent=lastMicOn?'mute mic':'unmute mic';
$('micbtn').disabled=lastMicOn===null||pendingActions.has('micbtn');
let ok=s.server_ok;$('srv').className='dot'+(ok?' ok':'');
$('srvlbl').textContent=ok?'server connected (local :8080)':'SERVER OFFLINE — start it in WSL2';
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
let c=$('wave'),x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,700,64);
lv.forEach((v,i)=>{let h=Math.max(2,v*60);
x.fillStyle=s.status==='recording'?'#f85149':(s.status==='speaking'?'#58a6ff':'#3fb950');
x.fillRect(i*5,32-h/2,4,h);});
}catch(e){$('srvlbl').textContent='app unreachable — is app.py running?';pollDelay=300;}
setTimeout(tick,pollDelay);}
tick();</script></body></html>"""
