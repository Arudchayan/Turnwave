"""Local browser interface, served by app.py."""
PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Turnwave</title>
<style>
:root{
--bg:#f4f0e8;--bg-elev:#fbf8f1;--card:#fffdf7;--inset:#ebe7dc;--line:#d5cebe;--line-strong:#bfb6a2;
--txt:#1f2521;--dim:#5c675a;--muted:#828c7e;
--green:#2d6348;--green-soft:#e1ece5;--red:#ad4335;--red-soft:#faeae5;--amber:#94621d;--amber-soft:#f8eedb;--blue:#2f5963;
--shadow:0 1px 2px rgba(31,37,33,.04),0 10px 28px -12px rgba(31,37,33,.08)
}
*{box-sizing:border-box}
body{
margin:0;color:var(--txt);background:radial-gradient(1100px 520px at 50% -140px,#fbf9f3 0%,var(--bg) 72%);
font-family:Georgia,"Iowan Old Style","Palatino Linotype",serif;line-height:1.5;-webkit-font-smoothing:antialiased
}
main{max-width:1040px;margin:0 auto;padding:clamp(18px,3.6vw,44px) clamp(16px,4vw,40px) 36px}
#srvlbl,#lvl,button,select,input,#stages,#banner,#cards,footer,summary,#mode,#sources,#notes,#diag,.chip,.sect-hdr,.ctrl-lbl{
font-family:"Segoe UI",-apple-system,BlinkMacSystemFont,system-ui,sans-serif
}
header{
display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;
padding-bottom:18px;margin-bottom:22px;border-bottom:1px solid var(--line)
}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{
width:34px;height:34px;border-radius:9px;background:var(--green);color:#fffdf7;
display:grid;place-items:center;font:600 .95rem "Segoe UI",sans-serif;letter-spacing:-.03em;
box-shadow:inset 0 0 0 1px rgba(255,255,255,.18)
}
h1{font-size:1.55rem;letter-spacing:-.04em;font-weight:normal;margin:0;line-height:1.1}
.srv-pill{
display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;
background:var(--card);border:1px solid var(--line);box-shadow:0 1px 2px rgba(0,0,0,.02)
}
.dot{width:8px;height:8px;border-radius:50%;background:var(--red);flex:none;transition:background .2s,box-shadow .2s}
.dot.ok{background:var(--green);box-shadow:0 0 0 3px var(--green-soft)}
#srvlbl{font-size:.76rem;color:var(--dim);letter-spacing:.01em}
#mode{
font-size:.8rem;color:var(--dim);background:var(--bg-elev);border:1px solid var(--line);
border-radius:8px;padding:8px 13px;margin:0 0 18px;display:flex;align-items:center;gap:8px;line-height:1.4
}
#mode::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--blue);flex:none}
#banner{
display:none;white-space:pre-line;background:var(--red-soft);border:1px solid #d79588;color:#6e2419;
border-radius:10px;padding:.75em 1.05em;margin:0 0 18px;font-size:.86rem;line-height:1.45;box-shadow:var(--shadow)
}
.workspace{display:grid;grid-template-columns:minmax(310px,390px) minmax(0,1fr);gap:22px;align-items:start}
.panel{
background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:var(--shadow)
}
.status-row{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:6px}
#st{
display:inline-block;font-size:clamp(1.45rem,2.6vw,2rem);font-weight:normal;letter-spacing:-.035em;
color:var(--txt);line-height:1.2
}
#st.listening{color:var(--green)}
#st.recording{color:var(--red);animation:pulse 1.1s ease-in-out infinite}
#st.thinking{color:var(--amber)}
#st.speaking{color:var(--blue)}
@keyframes pulse{50%{opacity:.55}}
#stages{
display:grid;grid-template-columns:repeat(3,1fr);gap:8px;list-style:none;margin:12px 0 16px;padding:0;
font-size:.7rem;text-transform:uppercase;letter-spacing:.08em
}
#stages li{
border:0;border-top:2.5px solid var(--line);padding:.45em 0 0;color:var(--muted);transition:border-color .2s,color .2s
}
#stages li.done{border-color:var(--green);color:var(--green);font-weight:600}
#stages li.active{border-color:var(--amber);color:var(--amber);font-weight:600}
#etawrap{
display:none;background:var(--inset);border:1px solid var(--line);border-radius:999px;
height:22px;margin:0 0 14px;position:relative;overflow:hidden
}
#etafill{height:100%;width:0;background:linear-gradient(90deg,#c58b37,var(--amber));transition:width .15s linear}
#etawrap.hit #etafill{background:linear-gradient(90deg,#3d7b5b,var(--green))}
#etalbl{position:absolute;inset:0;text-align:center;font-size:.73rem;line-height:20px;color:var(--txt);font-weight:500}
#wavewrap{
background:var(--inset);border:1px solid var(--line);border-radius:10px;padding:12px 12px 9px;margin:0 0 16px
}
canvas{width:100%;height:64px;display:block;border-radius:4px}
#lvl{font-size:.73rem;color:var(--dim);margin-top:6px;display:flex;justify-content:space-between}
#controls{display:flex;flex-direction:column;gap:10px;margin:0}
.btn-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button,select{
border-radius:8px;border:1px solid var(--line-strong);background:var(--bg-elev);color:var(--txt);
padding:.68em .95em;font-size:.84rem;font-weight:500;cursor:pointer;transition:background .15s,border-color .15s,transform .06s
}
button:hover,select:hover{border-color:var(--green);background:#e8ece1}
button:active{transform:translateY(1px)}
button:disabled,select:disabled{opacity:.52;cursor:wait;transform:none}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:2.5px solid #b47439;outline-offset:2px}
button.big{background:var(--green);border-color:var(--green);color:#fffdf7;font-weight:600}
button.big:hover{background:#234f39;border-color:#234f39}
.field-row{display:flex;flex-direction:column;gap:5px;padding-top:4px;border-top:1px solid var(--line)}
.ctrl-lbl,#pause-control{
display:flex;align-items:center;justify-content:space-between;gap:10px;
font-size:.78rem;color:var(--dim)
}
#controls select{width:100%;max-width:100%;background:var(--card)}
#pause-control select{width:auto;min-width:165px}
details#metrics,details#voicetune{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
details#metrics summary,details#voicetune summary{
cursor:pointer;color:var(--dim);font-size:.78rem;font-weight:600;text-transform:uppercase;
letter-spacing:.06em;user-select:none
}
.vgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:8px}
.vfield{display:flex;flex-direction:column;gap:3px;font-size:.73rem;color:var(--dim)}
.vfield input{
font:inherit;font-size:.82rem;border-radius:7px;border:1px solid var(--line-strong);
background:var(--bg-elev);color:var(--txt);padding:.42em .65em;font-variant-numeric:tabular-nums
}
#vstatus{font-size:.72rem;color:var(--green);margin-top:6px;font-variant-numeric:tabular-nums}
#cards{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
.card{
background:var(--bg-elev);border:1px solid var(--line);border-radius:8px;text-align:left;padding:9px 11px
}
.card b{display:block;font-size:1.05rem;font-weight:600;color:var(--txt);font-variant-numeric:tabular-nums}
.card span{font-size:.71rem;color:var(--dim)}
#diag{
font:.74rem "Segoe UI",sans-serif;color:var(--dim);background:var(--bg-elev);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;margin-top:8px;line-height:1.45;font-variant-numeric:tabular-nums
}
.conv-panel{display:flex;flex-direction:column;min-height:460px}
.sect-hdr{
display:flex;align-items:center;justify-content:space-between;padding-bottom:12px;
border-bottom:1px solid var(--line);margin:0 0 14px
}
.sect-hdr h2{
margin:0;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)
}
.chip{font-size:.73rem;color:var(--muted);background:var(--bg-elev);border:1px solid var(--line);padding:2px 9px;border-radius:999px}
#log{
display:flex;flex-direction:column;gap:12px;flex:1;min-height:310px;max-height:520px;overflow-y:auto;
padding:4px 4px 8px 0
}
#log:empty::before{
content:"No turns yet — unmute the microphone or run a speaker check to begin.";
color:var(--muted);font:italic .88rem Georgia,serif;margin:auto;text-align:center;padding:36px 16px
}
.msg{
max-width:86%;padding:.7em 1em;border-radius:12px;font-size:.96rem;line-height:1.5;
box-shadow:0 1px 2px rgba(0,0,0,.03);word-wrap:break-word
}
.msg.u{
align-self:flex-end;background:var(--green-soft);color:#183526;border:1px solid #bfd4c7;
border-bottom-right-radius:3px
}
.msg.a{
align-self:flex-start;background:var(--bg-elev);border:1px solid var(--line);color:var(--txt);
border-bottom-left-radius:3px
}
.msg.s{align-self:center;color:var(--dim);font-style:italic;font-size:.83rem;background:none;border:0;box-shadow:none;padding:.2em .6em}
.msg.a.draft{border-style:dashed;opacity:.85;margin-top:10px}
#sources{
font:.8rem "Segoe UI",sans-serif;line-height:1.55;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)
}
#sources:empty{display:none}
#sources strong{display:block;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-bottom:6px}
#sources div{margin:4px 0}
#sources a{
display:inline-block;color:var(--green);text-decoration:none;background:var(--bg-elev);
border:1px solid var(--line);border-radius:6px;padding:3px 9px;overflow-wrap:anywhere;transition:border-color .15s,background .15s
}
#sources a:hover{border-color:var(--green);background:var(--green-soft)}
#notes{
font:.83rem "Segoe UI",sans-serif;line-height:1.5;margin-top:12px;padding:10px 12px;
background:var(--bg-elev);border:1px solid var(--line);border-radius:10px
}
#notes:empty{display:none}
#notes strong{display:block;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--blue);margin-bottom:6px}
.note-card{border-top:1px solid var(--line);padding-top:8px;margin-top:8px}
.note-card:first-of-type{border-top:0;padding-top:0;margin-top:0}
.note-hdr{font-weight:600;color:var(--txt);font-size:.8rem;margin-bottom:4px}
.note-body{white-space:pre-wrap;font-family:Consolas,"SFMono-Regular",monospace;font-size:.77rem;color:var(--dim);max-height:220px;overflow-y:auto;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}
#promptform{
display:flex;gap:8px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)
}
#promptin{
flex:1;border-radius:8px;border:1px solid var(--line-strong);background:var(--bg-elev);color:var(--txt);
padding:.62em .9em;font-size:.86rem
}
#promptin:focus-visible{outline:2.5px solid #b47439;outline-offset:2px}
footer{
display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
border-top:1px solid var(--line);padding-top:16px;margin-top:26px;font-size:.76rem;color:var(--dim)
}
@media(max-width:820px){
.workspace{grid-template-columns:1fr}
.conv-panel{min-height:340px}
}
@media(max-width:520px){
.btn-grid,#cards{grid-template-columns:1fr}
#pause-control{flex-direction:column;align-items:stretch}
#pause-control select{width:100%}
.msg{max-width:95%}
}
@media(prefers-reduced-motion:reduce){#st.recording{animation:none}#etafill{transition:none}}
</style></head><body><main>
<header>
<div class=brand><div class=brand-mark aria-hidden=true>Tw</div><div><h1>Turnwave</h1></div></div>
<div class=srv-pill><div class=dot id=srv></div><div id=srvlbl>checking server…</div></div>
</header>
<div id=mode>Checking processing mode…</div>
<div id=banner role=alert></div>
<div class=workspace>
<section class=panel aria-label="Voice controls and status">
<div class=status-row><span id=st>starting…</span></div>
<ol id=stages><li id=st_heard>heard</li><li id=st_under>understanding</li><li id=st_voice>voicing</li></ol>
<div id=etawrap><div id=etafill></div><span id=etalbl></span></div>
<div id=wavewrap><canvas id=wave width=700 height=64></canvas>
<div id=lvl>mic level —</div></div>
<div id=controls>
<div class=btn-grid>
<button id=micbtn disabled>checking mic…</button>
<button id=intbtn class=big>interrupt</button>
<button id=newbtn>new conversation</button>
<button id=tstbtn>test speaker</button>
</div>
<div class=field-row>
<label class=ctrl-lbl for=agentsel>Processing &amp; reasoning model</label>
<select id=agentsel aria-label="Processing mode and model">
<option value="local">Local Direct · LFM Speech-to-Speech</option>
<option value="opencode-go:beam-free">Cloud Agent · OpenCode Go (beam-free)</option>
<option value="opencode-go:iceman-contributor-free">Cloud Agent · OpenCode Go (iceman-contributor-free)</option>
<option value="opencode-go:kimi-k3">Cloud Agent · OpenCode Go (kimi-k3)</option>
<option value="custom:local-model">Second Model · Custom Endpoint (127.0.0.1:8081)</option>
</select>
</div>
<div class=field-row>
<label class=ctrl-lbl for=personasel>Voice persona &amp; intermediary style</label>
<select id=personasel aria-label="Voice persona">
<option value="jarvis">Jarvis · Warm Executive Intermediary</option>
<option value="direct">Direct · Zero-Fluff Operator</option>
<option value="natural">Natural · Empathetic Companion</option>
<option value="concise">Concise · Direct Tactical Briefing</option>
<option value="storyteller">Storyteller · Expressive Narrative</option>
</select>
</div>
<div class=field-row>
<label class=ctrl-lbl for=micsel>Input microphone</label>
<select id=micsel aria-label="Input device"></select>
</div>
<div class=field-row>
<label id=pause-control for=pausesel>Pause before reply
<select id=pausesel><option value=480>Quick · 0.48 s</option><option value=800>Balanced · 0.8 s</option><option value=1120>Patient · 1.12 s</option></select></label>
</div>
</div>
<details id=voicetune open><summary>voice &amp; sampling tuning</summary>
<div class=field-row style="border-top:0;margin-top:8px">
<label class=ctrl-lbl for=vpreset>Acoustic preset</label>
<select id=vpreset aria-label="Acoustic preset">
<option value="natural">Natural RVQ · temp 0.70 / k=40 / p=0.90</option>
<option value="crisp">Crisp Studio · temp 0.45 / k=25 / p=0.85</option>
<option value="expressive">Warm Expressive · temp 0.85 / k=60 / p=0.94</option>
<option value="deterministic">Deterministic · temp 0.00 / k=1 / p=1.00</option>
<option value="custom">Custom Values</option>
</select>
</div>
<div class=vgrid>
<label class=vfield for=v_temp>Temperature (0–2)<input id=v_temp type=number step=0.05 min=0 max=2 value=0.7></label>
<label class=vfield for=v_topk>Top-K (1–200)<input id=v_topk type=number step=1 min=1 max=200 value=40></label>
<label class=vfield for=v_topp>Top-P (0.05–1)<input id=v_topp type=number step=0.01 min=0.05 max=1 value=0.9></label>
<label class=vfield for=v_minp>Min-P (0–1)<input id=v_minp type=number step=0.01 min=0 max=1 value=0.05></label>
<label class=vfield for=v_rep>Repeat penalty<input id=v_rep type=number step=0.01 min=0.8 max=2 value=1.05></label>
<label class=vfield for=v_voice>Voice ID (optional)<input id=v_voice type=text maxlength=64 placeholder="default"></label>
</div>
<div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;gap:8px">
<button id=vapplybtn type=button>apply tuning</button>
<span id=vstatus>temp 0.7 · k=40 · p=0.9 · min_p=0.05 · rep=1.05</span>
</div>
</details>
<details id=metrics><summary>diagnostics &amp; metrics</summary>
<div id=cards>
<div class=card><b id=m_ttft>—</b><span>time to first audio</span></div>
<div class=card><b id=m_aud>—</b><span>reply audio</span></div>
<div class=card><b id=m_turn>—</b><span>turn time</span></div>
<div class=card><b id=m_n>0</b><span>replies</span></div>
</div><div id=diag></div></details>
</section>
<section class="panel conv-panel" aria-label="Conversation">
<div class=sect-hdr><h2>conversation</h2><span class=chip>live transcript</span></div>
<div id=log role=log aria-live=polite></div>
<div id=draft class="msg a draft" style=display:none></div>
<div id=notes></div>
<div id=sources></div>
<form id=promptform autocomplete=off>
<input id=promptin type=text maxlength=2000 placeholder="Speak into your mic, or type a prompt to Turnwave…" aria-label="Type a prompt">
<button id=sendbtn type=submit class=big>send</button>
</form>
</section>
</div>
<footer><span>Voice uses your configured server · headphones improve interruption handling</span><span>Local audio capture &amp; synthesis</span></footer>
</main><script>
let lv=[],lastRevision=-1,pollDelay=300;
const STLBL={st_heard:'heard',st_under:'understanding',st_voice:'voicing'};
function setStage(id,ts,activePh,ph){let e=$(id);
e.className=ts!=null?'done':(ph===activePh?'active':'');
e.textContent=(ts!=null?'✓ ':'')+STLBL[id];}
const $=id=>document.getElementById(id);
let actionError=null,lastState=null,lastMicOn=null,micVersion=0;
let lastDeviceIndex=null,deviceVersion=0;
let lastPause=null,pauseVersion=0;
let lastAgentVal=null,agentVersion=0;
let lastPersona=null,personaVersion=0;
let voiceVersion=0,voiceDirty=false;
const VPRESETS={
natural:{temperature:0.7,top_k:40,top_p:0.9,min_p:0.05,repeat_penalty:1.05},
crisp:{temperature:0.45,top_k:25,top_p:0.85,min_p:0.08,repeat_penalty:1.08},
expressive:{temperature:0.85,top_k:60,top_p:0.94,min_p:0.03,repeat_penalty:1.05},
deterministic:{temperature:0.0,top_k:1,top_p:1.0,min_p:0.0,repeat_penalty:1.0}
};
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
if($('agentsel'))$('agentsel').onchange=async e=>{
let sel=$('agentsel'),val=String(e.target.value);
if(pendingActions.has('agentsel')||!val)return;
agentVersion++;
let body=val==='local'?{mode:'local'}:{mode:val.split(':')[0],model:val.split(':').slice(1).join(':')};
if(await action('agentsel','/api/agent',body))lastAgentVal=val;
sel.value=lastAgentVal??'local';};
if($('personasel'))$('personasel').onchange=async e=>{
let sel=$('personasel'),val=String(e.target.value);
if(pendingActions.has('personasel')||!val)return;
personaVersion++;
if(await action('personasel','/api/persona',{persona:val}))lastPersona=val;
sel.value=lastPersona??'jarvis';};
function readVoiceInputs(){
return {
temperature:Number($('v_temp')?.value??0.7),
top_k:Math.round(Number($('v_topk')?.value??40)),
top_p:Number($('v_topp')?.value??0.9),
min_p:Number($('v_minp')?.value??0.05),
repeat_penalty:Number($('v_rep')?.value??1.05),
voice:String($('v_voice')?.value??'').trim()
};}
function writeVoiceInputs(vp){
if(!vp)return;
if($('v_temp'))$('v_temp').value=vp.temperature??0.7;
if($('v_topk'))$('v_topk').value=vp.top_k??40;
if($('v_topp'))$('v_topp').value=vp.top_p??0.9;
if($('v_minp'))$('v_minp').value=vp.min_p??0.05;
if($('v_rep'))$('v_rep').value=vp.repeat_penalty??1.05;
if($('v_voice'))$('v_voice').value=vp.voice??'';
if($('vstatus'))$('vstatus').textContent='temp '+(vp.temperature??0.7)+' · k='+(vp.top_k??40)+' · p='+(vp.top_p??0.9)+' · min_p='+(vp.min_p??0.05)+' · rep='+(vp.repeat_penalty??1.05)+(vp.voice?(' · voice='+vp.voice):'');}
['v_temp','v_topk','v_topp','v_minp','v_rep','v_voice'].forEach(id=>{
if($(id))$(id).oninput=()=>{voiceDirty=true;if($('vpreset'))$('vpreset').value='custom';};});
if($('vpreset'))$('vpreset').onchange=async e=>{
let p=VPRESETS[e.target.value];
if(!p||pendingActions.has('vapplybtn'))return;
writeVoiceInputs({...p,voice:$('v_voice')?.value||''});
voiceVersion++;voiceDirty=false;
await action('vapplybtn','/api/voice',readVoiceInputs());};
if($('vapplybtn'))$('vapplybtn').onclick=async()=>{
if(pendingActions.has('vapplybtn'))return;
let body=readVoiceInputs();
voiceVersion++;
if(await action('vapplybtn','/api/voice',body)){voiceDirty=false;writeVoiceInputs(body);}};
if($('promptform'))$('promptform').onsubmit=async e=>{
e.preventDefault();
let inp=$('promptin'),txt=(inp.value||'').trim();
if(!txt||pendingActions.has('sendbtn'))return;
if(await action('sendbtn','/api/prompt',{text:txt}))inp.value='';};
function bubble(t){let d=document.createElement('div');
if(t.startsWith('user:')){d.className='msg u';d.textContent=t.slice(5).trim();}
else if(t.startsWith('assistant:')){d.className='msg a';d.textContent=t.slice(10).trim();}
else{d.className='msg s';d.textContent=t;}return d;}
async function tick(){
try{
let observedMicVersion=micVersion;
let observedMicPending=pendingActions.has('micbtn');
let observedDeviceVersion=deviceVersion,observedDevicePending=pendingActions.has('micsel');
let observedPauseVersion=pauseVersion,observedPausePending=pendingActions.has('pausesel');
let observedAgentVersion=agentVersion,observedAgentPending=pendingActions.has('agentsel');
let observedPersonaVersion=personaVersion,observedPersonaPending=pendingActions.has('personasel');
let observedVoiceVersion=voiceVersion,observedVoicePending=pendingActions.has('vapplybtn');
let s=await(await fetch('/api/state')).json();
lastState=s;
if(s.voice_params&&!voiceDirty&&!observedVoicePending&&observedVoiceVersion===voiceVersion&&!pendingActions.has('vapplybtn')){
writeVoiceInputs(s.voice_params);}
let psel=$('personasel');
if(psel){
if(!observedPersonaPending&&observedPersonaVersion===personaVersion&&!pendingActions.has('personasel')&&s.agent_persona){
lastPersona=s.agent_persona;psel.value=lastPersona;}
psel.disabled=!s.agent_mode||pendingActions.has('personasel');}
let agsel=$('agentsel');
if(agsel&&!observedAgentPending&&observedAgentVersion===agentVersion&&!pendingActions.has('agentsel')){
let cur=s.agent_mode?((s.agent_provider==='Custom Endpoint'?'custom':s.agent_provider==='OpenCode Zen'?'opencode-zen':'opencode-go')+':'+(s.agent_model||'beam-free')):'local';
if(!Array.from(agsel.options).some(o=>o.value===cur)&&s.agent_mode){
let o=document.createElement('option');o.value=cur;o.textContent='Cloud Agent · '+(s.agent_provider||'OpenCode')+' ('+(s.agent_model||'custom')+')';agsel.appendChild(o);}
lastAgentVal=cur;agsel.value=cur;}
let pause=$('pausesel');
if(!observedPausePending&&observedPauseVersion===pauseVersion&&!pendingActions.has('pausesel')&&[480,800,1120].includes(s.end_silence_ms)){
lastPause=s.end_silence_ms;pause.value=String(lastPause);}
pause.disabled=lastPause===null||pendingActions.has('pausesel');
let toolsInfo=Array.isArray(s.agent_tools)&&s.agent_tools.length?(' · tools ('+s.agent_tools.length+')'):'';
let mode=$('mode');mode.textContent=s.agent_mode?
('Cloud conversation: recognized speech is sent to '+(s.agent_provider||'OpenCode')+(s.agent_model?' ('+s.agent_model+')':'')+' · web search '+(s.search_enabled?('enabled'+(s.search_provider?' ('+s.search_provider+')':'')):'not configured')+toolsInfo+(s.agent_phase?' · '+s.agent_phase:'')+(s.agent_provider==='OpenCode Go'?' · Go is intended for coding-agent traffic':'')):
'Voice conversation · processed by your configured voice server';
let notes=$('notes');
if(notes){notes.innerHTML='';
if(Array.isArray(s.agent_notes)&&s.agent_notes.length){
let label=document.createElement('strong');label.textContent='Expert Deep-Dive Brief';notes.appendChild(label);
s.agent_notes.forEach(n=>{if(!n||typeof n!=='object')return;let card=document.createElement('div');card.className='note-card';
let hdr=document.createElement('div');hdr.className='note-hdr';hdr.textContent=(n.model||'Expert LLM')+' · '+(n.task||'Analysis');
let body=document.createElement('div');body.className='note-body';body.textContent=n.analysis||'';
card.appendChild(hdr);card.appendChild(body);notes.appendChild(card);});}}
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
let ad=s.audio_diagnostics||{},pb=ad.playback||{},diag=$('diag');
if(diag){
let line='gain '+(ad.input_gain??'—')+' · drops '+(ad.input_drops??0)+' · echo '+(ad.echo_score??0).toFixed?.(2)+' · residual rejections '+(ad.residual_rejections??0)+' · underflows '+(pb.underflows??0)+' · max write gap '+(pb.max_gap_ms??0)+' ms';
if(s.last_trace){
let lt=s.last_trace,ck=lt.checks||{},toolsUsed=(lt.tools_used||[]).length?lt.tools_used.join(','):'none';
line+=' · trace #'+lt.turn_id+' ('+lt.status+', tools: '+toolsUsed+', checks: '+(ck.all_passed?'✓ pass':'⚠ check')+(s.trace_file?', saved '+s.trace_file:'')+')';}
diag.textContent=line;}
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
let c=$('wave'),x=c.getContext('2d');x.fillStyle='#ebe7dc';x.fillRect(0,0,700,64);
lv.forEach((v,i)=>{let h=Math.max(2,v*60);
x.fillStyle=s.status==='recording'?'#ad4335':(s.status==='speaking'?'#2f5963':'#7b8b75');
x.fillRect(i*5,32-h/2,4,h);});
}catch(e){$('srvlbl').textContent='app unreachable — is app.py running?';pollDelay=300;}
setTimeout(tick,pollDelay);}
tick();</script></body></html>"""
