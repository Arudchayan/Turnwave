"""Field-test console, served by app.py at /testing."""
TEST_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turnwave / field test</title>
<style>
:root{
  --paper:#f2eee3;--paper-2:#e9e2d3;--ink:#1c1a16;--muted:#6e675b;
  --line:#c8bfae;--orange:#c8511b;--teal:#196b62;--red:#a33a2b;
  --yellow:#9a6b0c;--green:#2f6b3a;--shadow:5px 5px 0 rgba(28,26,22,.10);
  --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;
  --mono:"Cascadia Code","SFMono-Regular",Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;color:var(--ink);background-color:var(--paper);
  background-image:
    linear-gradient(rgba(28,26,22,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(28,26,22,.05) 1px,transparent 1px);
  background-size:32px 32px;
  font-family:var(--serif);font-size:16px;line-height:1.45;
}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer}
.shell{max-width:1280px;margin:0 auto;padding:clamp(16px,3vw,40px)}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  border-bottom:3px solid var(--ink);padding-bottom:14px;margin-bottom:18px;position:relative}
.top:after{content:"";position:absolute;left:0;bottom:-3px;width:22%;height:3px;background:var(--orange)}
.kicker{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
h1{font-size:clamp(2.0rem,5vw,4.2rem);line-height:.92;letter-spacing:-.045em;margin:.12em 0 0;font-weight:600}
.live{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:12px;
  text-transform:uppercase;letter-spacing:.08em;white-space:nowrap}
.dot{width:10px;height:10px;border-radius:50%;background:var(--red);box-shadow:0 0 0 3px rgba(163,58,43,.14)}
.dot.ok{background:var(--teal);box-shadow:0 0 0 3px rgba(25,107,98,.14)}
.clock{border-left:1px solid var(--line);padding-left:10px;color:var(--muted)}
.control{display:grid;grid-template-columns:minmax(160px,220px) 1fr auto auto auto auto;
  gap:10px;align-items:end;padding:14px 0 18px;border-bottom:1px solid var(--line)}
.control > *{min-width:0}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
select,input[type=text]{background:rgba(255,255,255,.45);border:1px solid var(--line);
  padding:8px 10px;border-radius:2px;min-width:0}
select:focus,input:focus,button:focus-visible{outline:3px solid rgba(200,81,27,.35);outline-offset:2px}
.hint{font-family:var(--mono);font-size:10px;line-height:1.45;color:var(--muted);
  background:rgba(255,255,255,.35);border-left:3px solid var(--orange);
  padding:7px 10px;min-height:38px;min-width:0;max-height:96px;overflow:auto;
  overflow-wrap:anywhere}
.btn{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  border:1px solid var(--ink);background:transparent;padding:9px 12px;border-radius:2px;
  transition:transform .15s ease,background .15s ease,color .15s ease}
.btn:hover{transform:translateY(-1px)}
.btn.primary{background:var(--ink);color:var(--paper)}
.btn.accent{background:var(--orange);border-color:var(--orange);color:var(--paper)}
.btn.ghost{border-color:var(--line)}
.btn.danger{border-color:var(--red);color:var(--red)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);
  gap:clamp(20px,3vw,42px);align-items:start;padding-top:22px}
.run-head{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.progress{height:9px;background:var(--paper-2);border:1px solid var(--line);margin:8px 0 20px;overflow:hidden}
.progress b{display:block;height:100%;width:0;background:var(--orange);transition:width .35s cubic-bezier(.16,1,.3,1)}
ol.steps{list-style:none;margin:0;padding:0;counter-reset:step}
.step{display:grid;grid-template-columns:52px minmax(0,1fr);gap:14px;padding:16px 0;
  border-top:1px solid var(--line);position:relative;animation:rise .55s cubic-bezier(.16,1,.3,1) both;
  animation-delay:calc(var(--i) * 45ms)}
.step:first-child{border-top:2px solid var(--ink)}
.step .num{font-family:var(--serif);font-size:31px;line-height:1;color:var(--muted);
  font-variant-numeric:tabular-nums}
.step.pass .num{color:var(--teal)}.step.fail .num{color:var(--red)}.step.skip .num{color:var(--yellow)}
.step h3{margin:0;font-size:1.02rem;letter-spacing:-.01em}
.step .tag{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);border:1px solid var(--line);padding:2px 5px;margin-left:7px;vertical-align:2px}
.step p{margin:5px 0 0;color:var(--muted);font-size:.92rem}
.step .expect{margin-top:5px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.step .expect b{color:var(--ink);font-weight:600}
.outcomes{display:flex;gap:6px;margin-top:9px;flex-wrap:wrap}
.outcomes button{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.08em;
  border:1px solid var(--line);background:transparent;padding:5px 9px;border-radius:2px}
.outcomes button:hover{border-color:var(--ink)}
.outcomes button.pass[aria-pressed=true]{background:var(--teal);border-color:var(--teal);color:var(--paper)}
.outcomes button.fail[aria-pressed=true]{background:var(--red);border-color:var(--red);color:var(--paper)}
.outcomes button.skip[aria-pressed=true]{background:var(--yellow);border-color:var(--yellow);color:var(--paper)}
.note{margin-top:8px;width:100%;background:transparent;border:0;border-bottom:1px dotted var(--line);
  padding:5px 0;font-family:var(--mono);font-size:11px}
.note::placeholder{color:#9a9284}
.panel{border:1px solid var(--line);background:rgba(255,255,255,.38);padding:14px 15px;
  box-shadow:var(--shadow);margin-bottom:16px}
.panel h2{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);margin:0 0 10px;font-weight:500}
.statusline{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.badge{font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  border:1px solid var(--line);padding:5px 9px;border-radius:2px;background:var(--paper)}
.badge.listening{border-color:var(--teal);color:var(--teal)}
.badge.recording{border-color:var(--red);color:var(--red)}
.badge.thinking{border-color:var(--yellow);color:var(--yellow)}
.badge.speaking{border-color:var(--orange);color:var(--orange)}
.meta{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}
.metric{background:var(--paper);padding:9px 10px}
.metric b{display:block;font-family:var(--mono);font-size:19px;letter-spacing:-.04em;font-variant-numeric:tabular-nums}
.metric span{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.meter{height:7px;background:var(--paper-2);border:1px solid var(--line);margin-top:8px;overflow:hidden}
.meter i{display:block;height:100%;width:0;background:var(--teal);transition:width .15s linear}
.log{max-height:230px;overflow:auto;display:flex;flex-direction:column;gap:6px;
  padding-right:4px;scrollbar-width:thin}
.msg{max-width:92%;padding:6px 8px;font-size:.86rem;line-height:1.3;border:1px solid var(--line)}
.msg.u{align-self:flex-end;background:rgba(25,107,98,.08);border-color:rgba(25,107,98,.35)}
.msg.a{align-self:flex-start;background:rgba(255,255,255,.45)}
.msg.s{align-self:center;border:0;color:var(--muted);font-family:var(--mono);font-size:10px}
.events{list-style:none;margin:0;padding:0;font-family:var(--mono);font-size:10px;
  max-height:190px;overflow:auto}
.events li{padding:4px 0;border-top:1px dotted var(--line);color:var(--muted)}
.events li:first-child{border-top:0}
.events b{color:var(--ink);font-weight:500}
.events .floor{color:var(--orange)}.events .eot{color:var(--teal)}
.actionbar{margin-top:24px;padding:14px 0 4px;
  background:transparent;border-top:2px solid var(--ink);
  display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.actionbar .spacer{flex:1}
.actionbar .noteout{font-family:var(--mono);font-size:10px;color:var(--muted)}
.banner{display:none;border:1px solid var(--red);background:rgba(163,58,43,.08);
  color:var(--red);padding:9px 11px;font-family:var(--mono);font-size:11px;margin:0 0 14px}
.banner.show{display:block}
footer{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:18px;
  display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@media(max-width:960px){
  .grid{grid-template-columns:1fr}
  .grid > section:first-child{order:2}
  .grid > aside{order:1}
  .control{grid-template-columns:1fr 1fr}
  .hint{grid-column:1/-1}
  .top{align-items:flex-start;flex-direction:column}
}
@media(max-width:560px){
  .control{grid-template-columns:1fr}
  .step{grid-template-columns:38px minmax(0,1fr)}
  .step .num{font-size:24px}
  h1{font-size:2.2rem}
}
@media(prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important;scroll-behavior:auto!important}
}
</style>
</head>
<body>
<div class="shell">
  <header class="top">
    <div>
      <div class="kicker">Turnwave / field test</div>
      <h1>Conversation test bench</h1>
    </div>
    <div class="live">
      <span class="dot" id="link-dot"></span>
      <span id="link-label">connecting</span>
      <span class="clock" id="clock">00:00</span>
    </div>
  </header>

  <div class="banner" id="banner" role="alert"></div>

  <section class="control" aria-label="Run controls">
    <div class="field">
      <label for="arm">Arm</label>
      <select id="arm">
        <option value="baseline">Current default / no EOT</option>
        <option value="A">A / hard / no EOT</option>
        <option value="B">B / reversible / no EOT</option>
        <option value="C">C / reversible / EOT 1 s</option>
        <option value="D">D / LFM3 realtime</option>
      </select>
    </div>
    <div class="hint" id="config-hint"></div>
    <button class="btn primary" id="start">start run</button>
    <button class="btn ghost" id="reset-run">reset</button>
    <button class="btn danger" id="interrupt">interrupt</button>
    <button class="btn ghost" id="new-conv">new conversation</button>
  </section>

  <main class="grid">
    <section aria-labelledby="run-title">
      <div class="run-head">
        <span id="run-title">run progress</span>
        <span id="progress-text">0 / 8</span>
      </div>
      <div class="progress" aria-hidden="true"><b id="progress-bar"></b></div>
      <ol class="steps" id="step-list"></ol>
    </section>

    <aside aria-label="Live telemetry">
      <section class="panel">
        <h2>Live state</h2>
        <div class="statusline">
          <span class="badge" id="state-badge">starting</span>
          <span class="badge" id="floor-badge">floor -</span>
          <span class="badge" id="eot-badge">eot -</span>
        </div>
        <div class="meta" id="device-meta">mic -</div>
        <div class="meter" aria-hidden="true"><i id="level-bar"></i></div>
      </section>

      <section class="panel">
        <h2>Turn metrics</h2>
        <div class="metrics">
          <div class="metric"><b id="m-ttft">-</b><span>first audio</span></div>
          <div class="metric"><b id="m-turn">-</b><span>turn total</span></div>
          <div class="metric"><b id="m-audio">-</b><span>reply audio</span></div>
          <div class="metric"><b id="m-turns">0</b><span>replies</span></div>
        </div>
        <div class="meta" id="eot-meta">no EOT snapshot</div>
      </section>

      <section class="panel">
        <h2>Conversation</h2>
        <div class="log" id="log" role="log" aria-live="polite"></div>
      </section>

      <section class="panel">
        <h2>Diagnostics</h2>
        <ul class="events" id="events"></ul>
      </section>
    </aside>
  </main>

  <div class="actionbar">
    <button class="btn accent" id="export-json">export json</button>
    <button class="btn" id="export-csv">export csv</button>
    <button class="btn ghost" id="copy-summary">copy summary</button>
    <span class="spacer"></span>
    <span class="noteout" id="save-note">saved locally</span>
  </div>

  <footer>
    <span>8-turn protocol · one variable per arm</span>
    <span><a href="/">back to console</a></span>
  </footer>
</div>

<script>
const STEPS = [
  {id:'normal-1', tag:'floor', title:'Normal turn',
   prompt:'Ask a short factual question.', expect:'Reply starts; no interruption.'},
  {id:'backchannel', tag:'floor', title:'Backchannel',
   prompt:'While the reply plays, say “mm-hmm” once, then stop.', expect:'Playback should not be destroyed.'},
  {id:'stop', tag:'floor', title:'True stop',
   prompt:'While the reply plays, say “stop” clearly.', expect:'Playback stops promptly.'},
  {id:'pause', tag:'eot', title:'Mid-turn pause',
   prompt:'Ask a question with a deliberate 0.6–1.0 s pause before the end.', expect:'Turn is not cut off.'},
  {id:'long', tag:'eot', title:'Long turn',
   prompt:'Speak for 5–8 s with a short pause.', expect:'One turn; no premature submit.'},
  {id:'memory', tag:'memory', title:'Memory after interruption',
   prompt:'State a fact, let the reply finish, interrupt, then ask for it.', expect:'Fact survives the interruption.'},
  {id:'echo', tag:'floor', title:'Echo only',
   prompt:'Let the reply play; do not speak.', expect:'No false interruption.'},
  {id:'normal-2', tag:'floor', title:'Normal turn, again',
   prompt:'Ask a second short question.', expect:'Reply starts normally.'}
];

const ARMS = {
  baseline:{name:'Current default / no EOT', backend:'LFM2.5 HTTP', env:[
    '$env:LOCALVOICE_BACKEND = "lfm2"',
    '$env:LOCALVOICE_TRANSPORT = "http"',
    'Remove-Item Env:LOCALVOICE_FLOOR_MODE, Env:LOCALVOICE_EOT_MODEL -ErrorAction SilentlyContinue']},
  A:{name:'Hard / no EOT', backend:'LFM2.5 HTTP', env:[
    '$env:LOCALVOICE_BACKEND = "lfm2"',
    '$env:LOCALVOICE_TRANSPORT = "http"',
    '$env:LOCALVOICE_FLOOR_MODE = "hard"',
    'Remove-Item Env:LOCALVOICE_EOT_MODEL -ErrorAction SilentlyContinue']},
  B:{name:'Reversible / no EOT', backend:'LFM2.5 HTTP', env:[
    '$env:LOCALVOICE_BACKEND = "lfm2"',
    '$env:LOCALVOICE_TRANSPORT = "http"',
    '$env:LOCALVOICE_FLOOR_MODE = "reversible"',
    'Remove-Item Env:LOCALVOICE_EOT_MODEL -ErrorAction SilentlyContinue']},
  C:{name:'Reversible / EOT 1 s', backend:'LFM2.5 HTTP', env:[
    '$env:LOCALVOICE_BACKEND = "lfm2"',
    '$env:LOCALVOICE_TRANSPORT = "http"',
    '$env:LOCALVOICE_FLOOR_MODE = "reversible"',
    '$env:LOCALVOICE_EOT_MODEL = "smart-turn"',
    '$env:LOCALVOICE_EOT_TIMEOUT_MS = "1000"',
    '$env:LOCALVOICE_EOT_FIRST_THRESHOLD = "0.9"',
    '$env:LOCALVOICE_EOT_THRESHOLD = "0.7"']},
  D:{name:'LFM3 realtime', backend:'LFM3 /v1/realtime', env:[
    '$env:LOCALVOICE_BACKEND = "lfm3"',
    'Remove-Item Env:LOCALVOICE_AGENT, Env:LOCALVOICE_ASR_SERVER -ErrorAction SilentlyContinue',
    '$env:LOCALVOICE_TRANSPORT = "realtime"',
    '$env:LOCALVOICE_REALTIME = "ws://127.0.0.1:8080/v1/realtime"',
    '$env:LOCALVOICE_FLOOR_MODE = "reversible"',
    '$env:LOCALVOICE_EOT_MODEL = "smart-turn"',
    '$env:LOCALVOICE_EOT_TIMEOUT_MS = "1000"',
    '$env:LOCALVOICE_EOT_FIRST_THRESHOLD = "0.9"',
    '$env:LOCALVOICE_EOT_THRESHOLD = "0.7"']}
};

const KEY = 'turnwave-test-v1';
const $ = id => document.getElementById(id);
let arm = localStorage.getItem(KEY + ':arm') || 'baseline';
if(!ARMS[arm]) arm = 'baseline';
let run = load(arm);
let lastTranscript = -1;
let lastState = null;
let busy = false;
let bannerTimer = null;

function load(selected){
  try{
    const raw = localStorage.getItem(KEY + ':' + selected);
    if(!raw) return {arm:selected, startedAt:null, finishedAt:null, steps:{}};
    const parsed = JSON.parse(raw);
    return {arm:selected, startedAt:parsed.startedAt || null,
            finishedAt:parsed.finishedAt || null, steps:parsed.steps || {}};
  }catch(e){ return {arm:selected, startedAt:null, finishedAt:null, steps:{}}; }
}
function save(){
  localStorage.setItem(KEY + ':' + arm, JSON.stringify(run));
  localStorage.setItem(KEY + ':arm', arm);
  $('save-note').textContent = 'saved ' + new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
function showBanner(message){
  const el = $('banner');
  if(!message){ el.className = 'banner'; el.textContent = ''; return; }
  el.className = 'banner show'; el.textContent = message;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { el.className = 'banner'; el.textContent = ''; }, 7000);
}
function fmt(seconds){
  if(seconds === null || seconds === undefined || isNaN(seconds)) return '-';
  return Math.round(Number(seconds) * 1000) + ' ms';
}
function pct(value){
  if(value === null || value === undefined || isNaN(value)) return '-';
  return Math.round(Number(value) * 100) + '%';
}
function clock(ms){
  const total = Math.max(0, Math.round(ms / 1000));
  return String(Math.floor(total / 60)).padStart(2,'0') + ':' + String(total % 60).padStart(2,'0');
}
function armInfo(){ return ARMS[arm] || ARMS.baseline; }
function renderArm(){
  const info = armInfo();
  $('arm').value = arm;
  $('config-hint').innerHTML = '<b>' + info.name + '</b> · ' + info.backend +
    '<br><span>' + info.env.join('<br>') + '</span>';
}
function renderSteps(){
  const list = $('step-list');
  list.innerHTML = '';
  STEPS.forEach((step, index) => {
    const item = document.createElement('li');
    item.className = 'step';
    item.style.setProperty('--i', index);
    item.dataset.id = step.id;
    item.innerHTML =
      '<div class="num">' + String(index + 1).padStart(2,'0') + '</div>' +
      '<div><h3>' + step.title + '<span class="tag">' + step.tag + '</span></h3>' +
      '<p>' + step.prompt + '</p>' +
      '<div class="expect"><b>expect</b> ' + step.expect + '</div>' +
      '<div class="outcomes">' +
      '<button type="button" class="pass" data-outcome="pass">pass</button>' +
      '<button type="button" class="fail" data-outcome="fail">fail</button>' +
      '<button type="button" class="skip" data-outcome="skip">skip</button>' +
      '</div>' +
      '<input class="note" type="text" maxlength="240" placeholder="note (optional)"></div>';
    const note = item.querySelector('.note');
    const current = run.steps[step.id];
    if(current){
      item.classList.add(current.outcome);
      if(current.note) note.value = current.note;
    }
    item.querySelectorAll('[data-outcome]').forEach(button => {
      button.setAttribute('aria-pressed', current && current.outcome === button.dataset.outcome ? 'true' : 'false');
      button.addEventListener('click', () => mark(step.id, button.dataset.outcome));
    });
    note.addEventListener('input', () => {
      if(!run.steps[step.id]) run.steps[step.id] = {outcome:null, note:'', at:null};
      run.steps[step.id].note = note.value.slice(0,240);
      save();
    });
    list.appendChild(item);
  });
}
function mark(id, outcome){
  const current = run.steps[id] || {note:'', at:null};
  current.outcome = current.outcome === outcome ? null : outcome;
  current.at = new Date().toISOString();
  run.steps[id] = current;
  if(!run.startedAt) run.startedAt = Date.now();
  if(Object.values(run.steps).filter(x => x.outcome).length === STEPS.length) run.finishedAt = Date.now();
  else run.finishedAt = null;
  save(); renderSteps(); progress();
}
function progress(){
  const marked = STEPS.filter(step => run.steps[step.id] && run.steps[step.id].outcome).length;
  const passed = STEPS.filter(step => run.steps[step.id] && run.steps[step.id].outcome === 'pass').length;
  $('progress-text').textContent = marked + ' / ' + STEPS.length + '  ·  ' + passed + ' pass';
  $('progress-bar').style.width = (marked / STEPS.length * 100) + '%';
}
async function post(path, body){
  if(busy) return;
  busy = true;
  try{
    const response = await fetch(path, {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body)});
    if(!response.ok){
      let message = 'Request failed (HTTP ' + response.status + ').';
      try{ const data = await response.json(); if(data.error) message = String(data.error); }catch(e){}
      throw new Error(message);
    }
    showBanner('');
  }catch(error){
    showBanner(error.message || String(error));
  }finally{ busy = false; }
}
function renderState(state){
  lastState = state;
  const badge = $('state-badge');
  badge.textContent = state.status || 'unknown';
  badge.className = 'badge ' + (state.status || '');
  $('floor-badge').textContent = 'floor ' + (state.floor_mode || '-') + ' / ' + (state.floor_state || '-');
  $('eot-badge').textContent = state.eot_enabled ? 'eot on' : 'eot off';
  $('device-meta').textContent = (state.mic || 'mic -') +
    (state.mic_on ? ' · live' : ' · muted') +
    (state.server_ok ? ' · server ok' : ' · server offline');
  $('m-ttft').textContent = fmt(state.last_ttft);
  $('m-turn').textContent = state.last_turn_s == null ? '-' : Number(state.last_turn_s).toFixed(2) + ' s';
  $('m-audio').textContent = state.last_audio_s == null ? '-' : Number(state.last_audio_s).toFixed(2) + ' s';
  $('m-turns').textContent = state.turns == null ? '0' : state.turns;
  const eot = state.eot;
  $('eot-meta').textContent = eot
    ? 'eot last ' + pct(eot.last_score) + ' · queries ' + eot.queries + ' · timeout ' + eot.timeout_ms + ' ms'
    : (state.eot_enabled ? 'eot enabled' : 'no EOT snapshot');
  $('level-bar').style.width = Math.min(100, Math.max(0, Number(state.level || 0) * 800)) + '%';
  $('level-bar').style.background = state.status === 'recording' ? 'var(--red)' :
    (state.status === 'speaking' ? 'var(--orange)' : 'var(--teal)');
  const alerts = [state.error, state.warning].filter(Boolean);
  showBanner(alerts.length ? alerts.join(' · ') : '');
  if(state.transcript_revision !== lastTranscript){
    lastTranscript = state.transcript_revision;
    const log = $('log');
    log.innerHTML = '';
    (state.transcript || []).forEach(text => {
      const msg = document.createElement('div');
      if(String(text).startsWith('user:')){ msg.className = 'msg u'; msg.textContent = String(text).slice(5).trim(); }
      else if(String(text).startsWith('assistant:')){ msg.className = 'msg a'; msg.textContent = String(text).slice(10).trim(); }
      else { msg.className = 'msg s'; msg.textContent = String(text); }
      log.appendChild(msg);
    });
    log.scrollTop = log.scrollHeight;
  }
}
function renderEvents(events){
  const list = $('events');
  list.innerHTML = '';
  (events || []).slice(-12).reverse().forEach(event => {
    const item = document.createElement('li');
    const name = String(event.event || 'event');
    const cls = name.startsWith('floor_') ? 'floor' : (name.startsWith('speech_') || name.startsWith('eot_') ? 'eot' : '');
    const bits = ['reason','state','speech_ms','eot_score','turn_id','seconds']
      .filter(key => event[key] !== undefined && event[key] !== null)
      .map(key => key + '=' + event[key]);
    const label = document.createElement('b');
    label.className = cls;
    label.textContent = name;
    item.appendChild(label);
    item.appendChild(document.createTextNode(' ' + bits.join(' · ')));
    list.appendChild(item);
  });
}
async function tick(){
  try{
    const [stateResponse, eventsResponse] = await Promise.all([
      fetch('/api/state'), fetch('/api/diagnostics')]);
    if(!stateResponse.ok) throw new Error('state HTTP ' + stateResponse.status);
    const state = await stateResponse.json();
    renderState(state);
    if(eventsResponse.ok) renderEvents(await eventsResponse.json());
    $('link-dot').className = 'dot ok';
    $('link-label').textContent = 'app connected';
  }catch(error){
    $('link-dot').className = 'dot';
    $('link-label').textContent = 'app unreachable';
  }
  const started = run.startedAt;
  if(started){
    const end = run.finishedAt || Date.now();
    $('clock').textContent = clock(end - started);
  }
  setTimeout(tick, 500);
}
function exportData(){
  return {schema_version:1, arm, arm_name:armInfo().name, backend:armInfo().backend,
    env:armInfo().env, started_at:run.startedAt ? new Date(run.startedAt).toISOString() : null,
    finished_at:run.finishedAt ? new Date(run.finishedAt).toISOString() : null,
    steps:STEPS.map(step => ({id:step.id, title:step.title, tag:step.tag,
      outcome:(run.steps[step.id] && run.steps[step.id].outcome) || null,
      note:(run.steps[step.id] && run.steps[step.id].note) || '',
      marked_at:(run.steps[step.id] && run.steps[step.id].at) || null})),
    telemetry:lastState ? {status:lastState.status, floor_mode:lastState.floor_mode,
      floor_state:lastState.floor_state, eot_enabled:lastState.eot_enabled,
      last_ttft:lastState.last_ttft, last_turn_s:lastState.last_turn_s,
      last_audio_s:lastState.last_audio_s, turns:lastState.turns} : null};
}
function download(name, text, type){
  const blob = new Blob([text], {type});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function csvCell(value){
  value = value == null ? '' : String(value);
  return '"' + value.replaceAll('"','""') + '"';
}
function exportCsv(){
  const rows = [['arm','step','title','tag','outcome','note','marked_at']];
  exportData().steps.forEach(step => rows.push([arm, step.id, step.title, step.tag,
    step.outcome || '', step.note || '', step.marked_at || '']));
  download('turnwave-test-' + arm + '.csv',
    rows.map(row => row.map(csvCell).join(',')).join('\\n') + '\\n', 'text/csv');
}
function summary(){
  const data = exportData();
  const lines = ['Turnwave test run', 'arm: ' + data.arm + ' / ' + data.arm_name,
    'backend: ' + data.backend];
  data.steps.forEach(step => lines.push((step.outcome || '-').padEnd(5) + ' ' + step.id +
    (step.note ? ' — ' + step.note : '')));
  return lines.join('\\n');
}
async function copySummary(){
  const text = summary();
  try{ await navigator.clipboard.writeText(text); showBanner(''); $('save-note').textContent = 'summary copied'; }
  catch(error){
    const area = document.createElement('textarea'); area.value = text; document.body.appendChild(area);
    area.select(); document.execCommand('copy'); area.remove(); $('save-note').textContent = 'summary copied';
  }
}
$('arm').addEventListener('change', event => {
  arm = event.target.value; run = load(arm); save(); renderArm(); renderSteps(); progress();
});
$('start').addEventListener('click', () => {
  run.startedAt = Date.now(); run.finishedAt = null; save(); progress();
  $('clock').textContent = '00:00';
});
$('reset-run').addEventListener('click', () => {
  run = {arm, startedAt:null, finishedAt:null, steps:{}};
  save(); renderSteps(); progress(); $('clock').textContent = '00:00'; lastTranscript = -1;
});
$('interrupt').addEventListener('click', () => post('/api/interrupt'));
$('new-conv').addEventListener('click', () => post('/api/reset'));
$('export-json').addEventListener('click', () => {
  download('turnwave-test-' + arm + '.json', JSON.stringify(exportData(), null, 2), 'application/json');
});
$('export-csv').addEventListener('click', exportCsv);
$('copy-summary').addEventListener('click', copySummary);
renderArm(); renderSteps(); progress(); tick();
</script>
</body>
</html>
"""
