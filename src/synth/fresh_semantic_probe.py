"""Preregister, render, gate and run a small CPU-only ownership-prompt screen.

Six fresh fact values, two real SAPI voices, two system profiles. No physical
audio, no production edit, no repeated deterministic trials counted as evidence.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from xml.sax.saxutils import escape

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from backend_session import backend_session
from client import ServerClient,Turn
from synth.checkpoint_experiment import checkpoint,digest
from synth.semantic_contract import VERSION,canonical,contract_from_setup,evaluate,audio_evidence
from synth.read_aloud_rubrics import normalize
from synth.native_abort_integration import port_free,stop_owned
from synth.native_abort_probe import wsl_path,wsl_home
from synth.snapshot_live_matrix import run as snapshot_completed

PRIVATE=None  # Optional explicit Linux runtime path; otherwise use the WSL home.
REFERENCE=ROOT/'outputs/experiments/semantic-remediation-r1-v1'
ENDPOINT='http://127.0.0.1:8081'
OWNERSHIP=(' Facts that the user tells you about themselves describe the user, not you.'
           ' Retain these facts for this conversation and use them when answering.'
           ' Use you for the user and I for yourself. Answer recall questions from'
           ' the conversation and follow requests for an answer containing only the requested fact.')


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=False)
    previous=json.loads((REFERENCE/'EPI-1-FA-david-r0-turn0-SETUP-request.json').read_text())
    baseline=previous['messages'][0]['content']
    specs=[('NEW-NAME','my name is Nora','What is my name? Just the name.'),
           ('NEW-NUMBER','my lucky number is three','What is my lucky number? Just the number.'),
           ('NEW-MEETING','my meeting is Tuesday at 3 PM','When is my meeting? Just the day and time.'),
           ('NEW-DOG','I walk my dog every afternoon','When do I walk my dog? Just the schedule.'),
           ('NEW-DRINK','I drink cocoa in the morning','What do I drink in the morning? Just the drink.'),
           ('NEW-TOWN','my hometown is Leeds','What is my hometown? Just the town.')]
    facts=[];fixtures=[]
    for fact_id,statement,question in specs:
        setup='For this test, '+statement+'. Please remember this and acknowledge briefly.'
        facts.append(dict(id=fact_id,setup_text=setup,question=question,directed=True,contract=contract_from_setup(setup)))
        for kind,text in [('SETUP',setup),('RECALL',question)]:
            fixtures.append(dict(id=fact_id+'-'+kind,spoken_text=text,ssml='<speak version="1.0" xml:lang="en-US">'+escape(text)+'</speak>'))
    voices=[dict(id='david',name='Microsoft David Desktop'),dict(id='zira',name='Microsoft Zira Desktop')]
    pairs=[(fact['id'],voice['id']) for fact in facts for voice in voices]
    random.Random(20261017).shuffle(pairs)
    runs=[]
    for index,(fact,voice) in enumerate(pairs):
        for arm in (['baseline','ownership'] if index%2==0 else ['ownership','baseline']):
            runs.append(dict(id=fact+'-'+voice+'-'+arm,fact=fact,voice=voice,arm=arm))
    protocol=dict(version='fresh-ownership-screen-1',created_utc=datetime.now(timezone.utc).isoformat(),
        facts=facts,voices=voices,cases=fixtures,runs=runs,seed=20261017,
        profiles=dict(baseline=baseline,ownership=baseline+OWNERSHIP),evaluator=VERSION,
        variable='Only a general system ownership/recall instruction changes; audio and questions identical between profiles.',
        gates=dict(asr_similarity=.9,setup_value_required=True,recall_tail_required=True,max_tokens=512),
        scope='24 fresh reset contexts, six fact values, two real SAPI voices; small exploratory screen, not a benchmark or deployment gate.',
        stop_rule='Stop on fixture/transport failure; preserve partial evidence. No prompt or question edits after inference begins.',
        promotion_rule='No deployment from this screen. Inspect every paired loss and manual answer. Any verified baseline pass lost rejects an unconditional improvement claim; require new validation before a larger run.',
        physical_audio=False,device='isolated CPU, 2 generation/batch threads, zero GPU offload',
        reference_server=json.loads((REFERENCE/'server-process.json').read_text()))
    write_json(folder/'protocol.json',protocol)
    shutil.copy2(__file__,folder/'probe-driver.py')
    for name in ['semantic_contract.py','generate_read_aloud.ps1']:
        shutil.copy2(ROOT/'src/synth'/name,folder/name)
    # Freeze protocol before synthesis, ASR or model inference.
    write_json(folder/'preregistration.json',dict(protocol_sha256=digest(folder/'protocol.json'),model_calls=0,
        note='New to the inspected twelve-fact matrix, not a claim of unseen model-training data. Values selected before new outputs.'))


def check_heard(fixture,heard,protocol):
    match=difflib.SequenceMatcher(a=normalize(fixture['spoken_text']),b=normalize(heard)).ratio()
    if fixture['id'].endswith('-SETUP'):
        fact=next(f for f in protocol['facts'] if f['id']+'-SETUP'==fixture['id'])
        detail=any(re.search(r'(?<!\w)'+re.escape(canonical(a))+r'(?!\w)',canonical(heard)) for a in fact['contract']['aliases'])
    else:
        detail=canonical(heard).split()[-3:]==canonical(fixture['spoken_text']).split()[-3:]
    return dict(similarity=match,critical_detail_pass=detail,passed=bool(match>=.9 and detail))


def gate(folder,model_path):
    from faster_whisper import WhisperModel
    from importlib.metadata import version
    target=folder/'local-asr.jsonl'
    if target.exists():raise ValueError('Preserve existing gate; use a new experiment')
    protocol=json.loads((folder/'protocol.json').read_text())
    if digest(folder/'protocol.json')!=json.loads((folder/'preregistration.json').read_text())['protocol_sha256']:
        raise ValueError('Preregistered protocol changed')
    inventory=[dict(path=p.relative_to(model_path).as_posix(),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(model_path.rglob('*')) if p.is_file()]
    if not inventory:raise ValueError('Explicit local ASR weights missing')
    write_json(folder/'asr-model.json',dict(path=str(model_path.resolve()),files=inventory,
        packages={p:version(p) for p in ['faster-whisper','ctranslate2']},device='cpu',compute_type='int8',threads=2))
    model=WhisperModel(str(model_path.resolve()),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    rows=[]
    with target.open('x',encoding='utf-8') as results:
        for voice in protocol['voices']:
            for fixture in protocol['cases']:
                key=voice['id']+'-'+fixture['id'];path=folder/'fixtures'/(key+'.wav')
                segments,_=model.transcribe(str(path),language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False)
                text=' '.join(s.text.strip() for s in segments)
                row=dict(key=key,fixture=fixture['id'],voice=voice['id'],text=text,sha256=digest(path),**check_heard(fixture,text,protocol))
                results.write(json.dumps(row)+'\n');results.flush();os.fsync(results.fileno());rows.append(row);print(json.dumps(row),flush=True)
    write_json(folder/'local-gate-summary.json',dict(checked=len(rows),passed=sum(r['passed'] for r in rows),complete=True))
    if not all(r['passed'] for r in rows):raise RuntimeError('Fixture ASR gate failed; no model conversation permitted')
    shutil.copytree(folder/'fixtures',folder/'inputs')


def run(folder):
    private=PRIVATE or wsl_home()+'/localvoice-native-abort-v1'
    if (folder/'results.jsonl').exists():raise ValueError('Preserve previous results; no implicit rerun')
    protocol=json.loads((folder/'protocol.json').read_text())
    if digest(folder/'protocol.json')!=json.loads((folder/'preregistration.json').read_text())['protocol_sha256']:
        raise ValueError('Preregistered protocol changed')
    gates=[json.loads(s) for s in (folder/'local-asr.jsonl').read_text().splitlines()]
    expected={v['id']+'-'+f['id'] for v in protocol['voices'] for f in protocol['cases']}
    if len(gates)!=len(expected) or {r['key'] for r in gates}!=expected or not all(r['passed'] and digest(folder/'inputs'/(r['key']+'.wav'))==r['sha256'] for r in gates):
        raise ValueError('Missing, invalid or changed fixture gate')
    facts={f['id']:f for f in protocol['facts']};fixtures={f['id']:f for f in protocol['cases']}
    with backend_session(ENDPOINT,test=True):
        if not port_free():raise RuntimeError('8081 occupied; leave owner alone')
        (folder/'reply-audio').mkdir()
        command=['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/serve_abort_variant_linux.py'),private,'lifecycle',wsl_path(folder)]
        rows=[]
        with (folder/'server.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                client=ServerClient(ENDPOINT,timeout=120);deadline=time.monotonic()+120
                while not client.health():
                    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Private CPU server startup failed')
                    time.sleep(.5)
                actual=json.loads((folder/'server-process.json').read_text());prior=protocol['reference_server']
                if any(actual[k]!=prior[k] for k in ['binary_sha256','models','command','cuda_visible_devices']):
                    raise ValueError('CPU server or model differs from preregistration')
                with (folder/'server-asr.jsonl').open('x',encoding='utf-8') as checks:
                    for gate_row in gates:
                        heard=client.asr_text((folder/'inputs'/(gate_row['key']+'.wav')).read_bytes())
                        row=dict(key=gate_row['key'],text=heard,**check_heard(fixtures[gate_row['fixture']],heard,protocol))
                        checks.write(json.dumps(row)+'\n');checks.flush();os.fsync(checks.fileno())
                        if not row['passed']:raise RuntimeError('Server ASR fixture gate failed: '+row['key'])
                with (folder/'results.jsonl').open('x',encoding='utf-8') as results:
                    for case in protocol['runs']:
                        fact=facts[case['fact']]
                        for pos,kind in enumerate(['SETUP','RECALL']):
                            key=case['voice']+'-'+case['fact']+'-'+kind;turn=Turn()
                            payload=client._payload(protocol['profiles'][case['arm']],client._audio_input((folder/'inputs'/(key+'.wav')).read_bytes()),pos==0,512)
                            stem=case['id']+f'-turn{pos}-'+kind
                            write_json(folder/(stem+'-request.json'),payload)
                            path=folder/'reply-audio'/(stem+'.wav')
                            with wave.open(str(path),'wb') as writer:
                                writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000)
                                def receive(chunk):writer.writeframes((np.clip(chunk,-1,1)*32767).astype('<i2').tobytes())
                                client._stream(payload,turn,receive)
                            transport=bool(not turn.error and not turn.truncated and turn.audio_samples and not turn.cancel.is_set() and turn.finish_reason=='stop')
                            audio=audio_evidence(path.read_bytes(),turn.audio_samples/24000)
                            row=dict(run=case['id'],fact=case['fact'],voice=case['voice'],arm=case['arm'],case=kind,recall=bool(pos),
                                text=turn.text,error=turn.error,finish_reason=turn.finish_reason,truncated=turn.truncated,
                                audio_s=turn.audio_samples/24000,first_audio_s=turn.ttft,transport_error=not transport,audio=audio)
                            if pos:row['evaluation']=evaluate(turn.text,fact['contract'],True,transport and audio['valid'])
                            results.write(json.dumps(row)+'\n');results.flush();os.fsync(results.fileno());rows.append(row);print(json.dumps(row),flush=True)
                            if not transport or not audio['valid']:raise RuntimeError('Transport/audio failed; preserve broken context without proceeding')
                        snapshot=folder.parent/(folder.name+'-durable-'+case['id'])
                        snapshot_completed(snapshot,folder)
                        for name in ['preregistration.json','local-asr.jsonl','asr-model.json','server-asr.jsonl']:
                            shutil.copy2(folder/name,snapshot/'partial-experiment'/name)
                        print('Verified pair checkpoint:',checkpoint(snapshot),flush=True)
            finally:
                stop_owned(process,folder,private+'/lifecycle/llama-liquid-audio-server')
                write_json(folder/'server-exit.json',dict(exit_code=process.poll()))
        summary=dict(complete=True,runs=len(protocol['runs']),profiles={arm:dict(Counter(r['evaluation']['outcome'] for r in rows if r['recall'] and r['arm']==arm)) for arm in protocol['profiles']},
            deployed=False,scope=protocol['scope'],manual_review_required=True)
        write_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','gate','run']);p.add_argument('folder',type=Path);p.add_argument('--asr-model',type=Path);a=p.parse_args()
    if (a.folder/(a.stage+'-failure.txt')).exists():raise SystemExit('Preserve the failed stage; use a new experiment')
    if a.stage=='prepare' and a.folder.exists():raise SystemExit('Use a new experiment folder')
    if a.stage in ['gate','run'] and (a.folder/('local-asr.jsonl' if a.stage=='gate' else 'server.log')).exists():raise SystemExit('Stage already started; preserve it and use a new experiment')
    try:
        if a.stage=='prepare':prepare(a.folder.resolve())
        elif a.stage=='gate':
            if not a.asr_model:raise ValueError('Explicit local ASR model directory is required')
            gate(a.folder.resolve(),a.asr_model)
        else:run(a.folder.resolve())
    except Exception:
        if a.folder.exists():(a.folder/(a.stage+'-failure.txt')).write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
