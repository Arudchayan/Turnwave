"""Same-model text-input control for the frozen fresh voice screen.

Only input representation changes. This is a post-hoc mechanism diagnostic,
not new held-out evidence or an application/provider replacement.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from backend_session import backend_session
from client import ServerClient,Turn
from synth.checkpoint_experiment import checkpoint,digest
from synth.semantic_contract import VERSION,evaluate,audio_evidence
from synth.native_abort_integration import port_free,stop_owned
from synth.native_abort_probe import wsl_path
from synth.snapshot_live_matrix import run as snapshot_completed
from synth.fresh_semantic_probe import PRIVATE,ENDPOINT,write_json


def prepare(origin,folder,fact_ids=None,replay_origin=None):
    protocol=json.loads((origin/'protocol.json').read_text())
    if digest(origin/'protocol.json')!=json.loads((origin/'preregistration.json').read_text())['protocol_sha256']:
        raise ValueError('Voice protocol changed')
    if protocol['evaluator']!=VERSION:raise ValueError('Keep evaluator version identical')
    facts=protocol['facts']
    if fact_ids is not None:
        if len(fact_ids)!=len(set(fact_ids)):raise ValueError('Duplicate selected facts')
        facts=[f for f in facts if f['id'] in fact_ids]
        if not facts or {f['id'] for f in facts}!=set(fact_ids):raise ValueError('Unknown selected fact')
    expected={}
    if replay_origin:
        for fact in facts:
            for pos,(kind,text) in enumerate([('SETUP',fact['setup_text']),('RECALL',fact['question'])]):
                name=fact['id']+f'-text-turn{pos}-'+kind+'-request.json'
                payload=json.loads((replay_origin/name).read_text())
                proposed=ServerClient._payload(protocol['profiles']['baseline'],text,pos==0,512)
                if payload!=proposed:raise ValueError('Replay differs from saved request: '+name)
                expected[fact['id']+'-'+kind]=payload
    folder.mkdir(parents=True,exist_ok=False);(folder/'inputs').mkdir()
    shutil.copy2(origin/'protocol.json',folder/'audio-protocol.json')
    shutil.copy2(__file__,folder/'diagnostic-driver.py')
    shutil.copy2(ROOT/'src/synth/semantic_contract.py',folder/'semantic_contract.py')
    for fact in facts:write_json(folder/'inputs'/(fact['id']+'.json'),fact)
    control=dict(audio_origin=str(origin.resolve()),audio_protocol_sha256=digest(origin/'protocol.json'),
        system_prompt=protocol['profiles']['baseline'],facts=facts,reference_server=protocol['reference_server'],
        evaluator=VERSION,max_tokens=512,input_representation='text',output_modalities=['text','audio'],
        scope='Post-hoc input-path diagnostic using the same inspected facts, not independent generalization evidence.',
        variable='Text input instead of the already-gated speech input; same model, system, question and speech output.',
        no_voice_replication='One text context per fact; do not duplicate it to inflate comparisons against two audio voices.')
    if replay_origin:
        control.update(expected_payloads=expected,replay_origin=str(replay_origin.resolve()),
            replay_protocol_sha256=digest(replay_origin/'protocol.json'),
            scope='Exact saved text requests replayed as the first context in a fresh native process; post-hoc reproducibility diagnostic.',
            variable='Omit preceding Nora context; all selected request bodies checked equal to saved requests.',
            sampling_control='Same native launch arguments; audio decoder sampling state is not independently pinned. Do not infer lifecycle causality from a difference alone.')
    write_json(folder/'protocol.json',control)
    write_json(folder/'preregistration.json',dict(protocol_sha256=digest(folder/'protocol.json'),conversation_calls=0))


def validate_origin(origin,expected_sha):
    if digest(origin/'protocol.json')!=expected_sha:raise ValueError('Voice origin protocol changed')
    if not (origin/'summary.json').exists() or not (origin/'server-exit.json').exists():
        raise ValueError('Wait for the voice screen to finish and release its server')
    protocol=json.loads((origin/'protocol.json').read_text())
    rows=[json.loads(s) for s in (origin/'results.jsonl').read_text().splitlines()]
    recalls=[r['run'] for r in rows if r.get('recall')]
    if len(recalls)!=len(set(recalls)) or set(recalls)!={r['id'] for r in protocol['runs']}:
        raise ValueError('Voice screen did not complete its exact roster')


def run(folder):
    if (folder/'server.log').exists():raise ValueError('Preserve existing diagnostic; no implicit rerun')
    protocol=json.loads((folder/'protocol.json').read_text())
    if digest(folder/'protocol.json')!=json.loads((folder/'preregistration.json').read_text())['protocol_sha256']:
        raise ValueError('Diagnostic protocol changed')
    if protocol['evaluator']!=VERSION:raise ValueError('Evaluator changed since preparation')
    validate_origin(Path(protocol['audio_origin']),protocol['audio_protocol_sha256'])
    with backend_session(ENDPOINT,test=True):
        if not port_free():raise RuntimeError('8081 occupied; do not disturb its owner')
        (folder/'reply-audio').mkdir()
        command=['wsl.exe','-e','python3',wsl_path(ROOT/'src/synth/serve_abort_variant_linux.py'),PRIVATE,'lifecycle',wsl_path(folder)]
        rows=[]
        with (folder/'server.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                client=ServerClient(ENDPOINT,timeout=120);deadline=time.monotonic()+120
                while not client.health():
                    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Private CPU server failed startup')
                    time.sleep(.5)
                actual=json.loads((folder/'server-process.json').read_text());prior=protocol['reference_server']
                if any(actual[k]!=prior[k] for k in ['binary_sha256','models','command','cuda_visible_devices']):
                    raise ValueError('Model or CPU server differs from voice screen')
                with (folder/'results.jsonl').open('x',encoding='utf-8') as results:
                    for fact in protocol['facts']:
                        for pos,(kind,text) in enumerate([('SETUP',fact['setup_text']),('RECALL',fact['question'])]):
                            turn=Turn();run_id=fact['id']+'-text';stem=run_id+f'-turn{pos}-'+kind
                            payload=client._payload(protocol['system_prompt'],text,pos==0,protocol['max_tokens'])
                            if protocol.get('expected_payloads') and payload!=protocol['expected_payloads'][fact['id']+'-'+kind]:
                                raise ValueError('Executed replay payload differs from frozen original')
                            write_json(folder/(stem+'-request.json'),payload)
                            path=folder/'reply-audio'/(stem+'.wav')
                            with wave.open(str(path),'wb') as writer:
                                writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000)
                                def receive(chunk):writer.writeframes((np.clip(chunk,-1,1)*32767).astype('<i2').tobytes())
                                client._stream(payload,turn,receive)
                            transport=bool(not turn.error and not turn.truncated and turn.audio_samples and not turn.cancel.is_set() and turn.finish_reason=='stop')
                            audio=audio_evidence(path.read_bytes(),turn.audio_samples/24000)
                            row=dict(run=run_id,fact=fact['id'],case=kind,recall=bool(pos),text=turn.text,
                                error=turn.error,finish_reason=turn.finish_reason,truncated=turn.truncated,
                                audio_s=turn.audio_samples/24000,first_audio_s=turn.ttft,transport_error=not transport,audio=audio)
                            if pos:row['evaluation']=evaluate(turn.text,fact['contract'],fact['directed'],transport and audio['valid'])
                            results.write(json.dumps(row)+'\n');results.flush();os.fsync(results.fileno());rows.append(row);print(json.dumps(row),flush=True)
                            if not transport or not audio['valid']:raise RuntimeError('Diagnostic transport/audio failed')
                        snapshot=folder.parent/(folder.name+'-durable-'+fact['id'])
                        snapshot_completed(snapshot,folder)
                        shutil.copy2(folder/'audio-protocol.json',snapshot/'partial-experiment/audio-protocol.json')
                        print('Verified pair checkpoint:',checkpoint(snapshot),flush=True)
            finally:
                stop_owned(process,folder,PRIVATE+'/lifecycle/llama-liquid-audio-server')
                write_json(folder/'server-exit.json',dict(exit_code=process.poll()))
        summary=dict(complete=True,contexts=len(protocol['facts']),outcomes=dict(Counter(r['evaluation']['outcome'] for r in rows if r['recall'])),
            deployed=False,scope=protocol['scope'],manual_review_required=True)
        write_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);p.add_argument('folder',type=Path);p.add_argument('--audio-origin',type=Path);p.add_argument('--facts',nargs='+');p.add_argument('--replay-origin',type=Path);a=p.parse_args()
    if (a.folder/(a.stage+'-failure.txt')).exists():raise SystemExit('Preserve the failed stage; use a new experiment')
    if a.stage=='prepare' and a.folder.exists():raise SystemExit('Use a new diagnostic folder')
    if a.stage=='run' and (a.folder/'server.log').exists():raise SystemExit('Diagnostic already started; preserve it')
    try:
        if a.stage=='prepare':
            if not a.audio_origin:raise ValueError('Audio origin is required')
            prepare(a.audio_origin.resolve(),a.folder.resolve(),a.facts,a.replay_origin)
        else:run(a.folder.resolve())
    except Exception:
        if a.folder.exists():(a.folder/(a.stage+'-failure.txt')).write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
