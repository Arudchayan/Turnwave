"""Replay the human A-D protocol using two independent Windows TTS voices.

Saves inputs, ASR checks, generated reply WAVs, VAD boundaries and strict rubrics.
The shared backend must be available with the UI stopped. Each completed case
is flushed and synced; never reuse a results folder to overwrite an experiment.
"""
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import wave
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from backend_session import backend_session
from client import ServerClient, Turn, wav16k_bytes
from duplex import DuplexFrontend
from read_aloud_rubrics import grade, VERSION as RUBRIC_VERSION


def normalize(text):
    text=text.lower().replace('nine thirty','930').replace('ten fifteen','1015')
    for word,number in [('twenty','20'),('twelve','12'),('ten','10'),('nine','9'),('seven','7'),('six','6'),('five','5'),('four','4'),('three','3'),('two','2'),('one','1')]:
        text=re.sub(r'\b'+word+r'\b',number,text)
    return ''.join(re.findall('[a-z0-9]+',text))


def grade_v1_historical(case,text):
    compact=normalize(text);lower=text.lower()
    exact={'A3':'4','A4':'banana','A5':'9','A6':'riverwindoworange','C1':'b7d2','C2':'7','C3':'red','C5':'blue','D1':'coffee','D2':'6','D3':'window'}
    if case in exact:return dict(rubric='exact normalized answer: '+exact[case],passed=compact==exact[case])
    if case in ['A2','D4']:return dict(rubric='manual content review / deliberate long-pause probe',passed=None)
    required={'A1':['ready'],'B2':['maya'],'B3':['green'],'B5':['bristol','tuesday','930'],
              'B6':['bristol','tuesday','1015'],'B7':['930','1015'],'B8':['maya','green'],
              'C4':['small','tea'],'C6':['cat']}
    if case in required:
        passed=all(word in compact for word in required[case])
        if case in ['B2','B8'] and re.search(r"\b(i(?:'|’)?m|i am|my name is)\s+maya\b",lower):passed=False
        return dict(rubric='required facts: '+', '.join(required[case])+'; correct user ownership',passed=passed)
    return dict(rubric='introduction acknowledged; retention graded by subsequent questions',passed=bool(text.strip()))


def append(path,row):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=True)+'\n');stream.flush();os.fsync(stream.fileno())


def read_audio(path):
    with wave.open(str(path),'rb') as reader:
        assert reader.getframerate()==16000 and reader.getnchannels()==1 and reader.getsampwidth()==2
        return np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768


def main(folder):
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    if (folder/'results.jsonl').exists():raise RuntimeError('This experiment already has results. Use a new folder; previous runs must remain intact.')
    client=ServerClient();front=DuplexFrontend(aec=False);front.warmup()
    inputs={};verification={};rows=[]
    (folder/'reply-audio').mkdir(exist_ok=False);(folder/'inputs').mkdir(exist_ok=False)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
                  client_sha256=hashlib.sha256((ROOT/'src/client.py').read_bytes()).hexdigest(),
                  runner_patch_sha256=hashlib.sha256((ROOT/'patches/liquid-turn-reset.patch').read_bytes()).hexdigest(),
                  rubric_version=RUBRIC_VERSION,grading_sha256=hashlib.sha256((ROOT/'src/synth/read_aloud_rubrics.py').read_bytes()).hexdigest(),type='synthetic A-D replay',limits='Model-level direct WAV replay plus offline VAD. No real speakers or microphone. ASR checks are from the same model family, not an independent human validation.')
    (folder/'run-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with backend_session(test=True):
        # ASR resets context: complete every fixture check BEFORE conversation.
        for voice in protocol['voices']:
            for case in protocol['cases']:
                key=voice['id']+'-'+case['id'];audio=read_audio(folder/'fixtures'/(key+'.wav'))
                if case['id']=='D2':audio=audio*.35
                data=wav16k_bytes(audio);path=folder/'inputs'/(key+'.wav');path.write_bytes(data);inputs[key]=data
                transcript=client.asr_text(data);match=difflib.SequenceMatcher(a=normalize(case['spoken_text']),b=normalize(transcript)).ratio()
                front.reset();starts=[];ends=[]
                padded=np.concatenate([audio,np.zeros(16000,np.float32)])
                for offset in range(0,len(padded)-512,512):
                    _,started,ended,_=front.update(padded[offset:offset+512])
                    if started:starts.append(round((offset+512)/16000,3))
                    if ended:ends.append(round((offset+512)/16000,3))
                verification[key]=dict(key=key,script=case['script'],asr=transcript,similarity=round(match,3),fixture_check_pass=match>=.9,
                                       audio_s=round(len(audio)/16000,3),input_sha256=hashlib.sha256(data).hexdigest(),vad_starts_s=starts,vad_ends_s=ends,quiet_gain=.35 if case['id']=='D2' else 1)
                append(folder/'fixture-verification.jsonl',verification[key]);print('Verified',key,round(match,3),flush=True)
        for voice in protocol['voices']:
            first=True
            for case in protocol['cases']:
                key=voice['id']+'-'+case['id'];turn=Turn();path=folder/'reply-audio'/(key+'.wav')
                with wave.open(str(path),'wb') as writer:
                    writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000)
                    def receive(samples):writer.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
                    client.chat_turn(inputs[key],turn,first,max_tokens=512,on_audio=receive)
                result=grade(case['id'],turn.text)
                row=dict(key=key,case=case['id'],voice=voice['id'],text=turn.text,error=turn.error,
                         first=first,first_audio_s=round(turn.ttft,3) if turn.ttft is not None else None,audio_s=round(turn.audio_samples/24000,3),
                         fixture_check_pass=verification[key]['fixture_check_pass'],rubric=result,
                         passed=result['passed'] and not turn.error and turn.audio_samples>0 if result['passed'] is not None else None)
                first=bool(turn.error or turn.cancel.is_set());rows.append(row);append(folder/'results.jsonl',row)
                checkpoint=dict(completed=len(rows),total=len(protocol['voices'])*len(protocol['cases']),latest=key,results_synced=True)
                temporary=folder/'checkpoint.tmp';temporary.write_text(json.dumps(checkpoint,indent=2),encoding='utf-8');temporary.replace(folder/'checkpoint.json')
                print(json.dumps(row),flush=True)
    report=dict(completed=len(rows),total=len(protocol['voices'])*len(protocol['cases']),passed=sum(r['passed'] is True for r in rows),
                failed=sum(r['passed'] is False for r in rows),manual_review=sum(r['passed'] is None for r in rows),
                unverified_inputs=sum(not v['fixture_check_pass'] for v in verification.values()),
                finished_utc=datetime.now(timezone.utc).isoformat(),scope=manifest['limits'])
    (folder/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    from archive_experiment import capture_files
    capture_files(folder/'runtime-after')
    from checkpoint_experiment import checkpoint
    print('Verified backup:',checkpoint(folder),flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)
