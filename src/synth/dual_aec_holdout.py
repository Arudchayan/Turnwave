"""New voices, room paths and delays for the frozen dual-AEC prototype."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.acoustic_stress import read
from synth.dual_aec_experiment import make_candidate
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder,extension):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    Candidate,instances=make_candidate(folder,extension,True)
    origin=ROOT/'outputs/experiments/synthetic-ad-v1'
    checks={r['key']:r for r in map(json.loads,(origin/'independent-asr.jsonl').read_text().splitlines())}
    voices={};manifest=[]
    for key in ['david-C2','zira-A4']:
        source=origin/'inputs'/(key+'.wav');data=source.read_bytes()
        if not checks[key]['input_check_pass'] or hashlib.sha256(data).hexdigest()!=checks[key]['sha256']:
            raise RuntimeError('Unverified held-out input')
        (folder/(key+'.wav')).write_bytes(data)
        with wave.open(str(source),'rb') as reader:
            assert reader.getframerate()==16000
            audio=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
        first=np.flatnonzero(abs(audio)>.02)[0];audio=audio[max(0,first-160):]
        active=audio[abs(audio)>.01];factor=.12/float(np.sqrt(np.mean(active**2)))
        voices[key]=audio*factor;manifest.append(dict(key=key,sha256=checks[key]['sha256'],normalization_factor=factor))
    (folder/'protocol.json').write_text(json.dumps(dict(seed=20261017,voices=manifest,
        delays_ms=[60,140],echo_gains=[.3,.8],user_gains=[.15,.3],onset_s=1.2,
        reflection_delays_ms=[0,15,95],reflection_gains=[1,.25,.1],noise_sigma=.0007,
        scope='Frozen policy; new synthetic voices and room parameters, including matched echo-only and speech-only controls.'),indent=2))
    far=np.tile(read('_assistant.wav'),2)[:128000];shutil.copy2(ROOT/'data/synth/_assistant.wav',folder/'far-source.wav')
    scenarios=[];rng=np.random.default_rng(20261017)
    for delay in [60,140]:
        for echo_gain in [.3,.8]:
            echo=np.zeros(len(far),np.float32)
            for tail,scale in [(0,1),(15,.25),(95,.1)]:
                offset=(delay+tail)*16;echo[offset:]+=far[:-offset]*scale*echo_gain
            noise=rng.normal(0,.0007,len(far)).astype(np.float32)
            for voice,gain in [(None,0)]+[(v,g) for v in voices for g in [.15,.3]]:
                mic=echo+noise;onset=19200
                if voice:
                    mic=mic.copy();count=min(len(voices[voice]),len(mic)-onset);mic[onset:onset+count]+=voices[voice][:count]*gain
                scenarios.append((dict(delay_ms=delay,echo_gain=echo_gain,voice=voice,user_gain=gain),far,mic,onset))
    for voice in voices:
        for gain in [.15,.3]:
            mic=np.zeros(len(far),np.float32);onset=19200;count=min(len(voices[voice]),len(mic)-onset)
            mic[onset:onset+count]=voices[voice][:count]*gain
            scenarios.append((dict(speech_only=True,voice=voice,user_gain=gain),np.zeros_like(far),mic,onset))
    rows=[]
    with (folder/'results.jsonl').open('x') as output:
        for index,(scenario,render,mic,onset) in enumerate(scenarios):
            arrays=dict(near=mic,far=render)
            for version,frontend in [('baseline',DuplexFrontend),('candidate',Candidate)]:
                front=frontend(stream_delay_ms=80);front.warmup();starts=[];clean=[]
                for offset in range(0,len(render)-512,512):
                    # Feed silence too: a quiet source is not active playback.
                    front.push_far(render[offset:offset+512]);_,started,_,signal=front.update(mic[offset:offset+512]);clean.append(signal)
                    if started:starts.append((offset+512)/16000)
                false=[t for t in starts if not scenario.get('voice') or t<onset/16000]
                latency=next((t-onset/16000 for t in starts if t>=onset/16000),None) if scenario.get('voice') else None
                passed=not false and not front.aec_error and (not scenario.get('voice') or latency is not None and latency<.5)
                row=dict(case=index,version=version,scenario=scenario,starts=starts,false_starts=len(false),
                    barge_s=round(latency,3) if latency is not None else None,passed=bool(passed),aec_error=front.aec_error)
                rows.append(row);output.write(json.dumps(row)+'\n');output.flush();os.fsync(output.fileno());print(json.dumps(row),flush=True)
                arrays[version+'_clean']=np.concatenate(clean)
            np.savez_compressed(folder/f'case-{index:02d}.npz',**arrays)
            # The prototype's per-frame diagnostics were saved above. Release
            # them between cases rather than retaining a growing audio history.
            instances.clear()
    report={version:dict(passed=sum(r['passed'] for r in rows if r['version']==version),total=len(scenarios)) for version in ['baseline','candidate']}
    report['regressions']=[i for i in range(len(scenarios)) if rows[2*i]['passed'] and not rows[2*i+1]['passed']]
    report['improvements']=[i for i in range(len(scenarios)) if not rows[2*i]['passed'] and rows[2*i+1]['passed']]
    report['deployed']=False;(folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('extension',type=Path);args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder,args.extension)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
