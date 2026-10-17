"""Trace quiet-overlap failures with echo-only and speech-only controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.acoustic_stress import read
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def trace_case(folder,far,user,delay_ms,gain):
    name=f'delay{delay_ms}-user{gain}';echo=np.zeros(len(far),np.float32)
    for tail_ms,scale in [(0,1),(25,.3),(70,.15)]:
        offset=(delay_ms+tail_ms)*16;echo[offset:]+=far[:-offset]*scale
    mic=echo+np.random.default_rng(20261016).normal(0,.0005,len(far)).astype(np.float32)
    count=min(len(user),len(mic)-16000);mic[16000:16000+count]+=gain*user[:count]
    front=DuplexFrontend(stream_delay_ms=80);front.warmup()
    history=np.zeros(0,np.float32);samples=[];trace=[];starts=[];original=front._prob
    def probability(chunk):
        nonlocal history
        value=original(chunk);history=np.concatenate((history,chunk))[-1600:]
        end=front._sample_clock;raw=mic[max(0,end-512):end]
        clean_corr=DuplexFrontend._echo_similarity(SimpleNamespace(_near_history=history,_far_history=front._far_history))
        trace.append(dict(end_s=end/16000,probability=value,raw_corr=front.echo_score,
            clean_corr=clean_corr,raw_rms=float(np.sqrt(np.mean(raw**2))),clean_rms=float(np.sqrt(np.mean(chunk**2)))))
        return value
    front._prob=probability
    for index in range(0,len(far)-512,512):
        front.push_far(far[index:index+512]);_,started,_,clean=front.update(mic[index:index+512]);samples.append(clean)
        if started:starts.append((index+512)/16000)
    (folder/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in trace))
    np.savez_compressed(folder/(name+'.npz'),near=mic,far=far,clean=np.concatenate(samples))
    onset=[row for row in trace if 1<=row['end_s']<=1.5];hot=[row for row in onset if row['probability']>=.5]
    return dict(delay_ms=delay_ms,user_gain=gain,starts=starts,onset_hot_frames=len(hot),
        hot_blocked_raw_corr=sum(row['raw_corr']>=.6 for row in hot),
        max_prob_first500ms=max(row['probability'] for row in onset),
        max_prob_overall=max(row['probability'] for row in trace),
        top_onset=sorted(onset,key=lambda row:row['probability'],reverse=True)[:4])


def main(folder):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    fixtures=folder/'fixtures';fixtures.mkdir();manifest=[]
    for name in ['_assistant.wav','user_barge.wav']:
        original=ROOT/'data/synth'/name;shutil.copy2(original,fixtures/name)
        manifest.append(dict(name=name,sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    (folder/'protocol.json').write_text(json.dumps(dict(seed=20261016,fixtures=manifest,
        onset_s=1,window_s=.5,configured_delay_ms=80,
        note='Correlation fields use frontend call histories; VAD probability and RMS use exact 512-sample VAD frames.'),indent=2))
    far=np.tile(read('_assistant.wav'),2)[:128000];user=read('user_barge.wav')
    first=np.flatnonzero(abs(user)>.02)[0];user=user[max(0,first-160):]
    rows=[]
    for delay in [80,180]:
        for gain in [0,.15,.3]:
            row=trace_case(folder,far,user,delay,gain);rows.append(row);print(json.dumps(row),flush=True)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))
    controls=[]
    for gain in [.15,.3]:
        near=np.zeros(128000,np.float32);count=min(len(user),len(near)-16000);near[16000:16000+count]=gain*user[:count]
        near+=np.random.default_rng(20261016).normal(0,.0005,len(near)).astype(np.float32)
        front=DuplexFrontend(aec=False);front.warmup();starts=[];prob=[]
        for index in range(0,len(near)-512,512):
            _,started,_,_=front.update(near[index:index+512])
            if started:starts.append((index+512)/16000)
            if 1<=(index+512)/16000<=1.5:prob.append(front.probability)
        row=dict(user_gain=gain,starts=starts,max_prob_first500ms=max(prob),first_detection_latency_s=round(starts[0]-1,3) if starts else None)
        controls.append(row);np.savez_compressed(folder/f'near-only-{gain}.npz',near=near)
    (folder/'near-only-controls.json').write_text(json.dumps(controls,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:main(args.folder)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
