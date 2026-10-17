"""Offline negative controls for echo appearing after no-coupling calibration."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import traceback

import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.acoustic_stress import read
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder,source):
    import torch
    torch.set_num_threads(1)
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    spec=importlib.util.spec_from_file_location('uncoupled_candidate',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (folder/'candidate-duplex.py').write_bytes(source.read_bytes())
    user_source=ROOT/'outputs/experiments/synthetic-ad-v1/inputs/david-C2.wav'
    user=read(str(user_source));user *= .12/float(np.sqrt(np.mean(user[abs(user)>.01]**2)))
    shutil.copy2(user_source,folder/'speech-source.wav');shutil.copy2(ROOT/'data/synth/_assistant.wav',folder/'far-source.wav')
    begin=np.flatnonzero(abs(user)>.02)[0];user=user[max(0,begin-160):]
    size=((max(8*16000,2*16000+len(user))+319)//320)*320
    far=np.resize(read('_assistant.wav'),size)
    definitions=[dict(delay_ms=d,echo_gain=g,kind='echo_appears') for d in [20,80,180] for g in [.01,.05,.2,1.]]
    definitions += [dict(delay_ms=0,echo_gain=0.,kind=k) for k in ['headphones_silent','headphones_speech']]
    rows=[]
    for index,definition in enumerate(definitions):
        case=folder/f'case-{index:02d}';case.mkdir()
        raw=np.random.default_rng(20261020+index).normal(0,.0005,size).astype(np.float32)
        if definition['kind']=='echo_appears':
            for tail,scale in [(0,1),(25,.3),(70,.15)]:
                offset=(definition['delay_ms']+tail)*16
                echo=np.zeros(size,np.float32);echo[offset:]=far[:-offset]*definition['echo_gain']*scale
                raw[32000:]+=echo[32000:]
        if definition['kind']=='headphones_speech':raw[16000:16000+len(user)]+=.15*user
        np.savez_compressed(case/'signals.npz',raw=raw,render=far)
        for version,Front in [('baseline',DuplexFrontend),('candidate',module.DuplexFrontend)]:
            front=Front(stream_delay_ms=80);front.warmup();onsets=[];trace=[]
            for offset in range(0,size,320):
                front.push_far(far[offset:offset+320]);_,started,ended,clean=front.update(raw[offset:offset+320])
                if started:onsets.append((offset+320)/16000)
                trace.append(dict(time_s=(offset+320)/16000,uncoupled=getattr(front,'_uncoupled_render',False),
                    echo_score=front.echo_score,probability=front.probability,started=started))
                if ended:front.reset()
            speech=definition['kind']=='headphones_speech'
            false=[t for t in onsets if not speech or t<1.]
            latency=next((t-1 for t in onsets if t>=1.),None) if speech else None
            calibrated=any(t['uncoupled'] and t['time_s']<2 for t in trace)
            revoked=any(not t['uncoupled'] and t['time_s']>=2 for t in trace) if calibrated else None
            passed=not false and not front.aec_error and (not speech or latency is not None and latency<.5)
            if version=='candidate' and definition['kind']=='echo_appears' and definition['echo_gain']>=.2:passed=passed and calibrated and revoked
            row=dict(case=index,version=version,definition=definition,onsets=onsets,false_onsets=false,
                latency_s=latency,calibrated=calibrated,revoked=revoked,aec_error=front.aec_error,passed=bool(passed))
            rows.append(row);(case/(version+'-trace.json')).write_text(json.dumps(trace))
            (folder/'summary.json').write_text(json.dumps(dict(cases=rows,deployed=False,
                scope='Synthetic acoustic transition tests only; not physical-room acceptance.'),indent=2))
            print(json.dumps(row),flush=True)
    return all(row['passed'] for row in rows if row['version']=='candidate')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
    try:passed=run(a.folder,a.candidate)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
    raise SystemExit(0 if passed else 1)
