"""Separate AEC startup from quiet-overlap failures with fixed acoustic paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.acoustic_stress import read
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.dual_aec_experiment import make_candidate


def run(folder,extension,phase_aligned=False,reset_vad_at_common=False):
    if reset_vad_at_common and not phase_aligned:raise ValueError('VAD alignment requires phase-aligned audio')
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    Candidate,instances=make_candidate(folder,extension,True,True,.3,100)
    fixtures=[]
    for name in ['_assistant.wav','user_barge.wav']:
        path=ROOT/'data/synth'/name;shutil.copy2(path,folder/name)
        fixtures.append(dict(name=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    native=ROOT/'outputs/experiments/webrtc-linear-v1/variant/vendor/webrtc_audio/api/echo_canceller3_config.h'
    shutil.copy2(native,folder/'echo_canceller3_config.h')
    protocol=dict(seed=20261019,duration_s='onset + 7' if phase_aligned else 14,onsets_s=[1,3,6],delays_ms=[80,180],near_gains=[.15,.3],
        echo_gain=1.,noise_sigma=.0005,configured_delay_ms=80,fixtures=fixtures,
        candidate=dict(name='V4',correlation_window_ms=100,minimum_guard_correlation=.3,residual_floor=.005),
        phase_aligned=phase_aligned,
        reset_vad_before_common=reset_vad_at_common,frame_samples=320 if reset_vad_at_common else 512,
        scope=('Identical final one-second preparation and subsequent audio with extra adaptation prefixed; aligned input hashes checked.' if phase_aligned else
            'Onset positions change within continuous render: adaptation duration AND overlapping phonemes differ; this version cannot isolate warm-up causality.')+
            ' Offline AEC/VAD onset gate, not capture quality. Each condition constructs a fresh processor. No devices or inference backend.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=read('_assistant.wav');common_far=np.resize(base,8*16000)
    user=read('user_barge.wav');first=np.flatnonzero(abs(user)>.02)[0];user=user[max(0,first-160):]
    cases=[];aligned={};alignment_checks=[]
    for delay in [80,180]:
        for onset,gain in [(None,0)]+[(o,g) for o in [1,3,6] for g in [.15,.3]]:
            if phase_aligned:
                extra=((onset or 6)-1)*16000
                far=np.concatenate((np.resize(base,extra),common_far))
                noise=np.concatenate((np.random.default_rng(20261020+delay).normal(0,.0005,extra),
                    np.random.default_rng(20261019+delay).normal(0,.0005,len(common_far)))).astype(np.float32)
            else:
                far=np.resize(base,14*16000)
                noise=np.random.default_rng(20261019+delay).normal(0,.0005,len(far)).astype(np.float32)
            echo=np.zeros_like(far)
            for tail,scale in [(0,1),(25,.3),(70,.15)]:
                offset=(delay+tail)*16;echo[offset:]+=far[:-offset]*scale
            cases.append((dict(delay_ms=delay,onset_s=onset,user_gain=gain),echo+noise,far))
    rows=[]
    with (folder/'results.jsonl').open('x') as output:
        for index,(definition,mic,far) in enumerate(cases):
            mic=mic.copy();onset=definition['onset_s']
            if onset is not None:
                start=onset*16000;count=min(len(user),len(mic)-start);mic[start:start+count]+=user[:count]*definition['user_gain']
                if phase_aligned:
                    begin=round((onset-.5)*16000);finish=round((onset+6.5)*16000)
                    hashes={k:hashlib.sha256(a[begin:finish].tobytes()).hexdigest() for k,a in [('near',mic),('far',far)]}
                    key=(definition['delay_ms'],definition['user_gain'])
                    if key in aligned and aligned[key]!=hashes:raise RuntimeError('Aligned onset windows differ')
                    aligned[key]=hashes;alignment_checks.append(dict(case=index,hashes=hashes))
            arrays=dict(near=mic,far=far)
            for name,factory in [('baseline',DuplexFrontend),('candidate',Candidate)]:
                front=factory(stream_delay_ms=80);front.warmup();starts=[];clean=[]
                frame=320 if reset_vad_at_common else 512
                common_start=((onset or 6)-1)*16000
                for offset in range(0,len(far)-frame,frame):
                    if reset_vad_at_common and offset==common_start:
                        # reset clears Silero state and partial VAD frames,
                        # while retaining native AEC adaptation. This gives
                        # every warm-up the same VAD phase and entry state.
                        front.reset()
                    front.push_far(far[offset:offset+frame]);_,started,_,audio=front.update(mic[offset:offset+frame]);clean.append(audio)
                    if started:starts.append((offset+frame)/16000)
                false=[s for s in starts if onset is None or s<onset]
                detected=next((s-onset for s in starts if s>=onset),None) if onset is not None else None
                row=dict(case=index,version=name,definition=definition,starts=starts,false_onsets=false,
                    barge_s=round(detected,3) if detected is not None else None,aec_error=front.aec_error,
                    passed=not false and not front.aec_error and (onset is None or detected is not None and detected<.5))
                rows.append(row);output.write(json.dumps(row)+'\n');output.flush();os.fsync(output.fileno());print(json.dumps(row),flush=True)
                arrays[name+'_clean']=np.concatenate(clean)
                if name=='candidate':
                    record=instances[-1].records
                    if record:
                        arrays['candidate_linear']=np.concatenate([r[2] for r in record])
                        arrays['candidate_postfiltered']=np.concatenate([r[3] for r in record])
                    instances.clear()
            np.savez_compressed(folder/f'case-{index:02d}.npz',**arrays)
    summary={name:dict(passed=sum(r['passed'] for r in rows if r['version']==name),total=len(cases)) for name in ['baseline','candidate']}
    (folder/'alignment-checks.json').write_text(json.dumps(alignment_checks,indent=2))
    (folder/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows,deployed=False),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('extension',type=Path);parser.add_argument('--phase-aligned',action='store_true');parser.add_argument('--reset-vad-at-common',action='store_true');args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder,args.extension,args.phase_aligned,args.reset_vad_at_common)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
