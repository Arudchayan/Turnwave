"""Trace saved synthetic onset failures and a raw, no-reference counterfactual.

No hardware or model backend is accessed. Counterfactual raw speech detection
is diagnostic only: echo-only controls can trigger it, so it is not a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.dual_aec_experiment import make_candidate


def run(folder,origin,extension=None,normal=False):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    Candidate,instances=(DuplexFrontend,[]) if normal else make_candidate(folder,extension,True,True)
    rows=[];sources=[]
    for case in [4,10,11,13,15,16,17]:
        source=origin/f'case-{case:02d}'/'candidate'/'signals.npz'
        target=folder/f'case-{case:02d}';target.mkdir();shutil.copy2(source,target/'signals.npz')
        sources.append(dict(case=case,path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        signals=np.load(source);raw=signals['raw'];far=signals['render']
        versions=[('normal',DuplexFrontend,True),('raw-counterfactual',DuplexFrontend,False)] if normal else [('baseline',DuplexFrontend,True),('candidate',Candidate,True),('raw-counterfactual',DuplexFrontend,False)]
        for name,factory,reference in versions:
            front=factory(stream_delay_ms=80,aec=reference,end_silence_ms=1120);front.warmup()
            trace=[];starts=[];original=front._prob
            def probability(chunk):
                value=original(chunk);end=front._sample_clock;raw_chunk=raw[end-512:end]
                clean_rms=float(np.sqrt(np.mean(chunk**2)));raw_rms=float(np.sqrt(np.mean(raw_chunk**2)))
                trace.append(dict(end_s=end/16000,probability=value,vad_rms=float(np.sqrt(np.mean(chunk**2))),
                    raw_rms=float(np.sqrt(np.mean(raw_chunk**2))),echo_score=front.echo_score,
                    raw_echo_score=getattr(front,'raw_echo_score',front.echo_score),far_active=front._far_active,
                    normal_residual_veto=bool(front._far_active and clean_rms<front.residual_floor and clean_rms<raw_rms*.2),
                    correlation_veto=bool(front._far_active and front.echo_score>=.6)))
                return value
            front._prob=probability
            # First onset only. No resets or cancellation are injected: saved
            # render is authoritative through the original candidate's stop.
            for offset in range(0,min(len(raw),4*16000),320):
                if reference:front.push_far(far[offset:offset+320])
                _,started,_,_=front.update(raw[offset:offset+320])
                if started:starts.append((offset+320)/16000)
            (target/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in trace))
            onset=[r for r in trace if 1<=r['end_s']<=1.5]
            row=dict(case=case,version=name,starts=starts,max_probability_first500ms=max(r['probability'] for r in onset),
                hot_frames_first500ms=sum(r['probability']>=.5 for r in onset),residual_rejections=front.residual_rejections)
            if normal:
                row['eligible_frames_first500ms']=sum(r['probability']>=.5 and not r['normal_residual_veto'] and not r['correlation_veto'] for r in onset)
            rows.append(row);print(json.dumps(row),flush=True)
            if name=='candidate':
                record=instances[-1].records
                if record:np.savez_compressed(target/'aec.npz',**{key:np.concatenate([r[i] for r in record]) for i,key in enumerate(['near','far','linear','postfiltered'])})
                instances.clear()
    (folder/'summary.json').write_text(json.dumps(dict(cases=rows,sources=sources,deployed=False,
        scope='First-onset diagnosis using saved synthetic mixtures. Raw/no-reference is a counterfactual and unsafe with echo. Saved render stops at the original candidate cancellation; later replay is not a new closed-loop experiment.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('origin',type=Path);parser.add_argument('extension',type=Path,nargs='?');parser.add_argument('--normal',action='store_true');args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    if not args.normal and args.extension is None:parser.error('Provide the private extension or select --normal')
    try:run(args.folder,args.origin,args.extension,args.normal)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
