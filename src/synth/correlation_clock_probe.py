"""Check correlation alignment against the exact VAD frame boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.dual_aec_experiment import make_candidate


def run(folder,extension=None):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source');rows=[]
    Factory,instances=make_candidate(folder,extension,True,True,.3) if extension else (DuplexFrontend,[])
    for case in [13,15]:
        source=ROOT/f'outputs/experiments/closed-loop-capture-v3/case-{case:02d}/candidate/signals.npz'
        shutil.copy2(source,folder/f'case-{case:02d}.npz');signals=np.load(source)
        raw=signals['raw'][:32000];far=signals['render'][:32000]
        for frame in [160,320,512,1024]:
            front=Factory(aec=False);trace=[];vad_history=[]
            def probability(chunk):
                vad_history.append(chunk.copy())
                end=front._sample_clock
                aligned_near=np.concatenate(vad_history)[-1600:] if extension else raw[:end][-1600:]
                expected=DuplexFrontend._echo_similarity(SimpleNamespace(_near_history=aligned_near,_far_history=far[:end][-12000:])) if front._far_active else 0.
                trace.append(dict(end_s=end/16000,actual=front.echo_score,aligned_expected=expected,
                    absolute_error=abs(front.echo_score-expected),pending_audio_ms=len(front._vad_buf)/16))
                # Constant cold VAD: only correlation timing is under test;
                # no learned model, hardware, cancellation or gate is used.
                return 0.
            front._prob=probability
            for offset in range(0,len(raw),frame):
                front.push_far(far[offset:offset+frame]);front.update(raw[offset:offset+frame])
            (folder/f'case-{case:02d}-frame{frame}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
            row=dict(case=case,frame_samples=frame,frames=len(trace),different_frames=sum(r['absolute_error']>1e-6 for r in trace),
                maximum_error=max(r['absolute_error'] for r in trace),maximum_pending_audio_ms=max(r['pending_audio_ms'] for r in trace),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
            instances.clear()
    (folder/'summary.json').write_text(json.dumps(dict(cases=rows,deployed=False,
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest() if extension else None,
        scope='Clock invariant only: correlation uses raw microphone (or experimental linear AEC) and consumed render ending at the exact VAD boundary. Constant cold VAD; no hardware or behavioral acceptance score.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--extension',type=Path);args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder,args.extension)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
