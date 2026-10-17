"""Compare warmed frontend processing time on saved 20 ms audio frames.

No devices or backend requests. Two counterbalanced passes include playback
reference and endpoint resets. This is processing time, not voice reply latency.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder,candidate):
    import torch
    torch.set_num_threads(1)
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    spec=importlib.util.spec_from_file_location('profile_candidate',candidate);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (folder/'candidate-duplex.py').write_bytes(candidate.read_bytes())
    sources=[];rows=[]
    for case in [6,13,16]:
        path=ROOT/f'outputs/experiments/uncoupled-capture-v1/case-{case:02d}/candidate/signals.npz'
        data=np.load(path);sources.append(dict(case=case,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        raw=data['raw'];far=data['render'];voiced=np.flatnonzero(abs(far)>1e-8);render_end=int(voiced[-1]+1) if len(voiced) else 0
        for cycle,order in enumerate([['baseline','candidate'],['candidate','baseline']]):
            for version in order:
                front=(DuplexFrontend if version=='baseline' else module.DuplexFrontend)(stream_delay_ms=80);front.warmup()
                milliseconds=[];cleared=False
                for offset in range(0,len(raw),320):
                    started=time.perf_counter()
                    if offset<render_end:front.push_far(far[offset:offset+320])
                    elif not cleared:front.clear_playback();cleared=True
                    _,_,ended,_=front.update(raw[offset:offset+320])
                    if ended:front.reset()
                    milliseconds.append((time.perf_counter()-started)*1000)
                values=np.asarray(milliseconds)
                row=dict(case=case,cycle=cycle,version=version,frames=len(values),p50_ms=float(np.percentile(values,50)),
                    p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),max_ms=float(values.max()),
                    mean_ms=float(values.mean()),over_20ms=int(np.sum(values>20)),processing_s=float(values.sum()/1000),audio_s=len(raw)/16000)
                rows.append(row);np.save(folder/f'{case}-{cycle}-{version}-timings-ms.npy',values)
                (folder/'results.jsonl').open('a').write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    report=dict(cases=rows,sources=sources,frame_budget_ms=20,torch_threads=1,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        candidate_within_budget=all(r['p99_ms']<20 for r in rows if r['version']=='candidate'),
        deployed=False,scope='Saved-audio frontend processing in this host environment; no OS audio callbacks, model backend, playback or physical-room acceptance.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    return report['candidate_within_budget']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
    try:passed=run(a.folder,a.candidate)
    except Exception:
        if a.folder.exists():(a.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
    raise SystemExit(0 if passed else 1)
