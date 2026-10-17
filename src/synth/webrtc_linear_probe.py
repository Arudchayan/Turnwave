"""Compare installed WebRTC output with an isolated linear-output extension."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from pywebrtc_audio import AudioProcessor
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def vad_metrics(audio):
    front=DuplexFrontend(aec=False);front.warmup();starts=[];values=[]
    for index in range(0,len(audio)-512,512):
        _,started,_,_=front.update(audio[index:index+512])
        if started:starts.append((index+512)/16000)
        if 1<=(index+512)/16000<=1.5:values.append(front.probability)
    return dict(starts=starts,max_prob_first500ms=max(values),hot_frames_first500ms=sum(p>=.5 for p in values))


def run(folder):
    with (folder/'probe-started.json').open('x') as marker: marker.write('{}\n')
    capture_files(folder/'source')
    modules=list((folder/'build/Release').glob('_webrtc_audio_linear*.pyd'))
    if len(modules)!=1:raise RuntimeError('Expected one completed private extension build')
    module_path=modules[0].resolve()
    spec=importlib.util.spec_from_file_location('_webrtc_audio_linear',module_path)
    custom=importlib.util.module_from_spec(spec);spec.loader.exec_module(custom)
    rows=[]
    for delay in [80,180]:
        for gain in [0,.15,.3]:
            key=f'delay{delay}-user{gain}'
            origin=ROOT/'outputs/experiments/quiet-trace-reproduced-v1'/(key+'.npz')
            saved=np.load(origin);near=saved['near'];far=saved['far']
            # Identical per-frame calls and configuration, apart from exporting
            # the linear AEC buffer in the private extension.
            baseline=AudioProcessor(sample_rate=16000,num_channels=1,echo_cancellation=True,stream_delay_ms=80)
            candidate=custom.AudioProcessor(sample_rate=16000,num_channels=1,echo_cancellation=True,stream_delay_ms=80)
            normal=[];rebuilt=[];linear=[]
            for index in range(0,len(near)-159,160):
                a=near[index:index+160];b=far[index:index+160]
                normal.append(baseline.process(a,b));rebuilt.append(candidate.process(a,b));linear.append(candidate.linear_output)
            normal=np.concatenate(normal);rebuilt=np.concatenate(rebuilt);linear=np.concatenate(linear)
            difference=float(np.max(np.abs(normal-rebuilt)))
            np.savez_compressed(folder/(key+'-comparison.npz'),near=near,far=far,installed=normal,rebuilt=rebuilt,linear=linear)
            row=dict(delay_ms=delay,user_gain=gain,normal_max_absolute_difference=difference,
                normal_rms_difference=float(np.sqrt(np.mean((normal-rebuilt)**2))),
                installed_vad=vad_metrics(normal),rebuilt_vad=vad_metrics(rebuilt),linear_vad=vad_metrics(linear))
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(cases=rows,normal_output_parity_passed=all(r['normal_max_absolute_difference']<=1e-4 for r in rows),
        parity_tolerance=1e-4,extension_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        scope='VAD-only diagnostic on full AEC outputs. Correlation/residual onset guards are omitted here; this does not score a deployable barge-in policy.',deployed=False)
    (folder/'parity-summary.json').write_text(json.dumps(report,indent=2))
    shutil.copy2(module_path,folder/module_path.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    if (args.folder/'parity-summary.json').exists() or (args.folder/'probe-started.json').exists():parser.error('Preserve previous results; use a new folder')
    try:run(args.folder)
    except Exception:
        import traceback
        (args.folder/'probe-failure.txt').write_text(traceback.format_exc());raise
    finally:print('Verified backup:',checkpoint(args.folder),flush=True)
