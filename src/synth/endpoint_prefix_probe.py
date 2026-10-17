"""Offline semantic-endpoint checks at the app's ordinary VAD boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder):
    import onnxruntime as ort
    import torch
    from transformers import WhisperFeatureExtractor
    from importlib.metadata import version
    with (folder/'probe-started.json').open('x') as marker:marker.write('{}\n')
    capture_files(folder/'source');torch.set_num_threads(2)
    options=ort.SessionOptions();options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(folder/'smart-turn-v3.2-cpu.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    extractor=WhisperFeatureExtractor(chunk_length=8)
    origin=ROOT/'outputs/experiments/synthetic-ad-v1';checks={r['key']:r for r in map(json.loads,(origin/'independent-asr.jsonl').read_text().splitlines())}
    rows=[]
    with (folder/'results.jsonl').open('x') as output:
        for key in ['david-C2','zira-C2','david-A4','david-D4']:
            path=origin/'inputs'/(key+'.wav');data=path.read_bytes()
            if not checks[key]['input_check_pass'] or hashlib.sha256(data).hexdigest()!=checks[key]['sha256']:raise RuntimeError('Input validation mismatch')
            shutil.copy2(path,folder/(key+'.wav'))
            with wave.open(str(path),'rb') as reader:audio=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
            audio=np.pad(audio,(0,24000));front=DuplexFrontend(aec=False);front.warmup();ends=[]
            for offset in range(0,len(audio)-320,320):
                _,_,ended,_=front.update(audio[offset:offset+320])
                if ended:ends.append(offset+320);front.reset()
            for index,end in enumerate(ends):
                prefix=audio[:end];window=prefix[-128000:];window=np.pad(window,(128000-len(window),0))
                started=time.perf_counter()
                features=extractor(window,sampling_rate=16000,return_tensors='np',padding='max_length',max_length=128000,truncation=True,do_normalize=True).input_features.astype(np.float32)
                feature_time=time.perf_counter()-started;started=time.perf_counter()
                probability=float(np.asarray(session.run(None,{'input_features':features})[0]).ravel()[0]);inference_time=time.perf_counter()-started
                desired=index==len(ends)-1
                row=dict(key=key,prefix_end_s=end/16000,predicted_complete=probability>.5,desired_complete=desired,
                    probability_complete=probability,passed=(probability>.5)==desired,feature_ms=feature_time*1000,inference_ms=inference_time*1000)
                rows.append(row);output.write(json.dumps(row)+'\n');output.flush();print(json.dumps(row),flush=True)
    report=dict(cases=rows,passed=sum(r['passed'] for r in rows),total=len(rows),
        premature_completion_cases=[r for r in rows if r['predicted_complete'] and not r['desired_complete']],
        missed_completion_cases=[r for r in rows if not r['predicted_complete'] and r['desired_complete']],deployed=False,
        packages={name:version(name) for name in ['onnxruntime','transformers','torch']},
        scope='Clean-speech prefixes, official 0.5 threshold and input preparation; intended full quoted script supplies labels. Not a broad endpoint benchmark or proof of natural conversational intent.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=report['passed'],total=len(rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    if (args.folder/'probe-started.json').exists():parser.error('Use a new experiment folder')
    try:run(args.folder)
    except Exception:
        import traceback
        (args.folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:print('Verified backup:',checkpoint(args.folder),flush=True)
