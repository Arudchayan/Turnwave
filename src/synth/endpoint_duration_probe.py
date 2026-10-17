"""Probe long/short turn timeout restoration with controlled VAD decisions.

This checks state/timing invariants, not acoustic or model quality.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def run(folder,source):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    spec=importlib.util.spec_from_file_location('duration_frontend',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    front=module.DuplexFrontend(aec=False,holdoff_s=0);front._prob=lambda x:float(np.mean(x)>.02)
    rows=[]
    def feed(seconds):
        peak=front.end_chunks
        for _ in range(round(seconds*100)):
            front.update(np.full(160,.1,np.float32));peak=max(peak,front.end_chunks)
        return peak
    for duration in [3.2,.7,3.2,.7]:
        peak=feed(duration);begin=front._sample_clock
        for _ in range(200):
            _,_,ended,_=front.update(np.zeros(160,np.float32))
            if ended:break
        else:raise AssertionError('No endpoint')
        delay=(front._sample_clock-begin)/16
        expected=1120 if duration>2 else 480
        passed=abs(delay-expected)<=32 and front.end_chunks==15 and front._utterance_start_clock is None
        rows.append(dict(speech_s=duration,quiet_delay_ms=delay,expected_ms=expected,peak_end_chunks=peak,passed=passed))
    feed(3.2);front.reset();assert front.end_chunks==15 and front._utterance_start_clock is None
    report=dict(cases=rows,passed=all(r['passed'] for r in rows),active_reset_restores_short_timeout=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Controlled VAD state invariant only, not an acoustic acceptance test.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('source',type=Path);a=p.parse_args()
    try:passed=run(a.folder,a.source)
    finally:
        if a.folder.exists():print('Verified backup:',checkpoint(a.folder),flush=True)
    raise SystemExit(0 if passed else 1)
