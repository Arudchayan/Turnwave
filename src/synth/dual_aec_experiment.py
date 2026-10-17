"""Offline prototype: linear AEC audio with post-filter residual onset veto."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from duplex import DuplexFrontend
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint


def h3_acceptance(cases,floor):
    """Judge the candidate's actual floor, rather than a safer control."""
    worst={}
    for case in cases:
        if case['residual_floor']==floor:
            delay=case['configured_delay_ms']
            worst[delay]=max(worst.get(delay,0),len(case['quiet_h3_onsets']))
    return dict(false_onsets=sum(worst.values()),validated={40,80}.issubset(worst))


def make_candidate(folder,extension,echo_aware=False,capture_postfiltered=False,minimum_guard_correlation=None,correlation_window_ms=100,residual_floor=.005):
    if residual_floor<0:raise ValueError('Residual floor must be nonnegative')
    if minimum_guard_correlation is not None and not 0<=minimum_guard_correlation<=1:
        raise ValueError('Guard correlation must be between zero and one')
    if not 100<=correlation_window_ms<=500:raise ValueError('Correlation window must be 100–500 ms')
    window=round(correlation_window_ms*16/4)*4
    spec=importlib.util.spec_from_file_location('_webrtc_audio_linear',extension.resolve())
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    source=(ROOT/'src/duplex.py').read_text()
    if window!=1600:
        if source.count('[-1600:]')!=3:raise RuntimeError('Correlation history transform no longer matches source')
        source=source.replace('[-1600:]',f'[-{window}:]')
    replacements=[
        ('            self._raw_vad_buf = np.zeros(0, np.float32)', '            self._raw_vad_buf = np.zeros(0, np.float32)\n            self._guard_vad_buf = np.zeros(0, np.float32)'),
        ('        raw_out = []','        raw_out = []\n        guard_out = []'),
        ('            raw_out.append(near.copy())','            raw_out.append(near.copy())\n            guard = near.copy()'),
        ('near = np.asarray(self.aec.process(near, far)).ravel().astype(np.float32)',
         'near = np.asarray(self.aec.process(near, far)).ravel().astype(np.float32)\n                    guard = self.aec.postfiltered.copy()'),
        ('            out.append(near)','            out.append(near)\n            guard_out.append(guard)'),
        ('        self._raw_vad_buf = np.concatenate((self._raw_vad_buf, raw_clean))',
         '        self._raw_vad_buf = np.concatenate((self._raw_vad_buf, raw_clean))\n        self._guard_vad_buf = np.concatenate((self._guard_vad_buf, np.concatenate(guard_out) if guard_out else np.zeros(0, np.float32)))'),
        ('            self._advance_history(raw_chunk, far_chunk, chunk)','            guard_chunk, self._guard_vad_buf = self._guard_vad_buf[:F_VAD], self._guard_vad_buf[F_VAD:]\n            self._advance_history(raw_chunk, far_chunk, chunk)'),
        ('np.sqrt(np.mean(chunk ** 2))','np.sqrt(np.mean(guard_chunk ** 2))')]
    # Init is indented eight spaces, reset twelve; preserve each line's indent.
    before='        self._raw_vad_buf = np.zeros(0, np.float32)\n        self._far_vad_buf'
    if before not in source:raise RuntimeError('Research guard initialization no longer matches source')
    source=source.replace(before,'        self._raw_vad_buf = np.zeros(0, np.float32)\n        self._guard_vad_buf = np.zeros(0, np.float32)\n        self._far_vad_buf',1)
    for before,after in replacements:
        if before not in source:raise RuntimeError('Research transform no longer matches source: '+before)
        # The remaining raw-buffer initialization is the reset method.
        source=source.replace(before,after,1)
    if echo_aware:
        source=source.replace('clean_rms < raw_rms * .2',
            '(clean_rms < raw_rms * .2 or getattr(self, "raw_echo_score", 0.0) >= .6)',1)
    if minimum_guard_correlation is not None:
        before='if clean_rms < self.residual_floor and '
        if before not in source:raise RuntimeError('Residual guard transform no longer matches source')
        source=source.replace(before,
            f'if getattr(self, "raw_echo_score", 0.0) >= {minimum_guard_correlation!r} and clean_rms < self.residual_floor and ',1)
    if capture_postfiltered:
        source=source.replace('return self.speaking, started, ended, clean',
            'return self.speaking, started, ended, np.concatenate(guard_out) if guard_out else np.zeros(0, np.float32)',1)
        source=source.replace('self.level = float(np.sqrt(np.mean(clean ** 2)))',
            'self.level = float(np.sqrt(np.mean(np.concatenate(guard_out) ** 2)))',1)
    generated=folder/'duplex_dual.py';generated.write_text(source)
    spec=importlib.util.spec_from_file_location('duplex_dual',generated.resolve())
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    instances=[]
    class Dual:
        def __init__(self,delay):
            self.processor=native.AudioProcessor(sample_rate=16000,num_channels=1,echo_cancellation=True,stream_delay_ms=delay)
            self.history=np.zeros(0,np.float32);self.records=[]
        def process(self,near,far):
            self.postfiltered=self.processor.process(near,far)
            linear=self.processor.linear_output
            self.history=np.concatenate((self.history,linear))[-window:]
            self.records.append((near.copy(),far.copy(),linear.copy(),self.postfiltered.copy()))
            return linear
    class Candidate(module.DuplexFrontend):
        def __init__(self,**kwargs):
            kwargs.setdefault('residual_floor',residual_floor)
            kwargs['aec']=False;super().__init__(**kwargs)
            self.aec=Dual(kwargs.get('stream_delay_ms',40));instances.append(self.aec)
        def _echo_similarity(self):
            self.raw_echo_score=module.DuplexFrontend._echo_similarity(self)
            return module.DuplexFrontend._echo_similarity(SimpleNamespace(_near_history=self._clean_history,_far_history=self._far_history))
    return Candidate,instances


def run(folder,extension,human,echo_aware=False,capture_postfiltered=False,minimum_guard_correlation=None,correlation_window_ms=100,residual_floor=.005):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    Candidate,instances=make_candidate(folder,extension,echo_aware,capture_postfiltered,minimum_guard_correlation,correlation_window_ms,residual_floor)
    from synth import acoustic_stress,replay_human_acoustics
    acoustic_stress.DuplexFrontend=Candidate;replay_human_acoustics.DuplexFrontend=Candidate
    try:
        acoustic_stress.run(folder/'acoustic.json')
        replay_human_acoustics.main(human,floor=residual_floor,filename='h3.json',output_folder=folder)
    finally:
        for index,instance in enumerate(instances):
            if instance.records:
                values=[np.concatenate([r[i] for r in instance.records]) for i in range(4)]
                np.savez_compressed(folder/f'aec-{index:02d}.npz',near=values[0],far=values[1],linear=values[2],postfiltered=values[3])
    acoustic=json.loads((folder/'acoustic.json').read_text());h3=json.loads((folder/'h3.json').read_text())
    passed=sum(r['passed'] for r in acoustic['cases']);actual_h3=h3_acceptance(h3['cases'],residual_floor)
    decision=dict(acoustic_passed=passed,acoustic_total=len(acoustic['cases']),default_h3=actual_h3,
        decision='rejected; not deployed' if passed!=22 or actual_h3['false_onsets'] or not actual_h3['validated'] else 'preliminary pass; held-out validation required',deployed=False,
        policy=('Linear audio/correlation; low nonlinear residual veto uses either strong suppression or raw echo correlation.' if echo_aware
                else 'Linear audio and linear-correlation gate; original nonlinear residual energy veto, same thresholds.'),
        echo_aware_guard=echo_aware,
        capture_postfiltered=capture_postfiltered,
        minimum_guard_correlation=minimum_guard_correlation,
        correlation_window_ms=correlation_window_ms,
        default_residual_floor=residual_floor,
        limitation='Offline synthetic and recorded controls only. Generated research frontend is separate from production.')
    (folder/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('extension',type=Path);parser.add_argument('human',type=Path);parser.add_argument('--echo-aware-guard',action='store_true');parser.add_argument('--capture-postfiltered',action='store_true');parser.add_argument('--minimum-guard-correlation',type=float);parser.add_argument('--correlation-window-ms',type=int,default=100);parser.add_argument('--residual-floor',type=float,default=.005);args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder,args.extension,args.human,args.echo_aware_guard,args.capture_postfiltered,args.minimum_guard_correlation,args.correlation_window_ms,args.residual_floor)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
