"""Save the production frontend's acoustic suite and recorded quiet-H3 replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from synth import acoustic_stress,replay_human_acoustics
from synth.archive_experiment import capture_files
from synth.checkpoint_experiment import checkpoint
from synth.dual_aec_experiment import h3_acceptance


def run(folder,human,frontend_source=None,trace_h3=False):
    folder.mkdir(parents=True,exist_ok=False);capture_files(folder/'source')
    if frontend_source:
        shutil.copy2(frontend_source,folder/'frontend.py')
        spec=importlib.util.spec_from_file_location('gate_duplex',frontend_source.resolve())
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        acoustic_stress.DuplexFrontend=module.DuplexFrontend;replay_human_acoustics.DuplexFrontend=module.DuplexFrontend
    acoustic_stress.run(folder/'acoustic.json')
    traces=[]
    if trace_h3:
        Base=replay_human_acoustics.DuplexFrontend
        class TracedFrontend(Base):
            def __init__(self,**kwargs):
                super().__init__(**kwargs);self.frame_trace=[];traces.append(self)
            def _prob(self,chunk):
                value=super()._prob(chunk)
                if self._sample_clock and len(self._near_history)>=512:
                    clean=float(np.sqrt(np.mean(chunk**2)));raw=float(np.sqrt(np.mean(self._near_history[-512:]**2)))
                    self.frame_trace.append(dict(end_s=self._sample_clock/16000,probability=value,clean_rms=clean,raw_rms=raw,
                        echo_score=self.echo_score,far_active=self._far_active,speaking=self.speaking,hot_before=self._hot,
                        residual_floor=self.residual_floor,configured_delay_ms=self.aec.stream_delay_ms if self.aec else None,
                        residual_veto=bool(self._far_active and clean<self.residual_floor and clean<raw*.2),
                        correlation_veto=bool(self._far_active and self.echo_score>=.6)))
                return value
        replay_human_acoustics.DuplexFrontend=TracedFrontend
    replay_human_acoustics.main(human,filename='h3.json',output_folder=folder)
    for index,front in enumerate(traces):
        (folder/f'h3-trace-{index}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in front.frame_trace))
    acoustic=json.loads((folder/'acoustic.json').read_text());h3=json.loads((folder/'h3.json').read_text())
    report=dict(acoustic_passed=sum(c['passed'] for c in acoustic['cases']),acoustic_total=len(acoustic['cases']),
        default_h3=h3_acceptance(h3['cases'],.005),deployed=False,
        frontend_sha256=hashlib.sha256((frontend_source or ROOT/'src/duplex.py').read_bytes()).hexdigest(),
        scope='Current source offline only. Synthetic onset/echo checks and one recorded human quiet section; not a fresh physical-room test or answer-quality gate.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('human',type=Path);parser.add_argument('--frontend-source',type=Path);parser.add_argument('--trace-h3',action='store_true');args=parser.parse_args()
    if args.folder.exists():parser.error('Use a new experiment folder')
    try:run(args.folder,args.human,args.frontend_source,args.trace_h3)
    except Exception:
        if args.folder.exists():
            import traceback
            (args.folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.folder.exists():print('Verified backup:',checkpoint(args.folder),flush=True)
